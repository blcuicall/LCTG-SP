# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import torch
import torch.nn.functional as F
import time
from datetime import datetime
from arguments import get_args
from pretrain_glm import initialize_distributed
from pretrain_glm import set_random_seed
from pretrain_glm import get_masks_and_position_ids
from utils import load_checkpoint, get_checkpoint_iteration
from configure_data import prepare_tokenizer
from generation_utils import BeamSearchScorer
import mpu

from train_utils import get_model


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args, model_type="generation")

    # if args.deepspeed:
    #     print_rank_0("DeepSpeed is enabled.")
    #
    #     model, _, _, _ = deepspeed.initialize(
    #         model=model,
    #         model_parameters=model.parameters(),
    #         args=args,
    #         mpu=mpu,
    #         dist_init_required=False
    #     )
    if args.load_pretrained is not None:
        args.no_load_optim = True
        args.load = args.load_pretrained
        _ = load_checkpoint(
            model, None, None, args)
    # if args.deepspeed:
    #     model = model.module

    return model


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    if args.block_lm:
        attention_mask = torch.tensor([tokens.size(1)], device=device, dtype=torch.long)
        position_ids = torch.arange(tokens.size(1), device=device, dtype=torch.long)
        #print('*****',position_ids)
        if not args.no_block_position:
            block_position_ids = torch.zeros(tokens.size(1), device=device, dtype=torch.long)
            position_ids = torch.stack((position_ids, block_position_ids), dim=0)
            #print('#####', position_ids)
        position_ids = position_ids.unsqueeze(0)
        
    else:
        attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
            tokens,
            args.eod_token,
            reset_position_ids=False,
            reset_attention_mask=False,
            set_loss_mask=False,
            mem_length=args.mem_length)
    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits


def sample_sequence(model, tokenizer, context_tokens, context_length, args, device, mems=None, end_tokens=None):
    if not args.block_lm:  # False
        context_tokens, attention_mask, position_ids = get_batch(context_tokens, device, args)
        tokens = torch.empty((args.num_beams, 0), device=context_tokens.device, dtype=torch.long)
    else:
        tokens = context_tokens.new_full((1, 1), tokenizer.get_command('sop').Id)  # [[30502]]
    counter = 0
    if mems is None:  # False
        mems = []
    if end_tokens is None:  # False
        end_tokens = [args.eod_token]
    if args.num_beams > 1:  # True
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            max_length=args.out_seq_length,
            num_beams=args.num_beams,
            device=context_tokens.device,
            length_penalty=args.length_penalty,
            do_early_stopping=False,
        )
        beam_scores = torch.zeros(1, dtype=torch.float, device=context_tokens.device)  # [0.]
    last_beam_num = 1
    while counter < args.out_seq_length-1:
        if counter == 0 and not args.block_lm:  # False
            next_token_logits, *mems = model(context_tokens, position_ids, attention_mask, *mems)
        else:
            if args.block_lm:  # True
                if args.no_block_position:  # False
                    position_ids = context_tokens.new_full((last_beam_num, 1), context_length + counter)
                else:
                    position_ids = context_tokens.new_ones(last_beam_num, 2, 1)  # [[[1], [1]]]
                    position_ids[:, 0] = context_length  # [[[3], [1]]]
                    position_ids[:, 1] = counter + 1  # [[[3], [1]]]
                attention_mask = context_tokens.new_zeros([1], device=context_tokens.device, dtype=torch.long)  # [0]
            else:
                position_ids = context_tokens.new_ones((last_beam_num, 1)) * (context_length + counter - 1)
                attention_mask = context_tokens.new_ones(last_beam_num, 1, 1, args.mem_length + 1,
                                                         device=context_tokens.device, dtype=torch.float)
            last_token = tokens[:, -1:]  # [[30522]]
            next_token_logits, *mems = model(last_token, position_ids, attention_mask, *mems)  # logit [[[* * * ...]]]  [1, 1, 30592]
        next_token_logits = next_token_logits[:, -1]  # [1, 30592]
        if args.num_beams > 1:  # True
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # [1, 30592]
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)  # [1, 30592]
            vocab_size = next_token_scores.shape[-1]  # 30592
            next_token_scores = next_token_scores.view(1, last_beam_num * vocab_size)  # [1, 30592]

            probs = F.softmax(next_token_scores, dim=-1)  # [1, 30592]
            next_tokens = torch.multinomial(probs, num_samples=2 * args.num_beams)  # [1, 10]  #  [[30523,  2171,  2087, 22200,  2190,  2047, 23497, 13940, 15580, 26826]]
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)   # [1, 10]  # [[ -0.1554,  -3.8038,  -6.3507, -11.9161,  -2.6710,  -6.2726, -13.8187, -13.8422,  -7.9718,  -8.8781]]
            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)  # _indices  [[0, 4, 1, 5, 2, 8, 9, 3, 6, 7]]
            next_tokens = torch.gather(next_tokens, -1, _indices)  # [[30523,  2190,  2171,  2047,  2087, 15580, 26826, 22200, 23497, 13940]]

            next_indices = next_tokens // vocab_size  # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            next_tokens = next_tokens % vocab_size  # [[30523,  2190,  2171,  2047,  2087, 15580, 26826, 22200, 23497, 13940]]
            # stateless
            tokens = tokens.expand((args.num_beams, -1))  # [[30533],[30533],[30533],[30533],[30533]]  # num_beam = 5
            beam_outputs = beam_scorer.process(
                tokens,
                next_token_scores,
                next_tokens,
                next_indices,
                eos_token_id=end_tokens,
                mems=mems
            )
            """
            beam_outputs
            {
                'next_beam_scores': tensor([-2.6710, -3.8038, -6.2726, -6.3507, -7.9718], device='cuda:0'), 
                'next_beam_tokens': tensor([ 2190,  2171,  2047,  2087, 15580], device='cuda:0'), 
                'next_beam_indices': tensor([0, 0, 0, 0, 0], device='cuda:0')}
            """

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            beam_next_tokens = beam_next_tokens.unsqueeze(-1)  # [ [2190],  [2171],  [2047],  [2087], [15580]]
            tokens = torch.cat([tokens[beam_idx, :], beam_next_tokens], dim=-1)  # [[30522,  2190], [30522,  2171], [30522,  2047], [30522,  2087], [30522, 15580]]
            mems = [mem[beam_idx] for mem in mems] if mems else None
            if beam_scorer.is_done:  # False
                break
            last_beam_num = args.num_beams # 5
        else:
            next_token_logits /= args.temperature
            next_token_logits = top_k_logits(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(next_token_logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)[0]
            is_end = prev.item() in end_tokens
            if is_end:
                break
            prev = prev.view(1, 1)
            tokens = prev if tokens is None else torch.cat((tokens, prev), dim=1)
        counter += 1
        if not args.block_lm and mpu.get_model_parallel_rank() == 0 and counter % 16 == 0:  # False
            output_tokens_list = tokens.view(-1).contiguous()
            decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
            if mpu.get_model_parallel_rank() == 0 and (counter % 128 == 0 or is_end):
                #os.system('clear')
                trim_decode_tokens = decode_tokens
                print(trim_decode_tokens, flush=True)
    if args.num_beams > 1:
        tokens, mems = beam_scorer.finalize(tokens, beam_scores, next_tokens, next_indices, eos_token_id=args.eod_token,
                                            mems=mems)
    return torch.cat((context_tokens, tokens), dim=1), mems


def read_context(tokenizer, args, output, input_sent):
    
    terminate_runs, skip_run = 0, 0
    if mpu.get_model_parallel_rank() == 0:
        while True:
        #with open(input_path, 'r', encoding='utf-8') as f_input:
            raw_text = input_sent # input("\nContext prompt (stop to exit) >>> ")
            if not raw_text:
                print('Prompt should not be empty!')
                continue
            if raw_text == "stop":
                terminate_runs = 1
                break
            generation_mask = '[gMASK]' if args.task_mask else '[MASK]'   # [mask]
            if args.block_lm and 'MASK]' not in raw_text:  # False
                raw_text += ' ' + generation_mask
            
            output.write(raw_text)
            context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization  # return list
        
            if args.block_lm:
                context_tokens = [tokenizer.get_command('ENC').Id] + context_tokens
                
                if not raw_text.endswith('MASK]'):  # False
                    context_tokens = context_tokens + [tokenizer.get_command('eos').Id]
            
            context_length = len(context_tokens)

            if context_length >= args.seq_length:
                print("\nContext length", context_length,
                      "\nPlease give smaller context than the window length!")
                continue
            break
    else:
        context_length = 0

    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    terminate_runs = terminate_runs_tensor[0].item()

    if terminate_runs == 1:
        return terminate_runs, raw_text, None, None

    context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    context_length = context_length_tensor[0].item()
    if mpu.get_model_parallel_rank() == 0:
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    else:
        context_tokens_tensor = torch.cuda.LongTensor([0] * context_length)
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    return terminate_runs, raw_text, context_tokens_tensor, context_length


def generate_samples(model, tokenizer, args, device):
    model.eval()
    output_path = "./samples/blank/test_in_train_0.8"  
    input_path = '/data/private/njr/workspace/glm/data/one_billion_word/test_data/test_in_train_100.source' # add
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, f"sample-{datetime.now().strftime('%m-%d-%H-%M')}.txt")
    count = 0
    with torch.no_grad(), open(output_path, "w") as output, open(input_path, 'r', encoding='utf-8') as f_input:
        for line in f_input: # add
            count += 1
            if count%300 == 0:
                print('finished count:', count)
            input_sent = line.strip() # add
            torch.distributed.barrier(group=mpu.get_model_parallel_group())

            print('input sent **************************\n', input_sent)
            terminate_runs, raw_text, context_tokens_tensor, context_length = read_context(tokenizer, args, output, input_sent) # add
            # contxt_tokens_tensor = [input_length]
            if terminate_runs == 1:
                return
            start_time = time.time()
            if args.block_lm:
                mems = []
                tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
                # tokens = [[input_length]] = [batch * input_length]
                mask_tokens = ['MASK', 'sMASK', 'gMASK'] if args.task_mask else ['MASK']  # ['MASK']
                # print('mask tokens:::::', mask_tokens)
                mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]  # [103]
                end_tokens = [tokenizer.get_command('eop').Id, args.eod_token]  # [30523, 0]
                # print(mask_tokens)
                # print(end_tokens)
                mask_positions = []
                for token in mask_tokens:
                    mask_positions += (context_tokens_tensor == token).nonzero(as_tuple=True)[0].tolist()
                mask_positions.sort()  # [1, 3]
                if args.no_block_position:  # False
                    for mask_position in mask_positions:
                        position_ids[0, mask_position + 1:] += args.out_seq_length
                _, *mems = model(tokens, position_ids, attention_mask, *mems)   # _ = [1, 4, 30592]  mems = [1, 4, 1024]
                for mask_position in mask_positions:
                    if args.no_block_position:  # False
                        position = position_ids[0, mask_position].item()
                    else:
                        position = mask_position
                    tokens, mems = sample_sequence(model, tokenizer, tokens, position,
                                                   args, device, mems=mems, end_tokens=end_tokens)
            else:
                tokens, _ = sample_sequence(model, tokenizer,
                 context_tokens_tensor, context_length, args, device)
            output_tokens_list = tokens.view(-1).contiguous()
            if mpu.get_model_parallel_rank() == 0:
                # os.system('clear')
                # print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                # print("\nContext:", raw_text, flush=True)
                decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
                trim_decode_tokens = decode_tokens
                # print("\nGLM:", trim_decode_tokens, flush=True)
                print('output sent ##############################\n', trim_decode_tokens)
                output.write(trim_decode_tokens + "\n")

            torch.distributed.barrier(group=mpu.get_model_parallel_group())


def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1

    # generate samples
    generate_samples(model, tokenizer, args, torch.cuda.current_device())


if __name__ == "__main__":
    main()
