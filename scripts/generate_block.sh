#!/bin/bash
#CHECKPOINT_PATH=/data/private/njr/workspace/glm/data/finetune_checkpoints/generation-large-billion/     #/root/data/checkpoints
CHECKPOINT_PATH=/data/private/njr/workspace/glm/data/finetune_checkpoints/blank-large-blank-0.8
DATESTR=$(date +"%m-%d-%H-%M")
source $1
#export CUDA_VISIBLE_DIVICES=9
MPSIZE=1
MAXSEQLEN=32  # 512
# MAXSEQLEN=513
# MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

# MASTER_PORT=${MASTER_PORT} python generate_samples.py \
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch  --master_port 29504  generate_test_njr.py \
       --model-parallel-size $MPSIZE \
       --deepspeed_config ${config_json} \
       $MODEL_ARGS \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --fp16 \
       --cache-dir cache \
       --num-beams 5 \
       --length-penalty 1.0 \
       --out-seq-length $MAXSEQLEN \
       --seq-length 512 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       2>&1 | tee logs/generate-log-${DATESTR}.txt
