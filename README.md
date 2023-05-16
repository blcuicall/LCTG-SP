bash scripts/finetune_superglue.sh \
     config_tasks/model_blocklm_roberta_large.sh \
     config_tasks/task_copa.sh
```

- To apply GLM to a new NLU dataset with cloze-filling finetuning, implement a `DataProcessor` in
  [tasks/superglue/dataset.py](tasks/superglue/dataset.py) for data loading and add a `PVP` in 
  [tasks/superglue/pvp.py](tasks/superglue/pvp.py) for the cloze question. More details can be found 
  [here](tasks/superglue/README.md).
  
- The cloze questions (prompts) used in this work are written by human. We are also studying a P-tuning (prompt 
  tuning) approach to search for the optimal continuous prompt. Please refer to our 
  [paper](https://arxiv.org/abs/2103.10385) and [code](https://github.com/THUDM/P-tuning).

### Text Summarization

- Download the [Gigaword](https://github.com/harvardnlp/sent-summary) dataset and check the experiment setup in 
  [scripts/finetune_seq2seq.sh](scripts/finetune_seq2seq.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your 
  local path. 
  
- Run the following script

```
bash scripts/finetune_seq2seq.sh \ 
     config_tasks/model_blocklm_large_generation.sh \ 
     config_tasks/seq_gigaword.sh
```
- For calculating rouge, install file2rouge from [here](https://github.com/pltrdy/files2rouge) and run `bash scripts/evaluate_seq2seq.sh`

### Language Modeling
#### LAMBADA Cloze Accuracy
* Download the [LAMBADA](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl) data and change 
  `DATA_ROOT, CHECKPOINT_PATH` in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
```shell
bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lambada.sh 
```
#### LM Perplexity
* Download our [test set of wikibook](https://mailstsinghuaeducn-my.sharepoint.com/:t:/g/personal/duzx16_mails_tsinghua_edu_cn/EQa_B6KY_q1FjtUeG-T52iMBFtNrfhfHcZbzMxfkJKXKRQ?e=inTdHh) (or any dataset following the same format) and change `DATA_ROOT, CHECKPOINT_PATH` 
  in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
  ```shell
  bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lm.sh 
  ```

### Blank Language Model
- Download the [Yahoo](https://github.com/Varal7/blank_language_model) dataset and check the experiment setup in 
  [scripts/finetune_blank.sh](scripts/finetune_blank.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your 
  local path. 
  
- Run the following script

```
bash scripts/finetune_blank.sh \ 
     config_tasks/model_blocklm_large.sh \ 
     config_tasks/seq_blank.sh
```

### Blank Filling (Interactive)
* Change `CHECKPOINT_PATH` to your local path. Run the following script
```
bash scripts/generate_block.sh \
     config_tasks/model_blocklm_large.sh
```
Example:

Context: Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.

GLM: [CLS] ng is an adjunct professor at [MASK] ( formerly associate professor and director of its stanford ai lab or sail ) . also a pioneer in online education , ng co - founded coursera and deeplearning . ai . [PAD] <|startofpiece|> the stanford university

## Citation
Please cite our paper if you find this code useful for your research:
```
@article{DBLP:journals/corr/abs-2103-10360,
  author    = {Zhengxiao Du and
               Yujie Qian and
               Xiao Liu and
               Ming Ding and
               Jiezhong Qiu and
               Zhilin Yang and
               Jie Tang},
  title     = {All {NLP} Tasks Are Generation Tasks: {A} General Pretraining Framework},
  journal   = {CoRR},
  volume    = {abs/2103.10360},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.10360}
}
```
