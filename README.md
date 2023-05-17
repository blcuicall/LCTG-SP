## environment (requirements.txt)
apex 0.1

pytorch 1.7.0

...


## data process


## train

pretrain
```
bash scripts/pretrain_blank.sh
```

fintune
```
bash scripts/finetune_seq2seq_onebillionword.sh \ 
     config_tasks/model_blocklm_large.sh \ 
     config_tasks/seq_blank.sh
```

## evaluate
```
bash scripts/evaluate_seq2seq.sh
```

