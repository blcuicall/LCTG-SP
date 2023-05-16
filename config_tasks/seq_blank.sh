EXPERIMENT_NAME=${MODEL_TYPE}-blank-epoch10  #${MASK_RATIO}
TASK_NAME=blank
DATA_PATH="${DATA_ROOT}/one_billion_word_1kw/multi_mask_data"
TRAIN_ARGS="--epochs 10 \
            --batch-size 16 \
            --lr 2e-4 \
            --lr-decay-style cosine \
            --lr-decay-iters 160000 \
            --lr-decay-ratio 0.05 \
            --warmup 0.05 \
            --weight-decay 1.0e-1
            --label-smoothing 0.1 \
            --save-epoch 2"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

TASK_ARGS="--src-seq-length 32 \
           --tgt-seq-length 256 \
           --min-tgt-length 0 \
           --length-penalty 1 \
           --no-repeat-ngram-size 3 \
           --eval-batch-size 8"