DATA_ROOT=/data/private/njr/workspace/glm/data
CHECKPOINT_PATH=/data/private/njr/workspace/glm/data/checkpoints
SAVE_PATH=/data/private/njr/workspace/glm/data/finetune_checkpoints/
DATESTR=$(date +"%m-%d-%H-%M")

MASK_RATIO=0.95

source $1    # Model
source $2    # Task

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --checkpoint-activations \
       --overwrite \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt