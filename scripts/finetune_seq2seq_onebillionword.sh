DATA_ROOT=/data/private/njr/workspace/glm/data
CHECKPOINT_PATH=/data/private/njr/workspace/glm/data/checkpoints
SAVE_PATH=/data/private/njr/workspace/glm/data/finetune_checkpoints/mini-billion
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=en,eth,em,bond
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
# export PATH="/home/lx/anaconda3/envs/GLM_env/bin:$PATH"
# export PATH="/usr/local/cuda-10.1/bin:$PATH" ##NCCL PATH

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="./scripts/hostfile" #"/root/code/config/hostfile"

mkdir logs
deepspeed --include localhost:0,1,2,3 finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --checkpoint-activations \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       2>&1 | tee logs/log-${DATESTR}.txt
