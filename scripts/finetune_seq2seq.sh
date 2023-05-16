DATA_ROOT=/home/lx/projects/GLM/data
CHECKPOINT_PATH=/home/lx/projects/GLM/data/checkpoints
SAVE_PATH=/home/lx/projects/GLM/data/finetune_checkpoints
DATESTR=$(date +"%m-%d-%H-%M")

echo $1
source $1    # Model
source $2    # Task

export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=en,eth,em,bond
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
# export PATH="/home/lx/anaconda3/envs/GLM_env/bin:$PATH"
# export PATH="/usr/local/cuda-10.1/bin:$PATH" ##NCCL PATH

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="/home/lx/projects/GLM/hostfile" #"/root/code/config/hostfile"

mkdir logs
deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_glm.py \
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
