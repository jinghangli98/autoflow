#!/bin/bash
#SBATCH --job-name=baseline-l1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_multi
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=0

# Properly activate conda environment
eval "$(conda shell.bash hook)"
conda activate vsr

# Fix IB memory registration limit
ulimit -l unlimited

# Get master address as IPv4
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(getent ahostsv4 $MASTER_NODE | head -n 1 | awk '{print $1}')
export MASTER_PORT=29500

# Force IPv4, auto-detect interface
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=^lo,docker0
export GLOO_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=INFO

# Derived constants
GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "MASTER_NODE:   $MASTER_NODE"
echo "MASTER_ADDR:   $MASTER_ADDR"
echo "NODELIST:      $SLURM_JOB_NODELIST"
echo "NNODES:        $NNODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "WORLD_SIZE:    $WORLD_SIZE"
echo "Python:        $(which python)"
echo "ulimit -l:     $(ulimit -l)"
echo "============================================"

srun python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_baseline.py \
        --distributed \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --lr 1e-5 \
        --batch_size 3 \
        --max_epochs 100 \
        --save_model \
        --fp16 \
        --sample 100 \
        --compile \
        --ema_start_epoch 999
