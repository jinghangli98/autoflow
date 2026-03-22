#!/bin/bash                                     
#SBATCH --job-name=ddp-1000           
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_multi
#SBATCH --mail-user=jil202@pitt.edu    
#SBATCH --mail-type=END,FAIL               
#SBATCH --time=0-72:00:00
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

echo "MASTER_NODE=$MASTER_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "Python: $(which python)"
echo "ulimit -l: $(ulimit -l)"

srun python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=3 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_flow.py --lr 3e-4 --batch_size 4 --crop_size 384 --resize_size 384 --max_epochs 100 --save_model --fp16 --sample 100 --distributed --world_size 12 --compile --checkpoint_path /ix3/tibrahim/jil202/cfg_gen/src/training_mprage2mp2rage/training_2d/autoregressive/autoregressive_ssim_8.01_epoch_3_1000.pt

#python -m torch.distributed.run --nproc_per_node=4 train_flow.py   --distributed --fp16 --save_model --compile   --batch_size 2 --max_epochs 100 --save_model --fp16 --sample 10 --checkpoint_path ./checkpoints/flow_matching_best_epoch_38.pt
