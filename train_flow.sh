#!/bin/bash                                     
#SBATCH --job-name=ddp-1000           
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --mail-user=jil202@pitt.edu    
#SBATCH --mail-type=END,FAIL               
#SBATCH --time=0-98:00:00
#SBATCH --gres=gpu:8

# Properly activate conda environment
source activate vsr

python -m torch.distributed.run --nproc_per_node=8 train_flow.py \
        --contrast mprage mp2rage flair tse swi \
        --data_root /ix1/tibrahim/jil202/studies/dataset_grappa_nii \
        --distributed --fp16 --save_model --compile \
        --batch_size 2 --max_epochs 100 --sample 100 \
        --num_sampling_steps 2 --samples_per_contrast 0 --checkpoint_path /ix1/tibrahim/jil202/autoflow/checkpoints/flow_matching_3d_mprage.pt