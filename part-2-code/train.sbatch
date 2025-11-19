#!/bin/bash
#SBATCH --job-name=t5_train
#SBATCH --output=logs/train_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=5:00:00

source ~/.bashrc
conda activate hw4-part-1-nlp

mkdir -p logs

python train_t5.py \
    --finetune \
    --batch_size 8 \
    --test_batch_size 16 \
    --learning_rate 3e-4 \
    --max_n_epochs 20 \
    --patience_epochs 8 \
    --scheduler_type cosine \
    --experiment_name ft_exp_1118_04
