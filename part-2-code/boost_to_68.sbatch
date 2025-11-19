#!/bin/bash
#SBATCH --job-name=t5_boost
#SBATCH --output=logs/boost_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --time=8:00:00

module load anaconda3/2020.07
source activate hw4-part-1-nlp

python train_t5.py \
    --finetune \
    --batch_size 6 \
    --test_batch_size 32 \
    --learning_rate 5e-4 \
    --max_n_epochs 40 \
    --patience_epochs 12 \
    --scheduler_type linear \
    --num_warmup_epochs 3 \
    --experiment_name ft_boost_to_68
