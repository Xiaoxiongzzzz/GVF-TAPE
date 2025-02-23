#!/bin/bash

# conda init


source /mnt/home/zhangchuye/anaconda3/etc/profile.d/conda.sh

# conda init

# Activate conda environment
conda activate video_gen

# Create results directory if it doesn't exist
mkdir -p ./results/ik_policy

# Run the evaluation script
python -m scripts.eval_ik_proprio

# Optional: you can add date to the command output
echo "Evaluation completed at $(date)"
