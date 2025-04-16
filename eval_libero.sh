#!/bin/bash
# Check if GPU ID is provided, default to 0 if not
# GPU_ID=${1:-0}

# Set visible GPU
# export CUDA_VISIBLE_DEVICES=$GPU_ID

# # Activate conda environment
eval "$(conda shell.bash hook)"
conda activate videogeneration

# Run the evaluation script
# python -m scripts.eval_ik_pid_feedback
python -m scripts.eval_ik_proprio

# Optional: you can add date to the command output
echo "Evaluation completed at $(date)"