#!/bin/bash
source ~/mambaforge/etc/profile.d/conda.sh

# Activate conda environment
conda activate video_gen

# Create results directory if it doesn't exist
mkdir -p ./results/ik_policy

# Run the evaluation script
python -m scripts.eval_ik_pid_feedback

# Optional: you can add date to the command output
echo "Evaluation completed at $(date)"
