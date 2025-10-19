#!/bin/bash
#SBATCH --account=def-vganesh          # Your PI's account
#SBATCH --gpus-per-node=h100:1         # Request 1 H100 GPU (use :2 if needed)
#SBATCH --cpus-per-task=8              # Adjust CPU cores (for dataloader/parallelism)
#SBATCH --mem=128000
#SBATCH --time=02-00:00                # Walltime (2 days, adjust as needed)
#SBATCH --job-name=vllm
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

# Load required modules
module load gcc arrow/15.0.1 opencv/4.11.0

# Activate Lean virtual environment
source ~/lean_env/bin/activate

nvidia-smi

vllm serve deepseek-ai/DeepSeek-Prover-V2-7B --host 0.0.0.0 --port 8000


