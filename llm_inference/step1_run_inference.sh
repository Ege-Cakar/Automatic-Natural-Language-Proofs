#!/bin/bash
#SBATCH --account=def-vganesh          # Your PI's account
#SBATCH --gpus-per-node=h100:1         # Request 1 H100 GPU (use :2 if needed)
#SBATCH --cpus-per-task=8              # Adjust CPU cores (for dataloader/parallelism)
#SBATCH --mem=128000
#SBATCH --time=0-11:59                # Walltime (2 days, adjust as needed)
#SBATCH --job-name=vllm4
#SBATCH --output=./results_miniF2F/vllm4_%j.out
#SBATCH --error=./results_miniF2F/vllm4_%j.err

# Load required modules
module load gcc arrow/15.0.1 opencv/4.11.0

# Activate Lean virtual environment
source ~/lean_env/bin/activate

nvidia-smi

PORT=$(shuf -i 10000-20000 -n 1)
echo "Using port $PORT"

nohup vllm serve AI-MO/Kimina-Prover-RL-1.7B --port $PORT --tensor-parallel-size 1 --max_model_len 40960 &

echo "Starting sleep..."; sleep 5m; echo "Done sleeping"

#----INFERENCE WITH IN-CONTEXT EXAMPLES
python ./llm_inference/gpu_inference_Kimina-Prover-RL-1-7B.py \
  --port $PORT \
  --num_samples_per_task 64 \
  --model_id "AI-MO/Kimina-Prover-RL-1.7B" \
  --method_tag "OffTheShelfWithEg" \
  --eval_dir "results_miniF2F" \
  --dataset_path "./datasets_validation/minif2f/dataset.jsonl" \
  --use_examples_in_prompt 1

#----INFERENCE WITHOUT IN-CONTEXT EXAMPLES
# python ./llm_inference/gpu_inference_Kimina-Prover-RL-1-7B.py \
#   --port $PORT \
#   --num_samples_per_task 64 \
#   --model_id "AI-MO/Kimina-Prover-RL-1.7B" \
#   --method_tag "OffTheShelfWithoutEg" \
#   --eval_dir "results_miniF2F" \
#   --dataset_path "./datasets_validation/minif2f/dataset.jsonl" \
#   --use_examples_in_prompt 0

#cd ai_for_math/
#sbatch /home/pjana/projects/def-vganesh/pjana/ai_for_math/llm_inference/run_inference.sh

