#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --job-name=lean_eval
#SBATCH --output=./evaluation_miniF2F/lean_eval_%A_%a.out
#SBATCH --error=./evaluation_miniF2F/lean_eval_%A_%a.err
#SBATCH --array=1-61

# Load required modules
module load gcc arrow/15.0.1 opencv/4.11.0

# Activate Lean virtual environment
source ~/lean_env/bin/activate

# Calculate row ranges (4 rows per task)
START=$(( (SLURM_ARRAY_TASK_ID - 1) * 4 + 1 ))
END=$(( START + 3 ))

# Compute the log suffix (task_id * 1000)
LOG_SUFFIX=$(printf "%06d" $((SLURM_ARRAY_TASK_ID * 1000)))

# Run evaluation
python ./llm_inference/evaluate.py \
  --input_jsonl ./results_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithEg_output.jsonl \
  --startRow $START --endRow $END --offset 0\
| tee ./evaluation_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithEg_evalLOG-${LOG_SUFFIX}.txt

# python ./llm_inference/evaluate.py \
#   --input_jsonl ./results_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithoutEgv2_output.jsonl \
#   --startRow $START --endRow $END --offset 150000\
# | tee ./evaluation_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithoutEgv2_evalLOG-${LOG_SUFFIX}.txt
