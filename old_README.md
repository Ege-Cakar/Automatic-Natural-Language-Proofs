# AI for Math (Auto-Formalization, using Joint Embedding)

## Step 1: Load required modules (reqd. for ComputeCanada)
```bash
module load gcc arrow/15.0.1 opencv/4.11.0
```

## Step 2: Create & activate Python virtual environment

(do this only once; afterward you just source ~/lean_env/bin/activate)
```bash
cd ./LEAN_interaction
python -m venv ~/lean_env
source ~/lean_env/bin/activate
```

## Step 3: Install Python dependencies
```bash
pip install --upgrade pip
pip install pandas tqdm datasets lockfile rapidfuzz
pip install poetry
pip install vllm
pip install trl
pip install wandb
poetry lock
poetry install
```

## Step 4: Enter Poetry shell (if using poetry)
```bash
poetry shell
```

## Step 5: Install Lean via elan
```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source $HOME/.elan/env
```


## Step 6: Set default Lean version:
```bash
elan default leanprover/lean4:4.15.0
lean --version  # should print Lean (version 4.15.0)
lake --version  # should work too and print version 4.15.0
```

## Step 7: To activate the venv:
```bash
module load gcc arrow/15.0.1
source ~/lean_env/bin/activate
```

## To evaluate any SoTA tool's auto-formalization performance on miniF2F:

Step-0: Run the steps sequentially, begin each step only after the previous one has finished.

IMPORTANT NOTE: You need to do this only once, not needed for subsequent evaluations of different models.

```bash
./llm_inference/step0_setup.sh
```

Step-1: Next, use this directly for the SoTA LLMs on huggingface, just change the argparse arguments in the .sh file, based on the huggingface model_id you are testing. 

IMPORTANT NOTE: You can schedule multiple instances of this script in parallel.

```bash
sbatch ./llm_inference/step1_run_inference.sh
```

Step-2: Next, in the following script, change the `--input_jsonl` arg, and the name of the LOG file, based on what model you are testing.

IMPORTANT NOTE: Do not schedule multiple instances of this script in parallel, it'll create file conflicts. 

```bash
sbatch ./llm_inference/step2_run_evaluate.sh
```

Step-3: Next, run the following commands. Before that, see the prefix of the 61 `.jsonl` files written here: `./evaluation_miniF2F`. For example, `"./evaluation_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithEg_eval-*.jsonl"`. Accordingly, in the following .py file, change the prefix in the glob.glob arg.

IMPORTANT NOTE: You can schedule multiple instances of this script in parallel.

```bash
module load gcc arrow/15.0.1 opencv/4.11.0
source ~/lean_env/bin/activate
python ./llm_inference/step3_getMetrics.py
```

## Setting up interactive GPU shell

```bash
srun --account=def-account --gpus-per-node=h100:1 --cpus-per-task=8 --mem=128000 --time=0-01:15 --job-name=vllm --pty bash
```

## Read the Training Dataset:

```bash
python ./datasets_training/NuminaMath-LEAN/getDataset.py
```

This'll create a data structure called `pairs`, that's a list of tuples (informal_theorem, formal_theorem). You can use this list for training the joint-embedding or the auto-formalization LLM. This code also prints the first two elements from `pairs` for convenience.

## Training Joint Embeddings

The `train.py` script trains dual encoders to create joint embeddings between natural language theorem statements and proof sequences using contrastive learning.

### Usage Example:
```bash
python train.py \
    --csv_path data/herald_output.csv \
    --use_wandb \
    --wandb_project "nl-proof-embeddings" \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-5 \
    --nl_trainable_layers 3 \
    --proof_trainable_layers 2
```

### Key Features:
- **Selective Layer Training**: Instead of freezing entire models, train only the last N layers with `--nl_trainable_layers` and `--proof_trainable_layers`
- **Contrastive Learning**: Uses InfoNCE loss with in-batch negatives and optional cross-batch negatives
- **Memory Optimization**: Gradient checkpointing and automatic mixed precision support
- **Comprehensive Evaluation**: Multiple retrieval metrics tracked during training

## Evaluation Metrics

The training script computes comprehensive retrieval metrics to evaluate how well the learned embeddings can match natural language statements with their corresponding proofs. All metrics are computed bidirectionally (NL→Proof and Proof→NL).

### Core Retrieval Metrics

#### **Recall@K (R@K)**
- **Definition**: Fraction of queries where the correct document appears in the top-K retrieved results
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: R@5 = 0.8 means 80% of queries have their correct match in the top-5 results
- **Use Case**: Primary metric for retrieval quality evaluation
- **Note**: For our single-relevant-document task, Recall@K is equivalent to Precision@K and Hit Rate@K

### Advanced Ranking Metrics

#### **Mean Reciprocal Rank (MRR)**
- **Definition**: Average of reciprocal ranks of the first correct result
- **Formula**: `MRR = (1/N) * Σ(1/rank_i)` where rank_i is the rank of correct document for query i
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: MRR = 0.5 means correct documents are found at average rank 2
- **Use Case**: Emphasizes early retrieval of correct results

#### **Mean Average Precision@K (MAP@K)**
- **Definition**: Mean of Average Precision scores across all queries, truncated at rank K
- **Range**: 0.0 to 1.0 (higher is better)
- **Note**: For single relevant document, MAP@K = (1/rank) if rank ≤ K, else 0
- **Use Case**: Comprehensive ranking quality assessment

#### **Normalized Discounted Cumulative Gain@K (NDCG@K)**
- **Definition**: Position-discounted relevance score normalized by ideal ranking
- **Formula**: `NDCG = DCG / IDCG` where `DCG = Σ(rel_i / log2(pos_i + 1))`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Penalizes correct documents found at lower ranks
- **Use Case**: Ranking quality with position-based discounting

### Descriptive Statistics

#### **Average Rank & Median Rank**
- **Definition**: Mean and median position of correct documents in ranked results
- **Range**: 1 to N (lower is better, where N is collection size)
- **Interpretation**: Average rank = 3.5 means correct documents typically found around position 3-4
- **Use Case**: Intuitive understanding of typical retrieval performance

#### **Similarity Score Statistics**
- **Positive Similarity**: Statistics of similarity scores between matched NL-proof pairs
  - `pos_sim_mean`, `pos_sim_std`, `pos_sim_min`, `pos_sim_max`
- **Negative Similarity**: Statistics of similarity scores between unmatched pairs
  - `neg_sim_mean`, `neg_sim_std`
- **Similarity Gap**: Difference between mean positive and negative similarities
- **Use Case**: Understanding embedding space separation and model confidence

### Logging to Weights & Biases

All metrics are automatically logged to wandb when `--use_wandb` is enabled:

- **Training metrics**: `train/loss`, `train/learning_rate` (per step)
- **Validation metrics**: All retrieval metrics with `val/` prefix (per epoch)
- **Model info**: Parameter counts, architecture details
- **Best model tracking**: `val/best_avg_recall` when new best model is saved

The metrics help monitor training progress, compare different model configurations, and ensure the learned embeddings create meaningful semantic similarities between natural language statements and formal proofs.
