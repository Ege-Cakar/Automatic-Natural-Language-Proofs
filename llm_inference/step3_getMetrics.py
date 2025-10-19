import json
import glob
import re, os, sys
from collections import defaultdict

# 1. Use regex + glob to get all files
#all_files = glob.glob("./evaluation_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithEg_eval-*.jsonl")
all_files = glob.glob("./evaluation_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithEg_eval-*.jsonl")
# Optional: sort numerically by the number in the filename
all_files = sorted(all_files, key=lambda x: int(re.search(r"eval-(\d+)\.jsonl", x).group(1)))

assert all_files
# derive log filename from prefix
if all_files:
    base_dir = os.path.dirname(all_files[0])
    prefix = re.sub(r"-\d+\.jsonl$", "", os.path.basename(all_files[0]))
    log_file = os.path.join(base_dir, f"LOG-{prefix}.txt")
    print ("\nOutputs redirected to:", log_file, "\n")
# redirect stdout to log file
sys.stdout = open(log_file, "w")

# 2. Combine all JSONL files into a single list of dicts
all_dicts = []
for filename in all_files:
    with open(filename, 'r') as f:
        for line in f:
            all_dicts.append(json.loads(line))

print("Total number of instances:", len(all_dicts))

# 3. Define pass@k values
pass_k_values = [1, 2, 4, 8, 16, 32, 64]

# 4. Initialize accumulators for metrics
syntax_metrics = defaultdict(list)
semantics_metrics = defaultdict(list)

# 5. Compute correctness at pass@k for each instance
for instance in all_dicts:
    # Get all syntax and semantic keys
    syntax_keys = sorted([k for k in instance if re.match(r"LLM_Syntax\?#\d+", k)],
                         key=lambda x: int(x.split("#")[1]))
    semantics_keys = sorted([k for k in instance if re.match(r"LLM_Semantics\?#\d+", k)],
                            key=lambda x: int(x.split("#")[1]))

    # Convert "yes"/"no" to 1/0
    syntax_vals = [1 if instance[k].lower() == "yes" else 0 for k in syntax_keys]
    semantics_vals = [1 if instance[k].lower() == "yes" else 0 for k in semantics_keys]

    for k_val in pass_k_values:
        syntax_metrics[k_val].append(int(any(syntax_vals[:k_val])))
        semantics_metrics[k_val].append(int(any(semantics_vals[:k_val])))


# 6. Aggregate metrics (mean correctness)
print("\nSyntax correctness at pass@k:")
for k_val in pass_k_values:
    num = sum(syntax_metrics[k_val])
    denom = len(syntax_metrics[k_val])
    pct = num / 244
    print(f"pass@{k_val}: passed {num}, checked {denom}, {num}/244 = {pct:.4f}")

print("\nSemantic correctness at pass@k:")
for k_val in pass_k_values:
    num = sum(semantics_metrics[k_val])
    denom = len(semantics_metrics[k_val])
    pct = num / 244
    print(f"pass@{k_val}: passed {num}, checked {denom}, {num}/244 = {pct:.4f}")


# close log file
sys.stdout.close()