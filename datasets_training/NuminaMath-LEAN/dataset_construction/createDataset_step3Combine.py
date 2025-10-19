import pandas as pd
import glob

# Pattern to match your files
file_pattern = "./datasets_training/NuminaMath-LEAN/dataset_step3-*.csv"

# Read all matching files
all_files = glob.glob(file_pattern)
print ("Read", len(all_files), "files!")

# Load and concat
df_list = [pd.read_csv(f, dtype=str) for f in all_files]
len_df_list = [len(d) for d in df_list]
print (len_df_list)
final_df = pd.concat(df_list, ignore_index=True)

# Save combined file
#final_df.to_csv("./datasets_training/NuminaMath-LEAN/dataset_step3.csv", index=False)

# Total rows
print("Total rows:", len(final_df))

final_df["fl_proof_compiles?"] = final_df["fl_proof_compiles?"].fillna("n/a")

counts = (
    final_df.groupby(["has_fl_proof?", "fl_proof_compiles?", "has_nl_proof?"])
    .size()
    .reset_index(name="count")
)

#print(final_df["has_fl_proof?"].value_counts())
#counts = final_df.groupby(["has_fl_proof?", "fl_proof_compiles?"]).size().reset_index(name="count")
print(counts)
