import json
import re
import pandas as pd
from tqdm import tqdm
import sys, os
import argparse
import re

SAVE_FREQ = 5

# Add the directory containing checkLEAN.py to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LEAN_interaction')))
from checkLEAN import *

def saveData(data_list, output_file):
    # Convert back to DataFrame
    df_out = pd.DataFrame(data_list)
    # Save to CSV
    df_out.to_csv(output_file, index=False)
    # Count combinations of has_fl_proof? and fl_proof_compiles?
    counts = df_out.groupby(["has_fl_proof?", "fl_proof_compiles?"]).size().reset_index(name="count")
    print(counts)

def extract_lean_code(text: str):
    if text is None:
        print ("LLM_LOG: [No output returned]")
        return None
    FENCE_RE = re.compile(
        r"(?:^|\n)```(?:lean4|lean)\s*\n(.*?)\n```",
        re.DOTALL | re.IGNORECASE,
    )
    # 1) All lean/lean4 fenced blocks
    blocks = FENCE_RE.findall(text)
    if not blocks:
        print ("LLM_LOG: [No lean4 blocks]")
        return None

    # 2) Fall back to the longest Lean block
    return max(blocks, key=len).strip()

def prompt_LLM(client, fl_statement):
    prompt = """
Complete the following Lean 4 (v4.15.0) code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof. 
Ensure compatibility with LEAN v4.15.0.
    """.strip()

    chat_response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-Prover-V2-7B",
        messages=[
            {"role": "user", "content": prompt.format(fl_statement)},
        ],
        max_tokens=10000,
        temperature=0.6,
        top_p=0.95
    )
    #print ("LLMout:", message.content[0].text)
    lean_code = extract_lean_code(chat_response.choices[0].message.content)
    return lean_code

def generate_FLproofs(client, list_of_dicts, output_file, strt_indx_padded, run_dir, project_dir, lean_file_path):
    #run_dir, project_dir, lean_file_path = bootstrap_project()
    for rowIndx, row in enumerate(tqdm(list_of_dicts, desc="Checking Lean proofs")):
        print("\n" + "="*50)
        print(f"Row {rowIndx}")
        if row["has_fl_proof?"] == "no":
            TRY_SUCCESS = False
            for tryNum in range(5):
                print (f"Try #{tryNum}")
                fl_stmt = row["formal_statement"]
                generated_fl_proof = prompt_LLM(client, fl_stmt)
                if generated_fl_proof is None:
                    print ("Skipping, no parsable output from LLM")
                    continue
                lean_code = write_basic_lean("", generated_fl_proof, lean_file_path)
                ok_repl, out_repl = check_repl(lean_file_path, project_dir)
                # Print with clear delimiters
                print("-"*50)
                print("OK_REPL:")
                print(ok_repl)
                # Update the row
                if ok_repl:
                    TRY_SUCCESS = True
                    row["has_fl_proof?"] = "yes_generated"
                    row["fl_proof_compiles?"] = "yes"
                    break
            if not TRY_SUCCESS:
                row["has_fl_proof?"] = "no_couldntgenerate"              
        if rowIndx % SAVE_FREQ == 0:
            saveData(list_of_dicts, output_file)
        print("="*50 + "\n", flush = True)
    saveData(list_of_dicts, output_file)

if __name__ == "__main__":

    #------------------parse command line arguments------------------
    SERVER_NODE="localhost"
    SERVER_PORT=8000
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{SERVER_NODE}:{SERVER_PORT}/v1/"

    run_dir = Path("./tmpFolder/8d9a729d-dec6-452d-bb90-d63be139ee52")
    project_dir = run_dir / "TmpProjDir"
    lean_file_path = project_dir / "TmpProjDir" / f"Basic_{strt_indx_padded}.lean"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strt_indx", 
        type=int, 
        default=0,
        help="in multiples of 1000"
    )
    args = parser.parse_args()
    strt_indx = args.strt_indx
    strt_indx_padded = f"{strt_indx:06d}"

    # Get the "AI-MO/NuminaMath-LEAN" dataset
    output_file = f"./datasets_training/NuminaMath-LEAN/dataset_step2-{strt_indx_padded}.csv"
    if os.path.exists(output_file):
        input_file = output_file
    else:
        input_file = "./datasets_training/NuminaMath-LEAN/dataset_step2.csv"
    # Read CSV with pandas
    df = pd.read_csv(input_file, dtype=str)
    df = df.fillna("")
    # Convert to list of dicts (records)
    data_list = df.to_dict(orient="records")[strt_indx : strt_indx + 1000]
    generate_FLproofs(client, data_list, output_file, strt_indx_padded, run_dir, project_dir, lean_file_path)
    print ("Done!")
