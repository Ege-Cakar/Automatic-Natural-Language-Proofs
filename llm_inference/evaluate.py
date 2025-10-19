import json
import sys
import os, shutil
import argparse
import re
import google.generativeai as genai

# put your API key here
genai.configure(api_key="AIzaSyCmeQu6GS2Av3AH5_VM1iOp8twuF8gFAe0")

# Add the directory containing checkLEAN.py to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../LEAN_interaction')))
from checkLEAN import *

def parse_args():
    #------------------parse command line arguments------------------
    parser = argparse.ArgumentParser(description="Evaluate Lean4 outputs on a dataset")
    parser.add_argument(
        "--input_jsonl", 
        type=str, 
        default="./results_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithoutEgv2_output.jsonl",
        help="Path to the output JSONL file to evaluate"
    )

    parser.add_argument(
        "--startRow", 
        type=int, 
        default=1,
        help="Starting index in result jsonl (inclusive)"
    )

    parser.add_argument(
        "--endRow", 
        type=int, 
        default=4,
        help="Ending index in result jsonl (inclusive)"
    )

    parser.add_argument(
        "--offset", 
        type=int, 
        default=0,
    )

    args = parser.parse_args()
    input_jsonl_file = args.input_jsonl
    startRow = args.startRow
    endRow = args.endRow
    offset = args.offset
    return input_jsonl_file, startRow, endRow, offset

def get_outputFileName(input_file_name, indx_padded_str):
    # Directory for output
    output_dir = "./evaluation_miniF2F"
    # Extract base name without extension
    base_name = os.path.basename(input_file_name).replace("_output.jsonl", "")
    # Construct output path
    output_file = os.path.join(output_dir, f"{base_name}_eval-{indx_padded_str}.jsonl")
    return output_file

def saveJsonl(listOfDict, filePath):
    with open(filePath, "w", encoding="utf-8") as f:
        for dataInstance in listOfDict:
            f.write(json.dumps(dataInstance) + "\n") 

def extract_text_between_tags(text, TAG):
    if text is not None:
        text = str(text)
    else:
        text = ""
    START_DELIMITER = "<{}>".format(TAG)
    END_DELIMITER = "</{}>".format(TAG)
    if (START_DELIMITER in text) and (END_DELIMITER in text):
        inner_str = (text.split(START_DELIMITER)[-1].split(END_DELIMITER)[0]).strip()
        return inner_str
    return ""

def prompt_LLM(gold_FL_stmt, generated_FL_stmt):
    global model, leanEqv_file_path, project_dir
    prompt = """
You are an expert in formal mathematics and Lean 4 theorem proving (version v4.15.0).

You are given two Lean 4 statements:

<first_lean_statement>
```lean4
{target_FL_stmt}
```
</first_lean_statement>

<second_lean_statement>
```lean4
{gen_FL_stmt}
```
</second_lean_statement>

Your tasks are as follows:

A. Semantic Equivalence Justification: Determine whether these two statements express the same mathematical proposition.
    Two statements are considered equivalent if they are logically or semantically identical, even if they differ in superficial aspects such as variable names, order of quantifiers (when order does not affect meaning), or notational representation.
    They should be judged not equivalent if they differ in logical content, assumptions, or conclusions.
    Provide a clear, detailed, and unambiguous justification.

B. Equivalence Scoring: Based on your justification, assign a score, -1 or 1:
   1 means Semantically equivalent.
   -1 means Not equivalent.
   Output the score within the following HTML tags:

    <score>
    only your integer score here
    </score>

C. Bidirectional Equivalence Proof: Produce a Lean 4 proof (v4.15.0) demonstrating the bidirectional equivalence (<->) between the two statements.
    Two statements y1 and y2 are bidirectionally equivalent (denoted as y1 <-> y2) if and only if there exists a formal proof deriving y2 from y1 and vice versa using semantics-preserving tactics.
    If the statements are equivalent, provide a proof that compiles in Lean 4.15.0. Include appropriate headers.
    If you are certain the statements are not equivalent, output [NOT EQUIVALENT].
    Output the proof within the following HTML tags:

    <equivalence_proof>
    your full proof for bidirectional equivalence here, or "[NOT EQUIVALENT]"
    </equivalence_proof>""".strip()

    formatted_prompt = prompt.format(
        target_FL_stmt=gold_FL_stmt,
        gen_FL_stmt=generated_FL_stmt,
    )

    print("\n", "-"*25, "PROMPT", "-"*25)
    print (formatted_prompt)
    print("-"*25, "PROMPT", "-"*25, "\n")

    nl_proof = None
    for tryNum in range(10):
        try:
            response = (model.generate_content(formatted_prompt)).text
            score = int(extract_text_between_tags(response, "score").strip())
            eqv_proof = extract_text_between_tags(response, "equivalence_proof").strip("```lean4").strip("```")
            lean_code = write_basic_lean("", eqv_proof, leanEqv_file_path)
            ok_repl, out_repl = check_repl(leanEqv_file_path, project_dir)
        except BaseException as e:
            print(f"Attempt {tryNum+1} failed: {e}")
            continue
        if score and eqv_proof:
            break
    return score, eqv_proof, ok_repl, out_repl

def extract_statement_from_proof(lean_code: str) -> str:
    """
    Extract the Lean statement (everything before ':= by')
    from a Lean proof string.
    """
    marker = ":= by"
    idx = lean_code.find(marker)
    if idx == -1:
        return lean_code.strip()  # fallback: return full input if ':= by' not found
    return lean_code[:idx + len(marker)].strip()

def get_metrics(input_jsonl_file, startingRowNum, endingRowNum, runDir, projectDir):
    leanFile_indx = ((startingRowNum - 1) // (endingRowNum - startingRowNum + 1)) * 1000 + offset
    leanFile_indx_padded = f"{leanFile_indx:06d}"
    output_jsonl_file = get_outputFileName(input_jsonl_file, leanFile_indx_padded)
    lean_file_path = projectDir / "TmpProjDir" / f"Basic_{leanFile_indx_padded}.lean"
    assert os.path.exists(lean_file_path)
    listOfDictToWrite = []
    with open(input_jsonl_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start = 1):
            print("="*40, f"row{idx}", "="*40)
            if idx < startingRowNum:
                continue
            if idx > endingRowNum:
                break
            line = line.strip()
            obj = json.loads(line)

            # Step 1: collect (key, number) pairs
            llmOut_keys = [
                (k, int(re.search(r"\d+$", k).group()))
                for k in obj.keys()
                if re.match(r"^LLM_Output#\d+$", k)
            ]

            # Step 2: iterate over those keys
            for key, num in llmOut_keys:
                llmOut = obj[key]
                print("-"*20)
                print(f"Key={key}")
                print(f"Value={llmOut}")
                lean_code = write_basic_lean("", llmOut, lean_file_path)
                ok_repl, out_repl = check_repl(lean_file_path, projectDir)

                obj[f"LLM_Syntax?#{num}"] = "yes" if ok_repl else "no"
                print(f"Syntax? --> {obj[f'LLM_Syntax?#{num}']}")
                print(f"REPL out --> {out_repl}")
                obj[f"LLM_SyntaxError#{num}"] = out_repl if (not ok_repl) else ""
                if ok_repl:
                    generated_FL_stmt = extract_statement_from_proof(llmOut)
                    eqvJudgeScore, eqvProof, ok_eqvProof, out_eqvProof = prompt_LLM(obj["formal_statement"], generated_FL_stmt)
                    obj[f"LLM_Semantics?#{num}"] = "yes" if ok_eqvProof else "no"
                    obj[f"LLM_SemanticsError#{num}"] = f"<proof>{eqvProof}</proof>" + "\n" + f"<error>{out_eqvProof}</error>"
                    obj[f"LLM_SemanticsJudge?#{num}"] = "no" if (eqvJudgeScore == -1) else "yes"
                else:
                    obj[f"LLM_Semantics?#{num}"] = "no"
                    obj[f"LLM_SemanticsError#{num}"] = "N/A"
                    obj[f"LLM_SemanticsJudge?#{num}"] = "N/A"
                print("Semantics? -->", obj[f"LLM_Semantics?#{num}"])
                print("SemanticsJudge? -->", obj[f"LLM_SemanticsJudge?#{num}"])
                print("LLM_SemanticsError -->", obj[f"LLM_SemanticsError#{num}"])
                print(f"REPL out --> {out_repl}")
                print("-"*20)
                if ok_repl and ok_eqvProof:
                    break
            listOfDictToWrite.append(obj)
            saveJsonl(listOfDictToWrite, output_jsonl_file)

if __name__ == "__main__":
    input_jsonl_file, startRow, endRow, offset = parse_args()
    model = genai.GenerativeModel("gemini-2.5-pro")
    run_dir = Path("../autoformalization-jtemb/tmpFolder/8d9a729d-dec6-452d-bb90-d63be139ee52")
    assert os.path.exists(run_dir)
    project_dir = run_dir / "TmpProjDir"
    assert os.path.exists(project_dir)

    #CAUTION: Hard-coded value 62!!
    leanEqvFile_indx = 62000 + ((startRow - 1) // (endRow - startRow + 1)) * 1000 + offset
    leanEqvFile_indx_padded = f"{leanEqvFile_indx:06d}"
    leanEqv_file_path = project_dir / "TmpProjDir" / f"Basic_{leanEqvFile_indx_padded}.lean"
    print (leanEqv_file_path)
    assert os.path.exists(leanEqv_file_path)

    get_metrics(input_jsonl_file, startRow, endRow, run_dir, project_dir)

    #python /home/pjana/projects/def-vganesh/pjana/ai_for_math/llm_inference/evaluate.py | tee /home/pjana/projects/def-vganesh/pjana/ai_for_math/evaluation_miniF2F/Kimina-Prover-RL-1-7B-OffTheShelfWithoutEgv2_evalLOG-000000.txt