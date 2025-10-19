from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import json, re

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with an LLM, using vllm")

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for vllm"
    )

    parser.add_argument(
        "--num_samples_per_task",
        type=int,
        default=32,
        help="Number of samples to generate per task"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="AI-MO/Kimina-Prover-RL-1.7B",
        help="Model identifier (e.g., HuggingFace model repo)"
    )

    parser.add_argument(
        "--method_tag",
        type=str,
        default="OffTheShelfWithEg",
        help="Tag to identify evaluation method"
    )

    parser.add_argument(
        "--eval_dir",
        type=str,
        default="results_miniF2F",
        help="Directory to store evaluation results"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets_validation/minif2f/dataset.jsonl",
        help="Path to the dataset file"
    )

    parser.add_argument(
        "--use_examples_in_prompt",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set 1 to use examples in prompt, 0 to disable"
    )

    return parser.parse_args()

#-----------------Examples-----------------
examples = '''
### Example 1
<informal_statement>
Find the sum of all positive integers $n$ such that $\sqrt{n^2+85n+2017}$ is an integer.
</informal_statement>

<informal_proof>
We are looking for all positive integers $n$ such that $\sqrt{n^2+85n+2017}$ is an integer. Let this integer be $k$. Thus, we have the equation $k^2 = n^2+85n+2017$.

To find the values of $n$, we algebraically manipulate the equation. We aim to express it in a form that allows us to use factorization. The key step, as performed by the `ring_nf` and `linarith` tactics in the Lean proof, is to transform this equation into:
$(2k - 2n - 85)(2k + 2n + 85) = 843$

To verify this transformation:
Start with $k^2 = n^2 + 85n + 2017$.
Multiply by 4: $4k^2 = 4n^2 + 340n + 8068$.
Rearrange terms to complete the square for $n$:
$4k^2 = (4n^2 + 340n + 2809) - 2809 + 8068$
$4k^2 = (2n + 85)^2 + 5259$. This is not the expression yielding 843.

Let's use the structure derived from the Lean proof's `wkw` lemma:
Let $a = 2k - 2n - 85$ and $b = 2k + 2n + 85$.
Then $ab = 843$.
Adding these two expressions gives $a + b = (2k - 2n - 85) + (2k + 2n + 85) = 4k$.
Subtracting the first from the second gives $b - a = (2k + 2n + 85) - (2k - 2n - 85) = 4n + 170$.

Since $n$ is a positive integer, $4n+170$ must be a positive integer, which implies $b-a > 0$, or $b > a$.
Also, for $k$ and $n$ to be integers, $a+b$ must be divisible by 4, and $b-a-170$ must be divisible by 4.

The Lean proof uses a helper lemma `ta` which enumerates all integer divisors of 843. These divisors are $\{1, 3, 281, 843, -1, -3, -281, -843\}$.
We consider pairs of factors $(a, b)$ of 843 such that $ab = 843$. We also require $b > a$.

We check each relevant pair $(a, b)$:

1.  **Pair $(1, 843)$:**
    $a=1, b=843$.
    $b-a = 843-1 = 842$.
    We set $4n+170 = 842$, which gives $4n = 672$, so $n=168$.
    Since $n=168$ is positive, this is a valid solution.
    (Also, $a+b=844$, so $4k=844$, $k=211$, which is an integer).

2.  **Pair $(3, 281)$:**
    $a=3, b=281$.
    $b-a = 281-3 = 278$.
    We set $4n+170 = 278$, which gives $4n = 108$, so $n=27$.
    Since $n=27$ is positive, this is a valid solution.
    (Also, $a+b=284$, so $4k=284$, $k=71$, which is an integer).

The other pairs of factors for $ab=843$ where $b>a$ would involve negative numbers.

3.  **Pair $(-843, -1)$:**
    $a=-843, b=-1$.
    $b-a = -1 - (-843) = 842$.
    Setting $4n+170 = 842$, we get $4n=672$, so $n=168$. This is the same solution as case 1.

4.  **Pair $(-281, -3)$:**
    $a=-281, b=-3$.
    $b-a = -3 - (-281) = 278$.
    Setting $4n+170 = 278$, we get $4n=108$, so $n=27$. This is the same solution as case 2.

The pairs where $a>b$ (e.g., $(843, 1)$ or $(281, 3)$) would yield negative values for $n$, which are not considered.
Thus, the only positive integers $n$ that satisfy the condition are $168$ and $27$.

The problem asks for the sum of these values of $n$.
Sum $= 168 + 27 = 195$.
The Lean proof concludes by summing the elements of the finite set $\{168, 27\}$ to arrive at $195$.
</informal_proof>

<formal_proof>
```lean4
import Mathlib
/-- Find the sum of all positive integers $n$ such that $\sqrt{n^2+85n+2017}$ is an integer. -/
theorem algebra_53827 :
∑ᶠ n ∈ {n : ℕ | 0 < n ∧ ∃ k, k^2 = (↑n^2 : ℤ) + 85 * n + 2017}, n = 195 := by
  -- list all divisors
  have ta {a b : ℤ} (h : a * b = 843) :
      a = 1 ∨ a = 3 ∨ a = 281 ∨ a = 843 ∨ a = -1 ∨ a = -3 ∨ a = -281 ∨ a = -843 := by
    have ha : a.natAbs ∈ (843).divisors := by
      simp; use b.natAbs; rw [←Int.natAbs_mul, h]; rfl
    simp only [(by native_decide : (843 : ℕ).divisors = { 1, 3, 281, 843 }), Finset.mem_insert,
      Finset.mem_singleton] at ha
    omega
  have enum : {n : ℕ | 0 < n ∧ ∃ k, k^2 = (↑n^2 : ℤ) + 85 * n + 2017} = ↑({168, 27} : Finset ℕ) := by
    ext n
    simp only [Set.mem_setOf_eq, Finset.coe_insert, Finset.coe_singleton, Set.mem_insert_iff,
      Set.mem_singleton_iff]
    constructor <;> intro h
    · obtain ⟨a, k, s⟩ := h
      -- have something to factor
      have wkw : (2 * k - 2 * n - 85) * (2 * k + 2 * n + 85) = 843 := by ring_nf; linarith
      obtain ald | ald | ald | ald | ald | ald | ald | ald := ta wkw
      all_goals zify at *
      all_goals rw [ald] at wkw
      all_goals omega
    · obtain rfl | rfl | rfl | rfl := h
      all_goals norm_num
      exists 211
      exists 71
  -- calculate sum
  rw [enum, finsum_mem_coe_finset]
  decide
```
</formal_proof>

### Example 2
<informal_statement>
Prove that for each positive integer $k$ there exists a number base $b$ along with $k$ triples of Fibonacci numbers $(F_u,F_v,F_w)$ such that when they are written in base $b$, their concatenation is also a Fibonacci number written in base $b$. (Fibonacci numbers are defined by $F_1 = F_2 = 1$ and $F_{n+2} = F_{n+1} + F_n$ for all positive integers $n$.)
</informal_statement>

<informal_proof>
We want to prove that for any positive integer $k$, there exists a number base $b$ and $k$ triples of Fibonacci numbers $(F_u, F_v, F_w)$ such that their concatenation in base $b$ results in another Fibonacci number. The Fibonacci sequence is defined by $F_1 = 1$, $F_2 = 1$, and $F_{n+2} = F_{n+1} + F_n$ for $n \ge 1$. The specific condition relating these numbers is $F_{u+v+w} = F_u \cdot b^{F_v} + F_v \cdot b^{F_w}$.

To establish the existence of such $b, u, v, w$, we can provide explicit witnesses.
Let us choose the base $b = 1$.

Now, we need to select indices $u, v, w$ such that the conditions are met. Let's choose $u = 1$, $v = 1$, and $w = 1$.

We must first verify that the Fibonacci numbers for these indices are positive.
By definition, $F_1 = 1$.
Thus, $F_u = F_1 = 1$, $F_v = F_1 = 1$, and $F_w = F_1 = 1$.
Since $1 > 0$, the conditions $F_u > 0$, $F_v > 0$, and $F_w > 0$ are satisfied.

Next, we must verify the main equality: $F_{u+v+w} = F_u \cdot b^{F_v} + F_v \cdot b^{F_w}$.
Substituting our chosen values:
The left-hand side is $F_{1+1+1} = F_3$. By definition, $F_3 = F_2 + F_1 = 1 + 1 = 2$.
The right-hand side is $F_1 \cdot 1^{F_1} + F_1 \cdot 1^{F_1}$.
Substituting $F_1 = 1$, this becomes $1 \cdot 1^1 + 1 \cdot 1^1 = 1 \cdot 1 + 1 \cdot 1 = 1 + 1 = 2$.

Since the left-hand side ($2$) equals the right-hand side ($2$), the equality holds for these choices of $b, u, v, w$.
The problem requires that for a given $k$, we can find $k$ such triples. Our chosen values $b=1, u=1, v=1, w=1$ are constant and satisfy the condition regardless of the specific index $i$ (from $0$ to $k-1$) being considered. Therefore, we can use this single set of values $(b, u, v, w)$ to satisfy the condition for all $k$ required triples.

This construction demonstrates the existence of the required base $b$ and the $k$ triples of Fibonacci numbers.
</informal_proof>

<formal_proof>
```lean4
import Mathlib

theorem number_theory_62389 (k : ℕ) (hk : 0 < k) :
    ∃ b : ℕ, ∀ i ∈ Finset.range k, ∃ u v w : ℕ,
      Nat.fib u > 0 ∧ Nat.fib v > 0 ∧ Nat.fib w > 0 ∧
      Nat.fib (u + v + w) = Nat.fib u * b ^ (Nat.fib v) + Nat.fib v * b ^ (Nat.fib w) := by 
  use 1
  intro i hi
  use 1
  use 1
  use 1
  aesop
```
</formal_proof>

### Example 3
<informal_statement>
Find prime numbers $p , q , r$  such that $p+q^2+r^3=200$. Give all the possibilities.
Remember that the number $1$ is not prime.
</informal_statement>

<informal_proof>
We aim to prove the equivalence: \(p, q, r\) are prime numbers and \(p + q^2 + r^3 = 200\) if and only if \((p, q, r)\) is one of the tuples \((167, 5, 2)\), \((71, 11, 2)\), \((23, 13, 2)\), or \((71, 2, 5)\).

We prove this in both directions.

**Forward Direction: \(p, q, r\) are prime and \(p + q^2 + r^3 = 200 \implies (p, q, r) \in \{\dots\}\)**

First, we establish some bounds. Since \(p, q, r\) are prime, they are all greater than or equal to 2. From \(p + q^2 + r^3 = 200\):
*   \(p \leq 200\).
*   Since \(q \geq 2\) and \(r \geq 2\), we have \(q^2 \geq 4\) and \(r^3 \geq 8\). Thus, \(q^2 < 200 - p - r^3 \leq 200 - 2 - 8 = 190\). This implies \(q \leq \sqrt{190}\). As \(13^2 = 169\) and \(14^2 = 196\), we conclude \(q \leq 13\).
*   Similarly, \(r^3 < 200 - p - q^2 \leq 200 - 2 - 4 = 194\). This implies \(r \leq \sqrt[3]{194}\). As \(5^3 = 125\) and \(6^3 = 216\), we conclude \(r \leq 5\).

Next, we consider the parity of the primes. The sum \(p + q^2 + r^3 = 200\) is even. If \(p, q, r\) were all odd primes, then \(q^2\) would be odd and \(r^3\) would be odd. The sum \(p + q^2 + r^3\) would then be odd + odd + odd, which is odd. This contradicts the sum being 200 (even). Therefore, at least one of \(p, q, r\) must be an even prime. Since 2 is the only even prime, one of \(p, q, r\) must be equal to 2.

We now analyze these three cases:

1.  **If \(p = 2\):** The equation becomes \(2 + q^2 + r^3 = 200\), or \(q^2 + r^3 = 198\). We know \(q\) is prime and \(q \leq 13\), and \(r\) is prime and \(r \leq 5\). Also, \(q\) and \(r\) cannot both be 2 simultaneously. If \(r=2\), \(q^2+8=198 \implies q^2=190\) (not a square). If \(q=2\), \(4+r^3=198 \implies r^3=194\) (not a cube). Thus, \(q, r\) must be odd primes.
    *   For \(r=3\), \(q^2 + 3^3 = 198 \implies q^2 + 27 = 198 \implies q^2 = 171\), which is not a square.
    *   For \(r=5\), \(q^2 + 5^3 = 198 \implies q^2 + 125 = 198 \implies q^2 = 73\), which is not a square.
    There are no solutions when \(p=2\).

2.  **If \(q = 2\):** The equation becomes \(p + 2^2 + r^3 = 200\), or \(p + 4 + r^3 = 200\), simplifying to \(p + r^3 = 196\). We know \(p\) is prime and \(p \leq 200\), and \(r\) is prime and \(r \leq 5\). \(p\) and \(r\) cannot both be 2. If \(r=2\), \(p+8=196 \implies p=188\) (not prime). If \(p=2\), \(2+r^3=196 \implies r^3=194\) (not a cube). Thus, \(p, r\) must be odd primes.
    *   For \(r=3\), \(p + 3^3 = 196 \implies p + 27 = 196 \implies p = 169 = 13^2\), which is not prime.
    *   For \(r=5\), \(p + 5^3 = 196 \implies p + 125 = 196 \implies p = 71\). \(71\) is prime.
    This yields the solution \((p, q, r) = (71, 2, 5)\).

3.  **If \(r = 2\):** The equation becomes \(p + q^2 + 2^3 = 200\), or \(p + q^2 + 8 = 200\), simplifying to \(p + q^2 = 192\). We know \(p\) is prime and \(p \leq 200\), and \(q\) is prime and \(q \leq 13\). \(p\) and \(q\) cannot both be 2. If \(q=2\), \(p+4=192 \implies p=188\) (not prime). If \(p=2\), \(2+q^2=192 \implies q^2=190\) (not a square). Thus, \(p, q\) must be odd primes.
    *   For \(q=3\), \(p + 3^2 = 192 \implies p + 9 = 192 \implies p = 183 = 3 \times 61\), not prime.
    *   For \(q=5\), \(p + 5^2 = 192 \implies p + 25 = 192 \implies p = 167\). \(167\) is prime. This yields \((p, q, r) = (167, 5, 2)\).
    *   For \(q=7\), \(p + 7^2 = 192 \implies p + 49 = 192 \implies p = 143 = 11 \times 13\), not prime.
    *   For \(q=11\), \(p + 11^2 = 192 \implies p + 121 = 192 \implies p = 71\). \(71\) is prime. This yields \((p, q, r) = (71, 11, 2)\).
    *   For \(q=13\), \(p + 13^2 = 192 \implies p + 169 = 192 \implies p = 23\). \(23\) is prime. This yields \((p, q, r) = (23, 13, 2)\).

In summary, the solutions are \((71, 2, 5)\), \((167, 5, 2)\), \((71, 11, 2)\), and \((23, 13, 2)\).

**Backward Direction: \((p, q, r) \in \{\dots\} \implies p, q, r\) are prime and \(p + q^2 + r^3 = 200\)**

We verify each of the four possible tuples:
1.  \((167, 5, 2)\): \(167\), \(5\), and \(2\) are prime. \(167 + 5^2 + 2^3 = 167 + 25 + 8 = 200\).
2.  \((71, 11, 2)\): \(71\), \(11\), and \(2\) are prime. \(71 + 11^2 + 2^3 = 71 + 121 + 8 = 200\).
3.  \((23, 13, 2)\): \(23\), \(13\), and \(2\) are prime. \(23 + 13^2 + 2^3 = 23 + 169 + 8 = 200\).
4.  \((71, 2, 5)\): \(71\), \(2\), and \(5\) are prime. \(71 + 2^2 + 5^3 = 71 + 4 + 125 = 200\).

All four tuples satisfy the given conditions.

Thus, the equivalence is established.
</informal_proof>

<formal_proof>
```lean4
import Mathlib

/- Find prime numbers $p , q , r$  such that $p+q^2+r^3=200$. Give all the possibilities.

Remember that the number $1$ is not prime. -/
theorem number_theory_54583 (p q r : ℕ): 
p.Prime ∧ q.Prime ∧ r.Prime ∧ (p + q^2 + r^3 = 200) 
↔ 
(p = 167 ∧ q = 5 ∧ r = 2) ∨ (p = 71 ∧ q = 11 ∧ r = 2) ∨ (p = 23 ∧ q = 13 ∧ r = 2) ∨ (p = 71 ∧ q = 2 ∧ r = 5) := by
  
  constructor


  --From left to right, assume we have p,q,r primes and their sum equal to 200.
  intro ⟨pp,qp,rp,h⟩ 

  --Some general bounds on p, q, and r.
  have pge2 : 2 ≤ p := by exact Nat.Prime.two_le pp
  have qge2 : 2 ≤ q := by exact Nat.Prime.two_le qp
  have rge2 : 2 ≤ r := by exact Nat.Prime.two_le rp
  have ple200 : p ≤ 200 := by omega
  have qle14 : q ≤ 15 := by 
    by_contra qg15
    push_neg at qg15
    have : 15^2 < q^2 := by nlinarith
    omega
  have rle : r ≤ 6 := by 
    by_contra rg6
    push_neg at rg6
    have : 6^3 < r^3 := by apply pow_lt_pow_left₀; exact rg6; norm_num; norm_num
    omega
  
  --Consider the parity, one of p,q,r has to be even, so it is 2.
  have : Even p ∨ Even q ∨ Even r := by
    by_contra allodd
    push_neg at allodd
    simp at allodd
    obtain ⟨op,oq,or⟩ := allodd
    have t1: Odd (q^2) := by exact Odd.pow oq
    have t2: Odd (r^3) := by exact Odd.pow or
    have t3: Even (p+q^2) := by exact Odd.add_odd op t1
    have t4: Odd (p+q^2 + r^3) := by exact Even.add_odd t3 t2
    rw [h] at t4
    have : ¬ Odd 200 := by decide
    contradiction
  
  --Now, we just enumerate all cases.
  obtain ep | eq | er := this
  -- Case p even:
  have pe : p = 2 := by exact (Nat.Prime.even_iff pp).mp ep
  simp [pe] at h ⊢
  interval_cases q <;> interval_cases r <;> norm_num at h ⊢

  -- Case q even:
  have qe : q = 2 := by exact (Nat.Prime.even_iff qp).mp eq
  simp [qe] at h ⊢
  interval_cases p <;> interval_cases r <;> norm_num at h pp rp ⊢

  
  -- Case r even:
  have re : r = 2 := by exact (Nat.Prime.even_iff rp).mp er
  simp [re] at h ⊢
  interval_cases p <;> interval_cases q <;> norm_num at h pp qp ⊢


  --From right to left, simple caculation shows they satisfy the equation.
  intro h
  obtain ⟨pe,qe,re⟩ | ⟨pe,qe,re⟩ | ⟨pe,qe,re⟩ | ⟨pe,qe,re⟩ := h <;>
  simp [pe,qe,re] <;>
  norm_num
```
</formal_proof>
'''
#-----------------Examples-----------------

def prepare_jsonl_to_evaluation(dataset_jsonl_path, eval_path, model_name):
    global NUM_SAMPLES_PER_TASK, methodTag
    eval_filename = f"{model_name}-{methodTag}_output.jsonl"
    eval_file_path = os.path.join(eval_path, eval_filename)

    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]   
        print ("Output file already exists, resuming!")
        return data, eval_file_path

    # Read JSONL as a list of dicts
    with open(dataset_jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for dataInstance in data:
        dataInstance["INFERENCE_DONE"] = "no"
        for i in range(1, NUM_SAMPLES_PER_TASK + 1):
            for col_base in [
                "LLM_Output#",
                "LLM_Syntax?#",
                "LLM_SyntaxError#",
                "LLM_Semantics?#",
                "LLM_SemanticsError#",
            ]:
                dataInstance[col_base + str(i)] = ""

    with open(eval_file_path, "w", encoding="utf-8") as f:
        for dataInstance in data:
            f.write(json.dumps(dataInstance) + "\n")

    print(f"Modified file saved to {eval_file_path}")
    return data, eval_file_path

def saveJsonl(listOfDict, filePath):
    with open(filePath, "w", encoding="utf-8") as f:
        for dataInstance in listOfDict:
            f.write(json.dumps(dataInstance) + "\n")    

def extract_fl_proof_betweenTags(text, TAG):
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

def extract_fl_proof(text: str) -> str:
    """
    Extracts the last content enclosed in ```lean4 ... ``` from the input text.
    
    Returns the last match as a string, including newlines. 
    Returns an empty string if no match is found.
    """
    pattern = r"```lean4\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # last match
    return ""


def wrapPromptInQuery(informal_statement, informal_proof, inContextEg = False):
    global examples, useExamplesInPrompt
    inContextEg = useExamplesInPrompt
    if inContextEg:
        exampleInPrompt = "Here are a few examples:\n" + examples + "\n"
    else:
        exampleInPrompt = ""
    input_query_template = f'''
    You task is to take as input an informal proof in natural language and autoformalize it in Lean 4 with a header. 
    Think step-by-step and ensure that the output formal theorem is compilabile with Lean 4 (version 4.15.0).

    {exampleInPrompt}

    Here is the **actual** informal proof in natural language:
    <informal_statement>
    {informal_statement}
    </informal_statement>

    <informal_proof>
    {informal_proof}
    </informal_proof>

    Now first think step-by-step for the actual output and autoformalize it in Lean 4 with a header. Importantly, enclose the final formal proof in Lean 4 inside the following tags:

    <formal_proof>
    ```lean4
    (Provide your entire Lean 4 proof with header here)
    ```
    </formal_proof>
    '''

    return input_query_template


def inference_on_dataset():
    global eval_dir, model_id, methodTag, datasetPath, PORT
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    model_name = model_id.split("/")[-1].replace(".", "-")
    LOGFile = os.path.join(eval_dir, f"{model_name}-{methodTag}_LOG.txt")
    # Step 1: Clear the LOG file at the beginning
    with open(LOGFile, "w") as f:
        f.write("")

    listOfDataDict, final_save_path = prepare_jsonl_to_evaluation(datasetPath, eval_dir, model_name = model_name)

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{PORT}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    dataNum = 0
    for dataItem in tqdm(listOfDataDict):
        dataNum += 1
        if dataItem["INFERENCE_DONE"] == "yes":
            with open(LOGFile, "a") as f:
                f.write(f"==================NUM{dataNum}==================\n\n")
            continue
        informal_statement = str(dataItem['informal_statement'])
        informal_proof = str(dataItem['informal_proof'])
        with open(LOGFile, "a") as f:
            f.write(f"==================NUM{dataNum}==================\n\n")
            f.write("informal_statement: \n" + informal_statement + "\n")
            f.write("informal_proof: \n" + informal_proof + "\n")
        input_text = wrapPromptInQuery(informal_statement.strip(), informal_proof.strip())

        #'''
        chat_responses = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an expert in mathematics. Your task is to convert informal, natural-language proofs into correct Lean 4 formalizations."},
                {"role": "user", "content": input_text},
            ],
            n = NUM_SAMPLES_PER_TASK,
            max_tokens=12000,
            temperature=0.6,
            top_p=0.95
        )
        modelOuts = [c.message.content for c in chat_responses.choices]
        #'''
        '''
        modelOuts = ["<formal_proof>Just Trying haha</formal_proof>" for i in range(NUM_SAMPLES_PER_TASK)]
        '''
        assert len(modelOuts) == NUM_SAMPLES_PER_TASK
        responses = []
        for modelOutIndx, modelOut in enumerate(modelOuts):
            if modelOut is not None:
                parsedModelOut = extract_fl_proof(modelOut)
                if len(parsedModelOut.strip()) == 0:
                    parsedModelOut = "ERROR [No text within tags]"
            else:
                parsedModelOut = "ERROR [No output returned]"
            responses.append(parsedModelOut)
            with open(LOGFile, "a") as f:
                f.write(f"Generated output# {modelOutIndx}/{NUM_SAMPLES_PER_TASK}:" + "\n")
                #f.write(f"modelOut: {modelOut}" + "\n")
                f.write(f"parsedModelOut: {parsedModelOut}" + "\n")

        assert len(responses) == NUM_SAMPLES_PER_TASK

        for trialNum in range(1, NUM_SAMPLES_PER_TASK+1):
            dataItem[f"LLM_Output#{trialNum}"] = responses[trialNum - 1]
        dataItem["INFERENCE_DONE"] = "yes"

        saveJsonl(listOfDataDict, final_save_path)

        with open(LOGFile, "a") as f:
            f.write("Wrote all LLM outputs!!\n")
            f.write("=====================================\n\n")

    print("Inference and saving complete.")
    return


if __name__ == "__main__":
    args = parse_args()
    PORT = args.port
    NUM_SAMPLES_PER_TASK = args.num_samples_per_task
    model_id = args.model_id
    methodTag = args.method_tag
    eval_dir = args.eval_dir
    datasetPath = args.dataset_path
    useExamplesInPrompt = bool(args.use_examples_in_prompt)

    print(f"Samples: {NUM_SAMPLES_PER_TASK}, Model: {model_id}")
    print(f"Eval dir: {eval_dir}, Dataset: {datasetPath}, Use examples: {useExamplesInPrompt}")

    inference_on_dataset()
    #module load gcc arrow/15.0.1 opencv/4.11.0
    #source ~/lean_env/bin/activate
    #nohup vllm serve AI-MO/Kimina-Prover-RL-1.7B --port 8000 --tensor-parallel-size 1 --max_model_len 40960 > ./results_miniF2F/vllm.log 2>&1 &
    #https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
    #python llm_inference_gpu_IaC-Eval/gpu_inference_Qwen2.5-Coder-3B-Instruct.py
    #python evaluate.py --eval_name Qwen2.5-Coder-3B-Instruct