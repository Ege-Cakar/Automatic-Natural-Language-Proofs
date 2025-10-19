from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

__all__ = [
    "Event",
    "parse_event",
    "Tactic",
    "Goal",
    "TacticNode",
    "ProofTrace",
    "linearize_trace",
    "build_dataset_from_csv",
    "extract_json_blob",
    "normalize_tactic",
    "collect_symbols_from_norm",
    "parse_rw_item",
    "parse_bracket_list",
]


_LEMMA_NAME = r"[A-Za-z0-9_'.]+(?:\.[A-Za-z0-9_']+)*"


def parse_rw_item(item: str) -> Dict[str, Any]:
    """Parse a single rewrite item inside `rw [ ... ]` or `simp [ ... ]` lists.

    More Explanation
    A rewrite item may start with a direction arrow '←' or '→', followed by
    a lemma/constant head, and then arbitrary Lean args we keep as a tail
    string (not deeply parsed).

    Args:
        item: A single top-level list item.

    Returns:
        Dict with keys:
            - dir (int): -1 for '←', +1 otherwise.
            - lemma (str): Head symbol name best-effort.
            - args (str): The raw tail text after the head.
    """
    s = item.strip()
    dir_val = 1
    if s.startswith("←"):
        dir_val = -1
        s = s[1:].lstrip()
    elif s.startswith("→"):
        dir_val = 1
        s = s[1:].lstrip()

    m = re.match(rf"({_LEMMA_NAME})(.*)$", s)
    if m:
        lemma = m.group(1)
        args = m.group(2).strip()
        return {"dir": dir_val, "lemma": lemma, "args": args}

    parts = s.split()
    lemma = parts[0] if parts else ""
    args = " ".join(parts[1:]) if len(parts) > 1 else ""
    return {"dir": dir_val, "lemma": lemma, "args": args}


def parse_bracket_list(s: str) -> List[str]:
    """Split a Lean bracket list '[a, b, c]' content into items safely.

    More Explanation
    Avoids splitting on commas inside parentheses/brackets/braces/⟨⟩.

    Args:
        s: The string content between '[' and ']'.

    Returns:
        A list of top-level items as strings.
    """
    items: List[str] = []
    cur: List[str] = []
    depth = 0

    for ch in s:
        if ch == "," and depth == 0:
            token = "".join(cur).strip()
            if token:
                items.append(token)
            cur = []
            continue

        cur.append(ch)
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            depth = max(0, depth - 1)

    if cur:
        token = "".join(cur).strip()
        if token:
            items.append(token)

    return items


def normalize_tactic(raw: str) -> Dict[str, Any]:
    """Normalize a Lean tactic string into a structured dict.

    More Explanation
    Recognizes common tactics (`simp`, `simpa`, `rw`, `apply`, `exact`, `refine`,
    `rfl`, `dsimp`). Extracts lemmas/unfold lists, direction arrows, and `using`
    clauses. Leaves unrecognized tactics as a generic `{op: head, raw: ...}`.

    Args:
        raw: The raw tactic text.

    Returns:
        A normalized dictionary capturing the tactic head and its key arguments.
    """
    s = (raw or "").strip()
    if not s:
        return {"op": ""}

    # simp / simpa
    if s.startswith("simp") or s.startswith("simpa"):
        op = "simpa" if s.startswith("simpa") else "simp"
        out: Dict[str, Any] = {"op": op}

        # detect 'only' (within the head area is enough)
        head_span = s[:10]
        out["only"] = " only" in head_span

        # bracket list of lemmas/unfolds
        lemmas_block = re.search(r"\[(.*)\]", s)
        if lemmas_block:
            content = lemmas_block.group(1)
            items = parse_bracket_list(content)
            out["lemmas_or_unfold"] = items

        # 'using' clause at the end
        m_using = re.search(r"\busing\s+(.+)$", s)
        if m_using:
            out["using"] = m_using.group(1).strip()

        return out

    # rw
    if s.startswith("rw"):
        out = {"op": "rw", "rewrites": []}
        m = re.search(r"\[(.*)\]", s)
        if m:
            content = m.group(1)
            for item in parse_bracket_list(content):
                out["rewrites"].append(parse_rw_item(item))
        return out

    # exact
    if s.startswith("exact "):
        return {"op": "exact", "expr": s[len("exact ") :].strip()}

    # apply
    if s.startswith("apply "):
        return {"op": "apply", "expr": s[len("apply ") :].strip()}

    # refine
    if s.startswith("refine "):
        return {"op": "refine", "expr": s[len("refine ") :].strip()}

    # trivial heads
    if s in ("rfl", "dsimp", "simp", "simpa"):
        return {"op": s}

    # fallback
    head = s.split()[0]
    return {"op": head, "raw": s}


def collect_symbols_from_norm(norm: Dict[str, Any]) -> List[str]:
    """Collect symbol names referenced by a normalized tactic.

    More Explanation
    Extracts lemma heads and expression heads to populate a symbol table,
    without attempting full Lean expression parsing.

    Args:
        norm: Normalized tactic dict.

    Returns:
        A list of symbol names.
    """
    syms: List[str] = []
    op = norm.get("op")

    if op in ("simp", "simpa"):
        for name in norm.get("lemmas_or_unfold", []) or []:
            name = name.strip()
            name = name.lstrip("←→").strip()
            m = re.match(rf"({_LEMMA_NAME})", name)
            if m:
                syms.append(m.group(1))

    elif op == "rw":
        for r in norm.get("rewrites", []) or []:
            lemma = r.get("lemma")
            if lemma:
                syms.append(lemma)

    elif op in ("apply", "exact", "refine"):
        head = (norm.get("expr") or "").strip()
        m = re.match(rf"({_LEMMA_NAME})", head)
        if m:
            syms.append(m.group(1))

    return syms


# =========================
# Trace & Linearization IR
# =========================


@dataclass
class Goal:
    """A single goal snapshot (pretty text only for now).

    More Explanation
    Keeps the pretty-printed goal/ctx text from the REPL. You can extend
    this to include skeletal AST in the future.

    Args:
        text: The raw goal text.
    """

    text: str


@dataclass
class TacticNode:
    """One tactic step in a proof trace.

    More Explanation
    Sequential nodes per REPL output with normalized tactic and goal text.
    Branching is not modeled yet but can be added by splitting nodes and
    assigning (parent, branch_idx).

    Args:
        id: Node id within the trace.
        parent: Parent node id (None for first node).
        branch_idx: Subgoal index under the parent (0 for linear traces).
        tactic: Normalized tactic (dict).
        state_before: Goal snapshot prior to tactic application.
        pos: Optional source position (line/column) if present in input.
        endPos: Optional end position (line/column) if present.
        ok: Whether the tactic was successful (heuristic; default True).
    """

    id: int
    parent: Optional[int]
    branch_idx: Optional[int]
    tactic: Dict[str, Any]
    state_before: Goal
    pos: Optional[Dict[str, int]] = None
    endPos: Optional[Dict[str, int]] = None
    ok: bool = True


@dataclass
class ProofTrace:
    """A complete proof trace extracted from one REPL run.

    More Explanation
    Includes metadata, a local symbol table, and the tactic nodes.

    Args:
        herald_id: Unique id from the CSV (or row index).
        file: Lean source path or label if available (optional).
        theorem: Theorem name (synthetic if unknown).
        env: REPL environment index (optional).
        symbols: Local symbol table mapping names to integer ids.
        nodes: Tactic nodes in execution order.
    """

    herald_id: int
    file: Optional[str]
    theorem: str
    env: Optional[int]
    symbols: Dict[str, int]
    nodes: List[TacticNode] = field(default_factory=list)


def linearize_trace(trace: ProofTrace, target_trim: int = 140) -> str:
    """Convert a proof trace to a bracketed linear string.

    More Explanation
    Emits control/tactic markers suitable for a byte-level model (e.g., ByT5).
    Includes a short `[TARGET]` line per step for supervision.

    Args:
        trace: The proof trace to linearize.
        target_trim: Max chars to keep from the last line of goal text.

    Returns:
        A linearized string representation.
    """
    parts: List[str] = []
    parts.append("[QED_START]")
    for n in trace.nodes:
        op = n.tactic.get("op", "")
        parts.append(f"[TACTIC:{op}]")

        if op in ("simp", "simpa"):
            if n.tactic.get("only"):
                parts.append("[ONLY]")
            for name in n.tactic.get("lemmas_or_unfold") or []:
                parts.append(f"[LEM {name}]")
            if "using" in n.tactic:
                parts.append("[USING] " + str(n.tactic["using"]))

        elif op == "rw":
            for r in n.tactic.get("rewrites", []) or []:
                dir_tok = "[LEFT]" if r.get("dir", -1) == -1 else "[RIGHT]"
                parts.append(dir_tok + f" [LEM {r.get('lemma', '')}]")

        elif op in ("apply", "exact", "refine"):
            parts.append("[EXPR] " + str(n.tactic.get("expr", "")))

        # rfl/dsimp: no extra args

        # Tiny target signature
        goal = n.state_before.text or ""
        if goal:
            sig = goal.splitlines()[-1][:target_trim]
            parts.append("[TARGET] " + sig)

    parts.append("[QED_END]")
    return "\\n".join(parts)


# ===================
# Dataset Construction
# ===================


def build_dataset_from_csv(
    csv_path: str,
    out_dir: str,
    repl_col: str = "REPL Output",
    herald_col: str = "Herald ID",
    target_trim: int = 140,
) -> Dict[str, str]:
    """Extract traces and sequences from the CSV and write artifacts.

    More Explanation
    Produces three outputs in `out_dir`:
        - traces.jsonl: one JSON object per trace (nodes, tactics, local symbols).
        - linearized_sequences.jsonl: one JSON object with herald_id and sequence.
        - symbols.json: global symbol->id mapping across all traces.

    Args:
        csv_path: Path to source CSV file.
        out_dir: Output directory path.
        repl_col: Column name containing REPL JSON blobs mixed with logs.
        herald_col: Column name for uniquely identifying each row.
        target_trim: Max chars to keep from the last goal line in linearization.

    Returns:
        Dict with file paths for the written artifacts.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    traces_path = os.path.join(out_dir, "traces.jsonl")
    seqs_path = os.path.join(out_dir, "linearized_sequences.jsonl")
    symbols_path = os.path.join(out_dir, "symbols.json")

    global_symbols: Dict[str, int] = {}
    next_sym_id = 1

    with (
        open(traces_path, "w", encoding="utf-8") as ft,
        open(seqs_path, "w", encoding="utf-8") as fs,
    ):
        for idx, row in df.iterrows():
            herald_id = int(row.get(herald_col, idx))
            blob = extract_json_blob(row.get(repl_col, ""))

            if not blob or "tactics" not in blob:
                # Skip rows without a valid REPL tactics JSON
                continue

            local_syms: Dict[str, int] = {}
            nodes: List[TacticNode] = []

            for i, t in enumerate(blob["tactics"]):
                raw_tac = (t.get("tactic") or "").strip()
                norm = normalize_tactic(raw_tac)

                # collect symbols for tables
                for name in collect_symbols_from_norm(norm):
                    if name not in global_symbols:
                        global_symbols[name] = next_sym_id
                        next_sym_id += 1
                    if name not in local_syms:
                        local_syms[name] = global_symbols[name]

                goal_text = t.get("goals") or t.get("proofState") or ""
                node = TacticNode(
                    id=i,
                    parent=(i - 1) if i > 0 else None,
                    branch_idx=0 if i > 0 else None,
                    tactic=norm,
                    state_before=Goal(text=goal_text),
                    pos=t.get("pos"),
                    endPos=t.get("endPos"),
                    ok=True,
                )
                nodes.append(node)

            trace = ProofTrace(
                herald_id=herald_id,
                file=None,
                theorem=f"herald_{herald_id}",
                env=blob.get("env"),
                symbols=local_syms,
                nodes=nodes,
            )

            ft.write(json.dumps(asdict(trace), ensure_ascii=False) + "\\n")
            fs.write(
                json.dumps(
                    {
                        "herald_id": trace.herald_id,
                        "sequence": linearize_trace(trace, target_trim=target_trim),
                    },
                    ensure_ascii=False,
                )
                + "\\n"
            )

    # write global symbols
    with open(symbols_path, "w", encoding="utf-8") as f:
        json.dump(global_symbols, f, ensure_ascii=False, indent=2)

    return {
        "traces": traces_path,
        "sequences": seqs_path,
        "symbols": symbols_path,
    }
