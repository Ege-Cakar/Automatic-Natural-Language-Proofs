#!/usr/bin/env python3
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ====================== CONFIG ======================
FILE_PATTERN = "./datasets_training/NuminaMath-LEAN/dataset_step3/dataset*.csv"

REQUIRE_FL_PROOF = True
REQUIRE_COMPILES = False
REQUIRE_HAS_NL = False

PROJECT_NAME = "TmpProjDir"
BASE_PROJECT_DIR = Path(PROJECT_NAME)

MATHLIB_VERSION = "v4.11.0"
REPL_COMMIT = "adbbfcb9d4e61c12db96c45d227de92f21cc17dd"
FORCE_CLEAN_SETUP = True

RUN_REPL_ON_SUCCESS = True
TIMEOUT_SECS = 300

OUT_CSV_PREFIX = (
    "compile_results"  # final file becomes f"{OUT_CSV_PREFIX}_s{start}_n{limit}.csv"
)
SAVE_EVERY = 50  # checkpoint writes within a batch
# ====================================================


def run_cmd(cmd, cwd=".", timeout=TIMEOUT_SECS):
    try:
        p = subprocess.run(
            cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout
        )
        return p.returncode == 0, p.stdout + p.stderr
    except Exception as exc:
        return False, str(exc)


def bootstrap_project():
    """Bootstrap the base project directory for reference."""
    project_dir = BASE_PROJECT_DIR
    if FORCE_CLEAN_SETUP and project_dir.exists():
        print("** Removing old project...")
        shutil.rmtree(project_dir)

    if not project_dir.exists():
        print("** Initialising empty Lean project...")
        ok, out = run_cmd(["lake", "new", PROJECT_NAME])
        if not ok:
            sys.exit(f"X  'lake new' failed:\n{out}")

    lakefile = (
        f"import Lake\nopen Lake DSL\n\n"
        f'package "{PROJECT_NAME}" where\n'
        f"@[default_target]\nlean_lib «{PROJECT_NAME}» where\n\n"
        f"require mathlib from git "
        f'"https://github.com/leanprover-community/mathlib4" @"{MATHLIB_VERSION}"\n'
        f'require "REPL" from git '
        f'"https://github.com/leanprover-community/repl.git" @ "{REPL_COMMIT}"\n'
    )
    (project_dir / "lakefile.lean").write_text(lakefile, encoding="utf-8")

    for cmd in (["lake", "update"], ["lake", "exe", "cache", "get"]):
        ok, out = run_cmd(cmd, cwd=str(project_dir))
        if not ok:
            sys.exit(f"X  {' '.join(cmd)} failed:\n{out}")

    try:
        toolchain = (project_dir / "lean-toolchain").read_text().strip()
        print(f"✓ Using toolchain: {toolchain}")
    except Exception:
        print("! Could not read lean-toolchain; continuing.")


_HEADER_STARTERS = re.compile(
    r"""^\s*(?:import\s+\S+|open\s+.+|namespace\s+\S+|end\s+\S+|set_option\s+\S+\s+\S+|
             noncomputable\s+section|section|variable[s]?\s+|local\s+notation\s+.+|notation\s+.+|abbrev\s+.+)\s*$""",
    re.X,
)
_DECL_START = re.compile(
    r"""^\s*(?:theorem|lemma|example|def|instance|structure|class|inductive|mutual)\b"""
)


def split_header_body(src: str) -> tuple[list[str], list[str]]:
    lines = src.splitlines()
    header, body, in_header = [], [], True
    for ln in lines:
        if in_header:
            if _HEADER_STARTERS.match(ln) and not _DECL_START.match(ln):
                header.append(ln.rstrip())
                continue
            in_header = False
        body.append(ln.rstrip())
    return header, body


def normalize_row_to_source(row_header: str | None, row_formal: str) -> str:
    row_header = (row_header or "").strip()
    row_formal = (row_formal or "").strip()
    if row_header:
        header_src, body_src = row_header, row_formal
    else:
        hdr, body = split_header_body(row_formal)
        header_src, body_src = "\n".join(hdr), "\n".join(body)

    import_lines, other_header = [], []
    for _line in header_src.splitlines():
        if re.match(r"^\s*import\s+\S+", _line):
            import_lines.append(_line.rstrip())
        else:
            other_header.append(_line.rstrip())
    if not import_lines:
        import_lines = ["import Mathlib"]

    hdr2, body2 = split_header_body(body_src)
    if hdr2:
        for _line in hdr2:
            if _line.startswith("import "):
                import_lines.append(_line)
            else:
                other_header.append(_line)
        body_src = "\n".join(body2)

    def dedup(seq):
        seen, out = set(), []
        for x in seq:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    import_lines = dedup(import_lines)
    other_header = dedup(other_header)

    return "\n".join(
        [
            *import_lines,
            "",
            f"namespace {PROJECT_NAME}",
            "",
            *other_header,
            "",
            body_src,
            "",
            f"end {PROJECT_NAME}",
            "",
        ]
    )


# These functions are now handled within process_single_row for each worker


def run_repl(project_dir: Path, lean_src_rel: str):
    """Run REPL for a specific project directory.

    Args:
        project_dir: Path to the project directory
        lean_src_rel: Relative path to the Lean source file

    Returns:
        Tuple of (success: bool, output: str)
    """
    cmd = [
        "sh",
        "-c",
        f'echo \'{{"path": "{lean_src_rel}", "allTactics": true}}\' | lake exe repl',
    ]
    return run_cmd(cmd, cwd=str(project_dir))


def process_single_row(args_tuple):
    """Process a single row in a separate worker process.

    Args:
        args_tuple: Tuple containing (row_data, worker_id, global_row_idx, local_idx)

    Returns:
        Dictionary containing processing results
    """
    row_data, worker_id, global_row_idx, local_idx = args_tuple

    # Create worker-specific project directory
    project_dir = BASE_PROJECT_DIR.parent / f"{PROJECT_NAME}_worker_{worker_id}"
    src_dir = project_dir / PROJECT_NAME
    lean_src = src_dir / "Basic.lean"
    lean_src_rel = lean_src.relative_to(project_dir).as_posix()

    try:
        # Setup worker-specific project if it doesn't exist
        if not project_dir.exists() or FORCE_CLEAN_SETUP:
            setup_worker_project(project_dir, worker_id)

        # Process the row
        rec = {
            "id": str(row_data.get("id") or f"row_{global_row_idx}"),
            "header": row_data.get("header"),
            "formal_proof": row_data.get("formal_proof") or "",
            "informal_statement": row_data.get("informal_statement") or "",
            "informal_proof": row_data.get("informal_proof") or "",
        }

        lean_src_content = normalize_row_to_source(rec["header"], rec["formal_proof"])

        # Write source to worker-specific directory
        src_dir.mkdir(parents=True, exist_ok=True)
        lean_src.write_text(lean_src_content, encoding="utf-8")

        # Build
        ok_build, out_build = run_cmd(["lake", "build"], cwd=str(project_dir))

        if ok_build and RUN_REPL_ON_SUCCESS:
            ok_repl, out_repl = run_repl(project_dir, lean_src_rel)
        else:
            ok_repl, out_repl = (
                False,
                "REPL skipped (build failed)" if not ok_build else "REPL skipped",
            )

        return {
            "ID": rec["id"],
            "LEAN Source": lean_src_content,
            "Build OK": str(ok_build),
            "Build Output": out_build,
            "REPL OK": str(ok_repl),
            "REPL Output": out_repl,
            "Global Row Index": global_row_idx,
            "Worker ID": worker_id,
            "Local Index": local_idx,
            "Success": ok_build,
        }

    except Exception as e:
        return {
            "ID": str(row_data.get("id") or f"row_{global_row_idx}"),
            "LEAN Source": "",
            "Build OK": "False",
            "Build Output": f"Worker error: {str(e)}",
            "REPL OK": "False",
            "REPL Output": f"Worker error: {str(e)}",
            "Global Row Index": global_row_idx,
            "Worker ID": worker_id,
            "Local Index": local_idx,
            "Success": False,
        }


def setup_worker_project(project_dir: Path, worker_id: int):
    """Setup a worker-specific Lean project.

    Args:
        project_dir: Path to the worker's project directory
        worker_id: Unique worker identifier
    """
    # Clean up if needed
    if FORCE_CLEAN_SETUP and project_dir.exists():
        shutil.rmtree(project_dir)

    if not project_dir.exists():
        # Initialize project
        ok, out = run_cmd(["lake", "new", PROJECT_NAME], cwd=str(project_dir.parent))
        if not ok:
            raise Exception(f"Worker {worker_id}: 'lake new' failed:\n{out}")

        # Rename to worker-specific directory
        temp_dir = project_dir.parent / PROJECT_NAME
        if temp_dir.exists() and temp_dir != project_dir:
            temp_dir.rename(project_dir)

    # Create lakefile
    lakefile = (
        f"import Lake\nopen Lake DSL\n\n"
        f'package "{PROJECT_NAME}" where\n'
        f"@[default_target]\nlean_lib «{PROJECT_NAME}» where\n\n"
        f"require mathlib from git "
        f'"https://github.com/leanprover-community/mathlib4" @"{MATHLIB_VERSION}"\n'
        f'require "REPL" from git '
        f'"https://github.com/leanprover-community/repl.git" @ "{REPL_COMMIT}"\n'
    )
    (project_dir / "lakefile.lean").write_text(lakefile, encoding="utf-8")

    # Setup dependencies
    for cmd in (["lake", "update"], ["lake", "exe", "cache", "get"]):
        ok, out = run_cmd(cmd, cwd=str(project_dir))
        if not ok:
            raise Exception(f"Worker {worker_id}: {' '.join(cmd)} failed:\n{out}")

    print(f"✓ Worker {worker_id} project setup complete")


def load_and_filter_rows():
    files = glob.glob(FILE_PATTERN)
    print("Read", len(files), "files!")
    if not files:
        sys.exit("No CSV files matched FILE_PATTERN.")
    df_list = [pd.read_csv(f, dtype=str) for f in files]
    final_df = pd.concat(df_list, ignore_index=True)

    for col in [
        "has_fl_proof?",
        "fl_proof_compiles?",
        "has_nl_proof?",
        "header",
        "formal_proof",
        "informal_statement",
        "informal_proof",
        "id",
    ]:
        if col not in final_df.columns:
            final_df[col] = None

    mask = pd.Series([True] * len(final_df))
    if REQUIRE_FL_PROOF:
        mask &= (final_df["has_fl_proof?"] == "yes") & final_df["formal_proof"].notna()
    if REQUIRE_COMPILES:
        mask &= final_df["fl_proof_compiles?"] == "yes"
    if REQUIRE_HAS_NL:
        mask &= final_df["has_nl_proof?"] != "no"

    filtered = final_df[mask].reset_index(drop=True)
    return filtered  # return the DataFrame so we can slice by batch


def parse_args():
    ap = argparse.ArgumentParser(description="Batch Lean build/REPL harness")
    ap.add_argument("--batch-size", type=int, default=500, help="Rows per output CSV")
    ap.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help="0-based batch index; start = batch_index*batch_size",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=None,
        help="Absolute start row (overrides batch-index if set)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Absolute limit (overrides batch-size if set)",
    )
    ap.add_argument(
        "--out-prefix", type=str, default=OUT_CSV_PREFIX, help="Output CSV prefix"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count)",
    )
    ap.add_argument(
        "--serial",
        action="store_true",
        help="Run in serial mode (disable multiprocessing)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Derive slice
    if args.start is not None:
        start = max(0, args.start)
    elif args.batch_index is not None:
        start = max(0, args.batch_index * (args.batch_size if args.batch_size else 500))
    else:
        start = 0

    limit = args.limit if args.limit is not None else (args.batch_size or 500)
    if limit <= 0:
        sys.exit("limit/batch-size must be positive")

    # Determine number of workers
    if args.serial:
        num_workers = 1
    else:
        num_workers = args.workers if args.workers is not None else os.cpu_count()

    print(f"Running with {num_workers} worker(s)")

    # Prepare base project for reference (workers will create their own)
    if not args.serial:
        bootstrap_project()

    # Load once, then slice
    df = load_and_filter_rows()
    total_rows = len(df)
    end = min(total_rows, start + limit)
    if start >= total_rows:
        print(f"Start {start} >= total rows {total_rows}; nothing to do.")
        return

    print(
        f"Selected rows (after filters): {total_rows} | Taking slice [{start}:{end}) of size {end - start}"
    )

    # Output file for this batch
    out_csv_path = f"{args.out_prefix}_s{start}_n{end - start}.csv"
    results = []

    if args.serial:
        # Serial processing (original logic)
        print("Running in serial mode...")
        bootstrap_project()
        project_dir = BASE_PROJECT_DIR
        src_dir = project_dir / PROJECT_NAME
        lean_src = src_dir / "Basic.lean"
        lean_src_rel = lean_src.relative_to(project_dir).as_posix()

        built_ok = repl_ok = 0
        for local_idx, (i, row) in enumerate(df.iloc[start:end].iterrows(), start=1):
            rec = {
                "id": str(row.get("id") or f"row_{i}"),
                "header": row.get("header"),
                "formal_proof": row.get("formal_proof") or "",
                "informal_statement": row.get("informal_statement") or "",
                "informal_proof": row.get("informal_proof") or "",
            }

            lean_src_content = normalize_row_to_source(
                rec["header"], rec["formal_proof"]
            )

            # Write source
            src_dir.mkdir(parents=True, exist_ok=True)
            lean_src.write_text(lean_src_content, encoding="utf-8")

            # Build
            ok_build, out_build = run_cmd(["lake", "build"], cwd=str(project_dir))
            if ok_build:
                built_ok += 1
                if RUN_REPL_ON_SUCCESS:
                    ok_repl, out_repl = run_repl(project_dir, lean_src_rel)
                    repl_ok += int(ok_repl)
                else:
                    ok_repl, out_repl = False, "REPL skipped"
            else:
                ok_repl, out_repl = False, "REPL skipped (build failed)"

            status = "✓" if ok_build else "✗"
            print("\n" + "=" * 100)
            print(f"[{status}] id={rec['id']} (batch_row={local_idx}/{end - start})")
            print("-" * 40)
            print(">> Lean source:")
            print("-" * 40)
            print(textwrap.indent(lean_src_content.strip(), "   "))
            print("-" * 40)
            if not ok_build:
                print("Build error (first lines):")
                print("\n".join(out_build.splitlines()[:25]))

            results.append(
                {
                    "ID": rec["id"],
                    "LEAN Source": lean_src_content,
                    "Build OK": str(ok_build),
                    "Build Output": out_build,
                    "REPL OK": str(ok_repl),
                    "REPL Output": out_repl,
                    "Global Row Index": i,
                }
            )

            # Periodic checkpoint
            if SAVE_EVERY and (local_idx % SAVE_EVERY == 0):
                pd.DataFrame(results).to_csv(
                    out_csv_path, index=False, encoding="utf-8"
                )
                print(f"[checkpoint] wrote {len(results)} rows to {out_csv_path}")
    else:
        # Parallel processing
        print(f"Running in parallel mode with {num_workers} workers...")

        # Prepare work items
        work_items = []
        for local_idx, (i, row) in enumerate(df.iloc[start:end].iterrows(), start=1):
            assert isinstance(num_workers, int)
            worker_id = (local_idx - 1) % max(1, num_workers)
            work_items.append((row, worker_id, i, local_idx))

        print(f"Processing {len(work_items)} items...")

        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all work
            future_to_item = {
                executor.submit(process_single_row, item): item for item in work_items
            }

            completed = 0
            built_ok = repl_ok = 0

            # Collect results as they complete
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if result["Success"]:
                        built_ok += 1
                        if result["REPL OK"] == "True":
                            repl_ok += 1

                    # Progress update
                    if completed % 10 == 0 or completed == len(work_items):
                        print(
                            f"Completed {completed}/{len(work_items)} ({completed / len(work_items) * 100:.1f}%)"
                        )

                    # Periodic checkpoint
                    if SAVE_EVERY and (completed % SAVE_EVERY == 0):
                        pd.DataFrame(results).to_csv(
                            out_csv_path, index=False, encoding="utf-8"
                        )
                        print(
                            f"[checkpoint] wrote {completed} results to {out_csv_path}"
                        )

                except Exception as e:
                    print(f"Error processing work item: {e}")
                    completed += 1

    # Sort results by local index to maintain order
    if not args.serial:
        results.sort(key=lambda x: x["Local Index"])

    # Final write for the batch
    final_df = pd.DataFrame(results)
    # Remove helper columns used for parallel processing
    if "Worker ID" in final_df.columns:
        final_df = final_df.drop(["Worker ID", "Local Index", "Success"], axis=1)

    final_df.to_csv(out_csv_path, index=False, encoding="utf-8")
    print(
        f"\n*** Built {built_ok}/{end - start} successfully ({built_ok / (end - start) * 100:.1f}%)."
    )
    print(
        f"*** REPL ok {repl_ok}/{end - start} ({repl_ok / (end - start) * 100:.1f}%)."
    )
    print(f"Saved batch to {out_csv_path}")


if __name__ == "__main__":
    main()
