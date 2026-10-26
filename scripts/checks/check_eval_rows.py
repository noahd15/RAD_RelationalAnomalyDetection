"""Check that every per-seed row of one dataset was scored on the same evaluation rows.

usage: check_eval_rows.py [--require-scored-counts] [--per-seed] CSV [CSV ...]

For every row of every CSV: eval_fp, n_eval and n_pos are non-null, and all rows share one
(eval_fp, n_eval, n_pos). With --per-seed (H&M, Amazon: the reporting rows are drawn per seed),
the rows of each seed must share one (eval_fp, n_eval, n_pos) instead; every CSV needs a seed
column. If a CSV has test_rows/test_pos (counts taken from the y_true that was actually scored),
each row must have test_rows == n_eval and test_pos == n_pos.
--require-scored-counts makes test_rows/test_pos mandatory (used for LANL, where eval_fp is
computed from auth.test_mask and so does not by itself show what each model scored).
Exits 1 and prints every failing row on any failure.
Tests: python scripts/checks/test_check_eval_rows.py
"""
import sys
import pandas as pd

args = sys.argv[1:]
require = "--require-scored-counts" in args
per_seed = "--per-seed" in args
paths = [a for a in args if not a.startswith("--")]
if not paths:
    sys.exit("no CSVs given")

bad, keys = [], {}
for p in paths:
    df = pd.read_csv(p)
    label = [c for c in ("variant", "model") if c in df.columns]
    for c in ("eval_fp", "n_eval", "n_pos"):
        if c not in df.columns:
            bad.append(f"{p}: missing column {c}")
    has_scored = {"test_rows", "test_pos"} <= set(df.columns)
    if per_seed and "seed" not in df.columns:
        bad.append(f"{p}: missing column seed (needed for --per-seed)")
    if require and not has_scored:
        bad.append(f"{p}: missing test_rows/test_pos")
    if any(b.startswith(f"{p}:") for b in bad):
        continue
    if df.empty:
        bad.append(f"{p}: no rows")
        continue
    for i, r in df.iterrows():
        tag = f"{p} row {i} " + " ".join(f"{c}={r[c]}" for c in label + ["seed"] if c in df.columns)
        if pd.isna(r["eval_fp"]) or pd.isna(r["n_eval"]) or pd.isna(r["n_pos"]):
            bad.append(f"{tag}: null eval_fp/n_eval/n_pos")
            continue
        group = int(r["seed"]) if per_seed else "all"
        keys.setdefault(group, set()).add((str(r["eval_fp"]), int(r["n_eval"]), int(r["n_pos"])))
        if has_scored:
            if pd.isna(r["test_rows"]) or pd.isna(r["test_pos"]):
                bad.append(f"{tag}: null test_rows/test_pos")
            elif int(r["test_rows"]) != int(r["n_eval"]) or int(r["test_pos"]) != int(r["n_pos"]):
                bad.append(f"{tag}: test_rows={r['test_rows']} test_pos={r['test_pos']} "
                           f"vs n_eval={r['n_eval']} n_pos={r['n_pos']}")
    print(f"{p}: {len(df)} rows, scored-count check {'on' if has_scored else 'n/a (no test_rows/test_pos)'}")

for group, ks in sorted(keys.items(), key=lambda kv: str(kv[0])):
    if len(ks) > 1:
        bad.append(f"seed={group}: rows disagree on (eval_fp, n_eval, n_pos): {sorted(ks)}")
if bad:
    print("EVAL-ROW CHECK FAILED:")
    print("\n".join("  " + b for b in bad))
    sys.exit(1)
for group, ks in sorted(keys.items(), key=lambda kv: str(kv[0])):
    print(f"EVAL-ROW CHECK OK seed={group}: {next(iter(ks))}")
