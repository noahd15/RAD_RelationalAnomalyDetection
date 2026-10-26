#!/usr/bin/env python3
"""
Rebuild per-seed and mean/std ablation tables from a full-ablation run log.

The run log (stdout of run_full_ablation_hm_*.py / run_full_ablation_*.py) is the
authoritative record of a run: it contains the baselines output and both RAD
variants for every seed. This script re-derives the metric tables from it and
records the evaluation-set size (and, for runs that print it, the eval-set
fingerprint) on every row, failing if any seed mixes evaluation sets. The
Supervised MLP row is taken from the RAD (no BCE) section's FINAL COMPARISON.

Usage:
  python scripts/rebuild_ablation_from_log.py results/hm/full_ablation_hm_10seed/run.log
  (writes per_seed_metrics.csv, mean_std_metrics.csv, mean_std_metrics_pretty.csv next to the log)
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pandas as pd

from ablation_provenance import check_same_eval_set, parse_final_rows, parse_rad_final

ALIAS_MAP = {
    "IForest": "Isolation Forest",
    "VanillaAE": "AE",
    "AE-Rank": "AE+Rank",
    "DeepSVDD": "Deep SVDD",
    "DRL": "DRL",
}
BASELINE_RE = re.compile(r"^(IForest|DRL|DeepSVDD|VanillaAE|AE-Rank|Supervised RDL): (\{.*\})$", re.M)
BASELINE_VAL_RE = re.compile(r"Val\s+— total: ([\d,]+)\s+\| positives: ([\d,]+)")
SECTION_RE = re.compile(r"^\[seed=(\d+)\] Running (?:H&M )?(baselines script|RAD variant lambda_struct=([\d.]+))", re.M)


def _sections(text: str):
    marks = list(SECTION_RE.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        yield int(m.group(1)), m.group(3), text[m.end():end]


def _check_pak_granularity(row: dict) -> None:
    # P@k is (hits / n_pos), so a mismatched n_pos shows up as a non-integer hit count.
    hits = row["p@k"] * row["n_pos"]
    if abs(hits - round(hits)) > 1e-3 * max(1.0, row["n_pos"]) and row["p@k"] > 0:
        raise RuntimeError(f"{row['model']} seed={row['seed']}: P@k={row['p@k']} is inconsistent with n_pos={row['n_pos']}")


def rebuild(log_path: Path) -> pd.DataFrame:
    text = log_path.read_text()
    rows: list[dict] = []
    for seed, lam, body in _sections(text):
        if lam is None:
            val = BASELINE_VAL_RE.search(body)
            if not val:
                raise RuntimeError(f"seed={seed}: baselines section has no 'Val — total' line")
            n_eval, n_pos = (int(g.replace(",", "")) for g in val.groups())
            for name, metrics in BASELINE_RE.findall(body):
                met = ast.literal_eval(metrics)
                row = {"model": ALIAS_MAP.get(name, name), **{k: float(met[k]) for k in ("auroc", "auprc", "p@k", "lift")},
                       "n_eval": int(met.get("n_eval", n_eval)), "n_pos": int(met.get("n_pos", n_pos)),
                       "eval_fp": met.get("eval_fp"), "seed": seed, "source": "baselines_script"}
                _check_pak_granularity(row)
                rows.append(row)
        else:
            parsed = parse_rad_final(body)
            name = "RAD (no BCE)" if float(lam) == 0.0 else "RAD (with BCE)"
            rows.append({"model": name, **parsed, "seed": seed, "source": "rad_vs_script"})
            if float(lam) == 0.0 and parsed["eval_fp"] is not None:
                mlp = parse_final_rows(body)["MLP (supervised detector)"]
                row = {"model": "Supervised MLP", **mlp, "seed": seed, "source": "rad_vs_script"}
                _check_pak_granularity(row)
                rows.append(row)

    per_seed = pd.DataFrame(rows)[["model", "auroc", "auprc", "p@k", "lift", "n_eval", "n_pos", "eval_fp", "seed", "source"]]
    for seed, grp in per_seed.groupby("seed"):
        check_same_eval_set(grp, seed)
    return per_seed


def mean_std_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["auroc", "auprc", "p@k", "lift"]
    grp = df.groupby("model")[metrics]
    agg = pd.concat([grp.mean().add_suffix("_mean"), grp.std().add_suffix("_std")], axis=1).reset_index()
    agg = agg[["model"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]]
    agg["n_seeds"] = df.groupby("model")["seed"].nunique().reindex(agg["model"]).to_numpy()

    pretty = agg.copy()
    for m in metrics:
        pretty[m] = pretty.apply(lambda r: f"{r[f'{m}_mean']:.4f} +/- {r[f'{m}_std']:.4f}", axis=1)
    pretty = pretty[["model"] + metrics]

    agg = agg.sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False]).reset_index(drop=True)
    pretty = pretty.set_index("model").loc[agg["model"]].reset_index()
    return agg, pretty


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    log_path = Path(sys.argv[1])
    out_dir = log_path.parent

    per_seed = rebuild(log_path)
    mean_std, pretty = mean_std_report(per_seed)

    per_seed.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    mean_std.to_csv(out_dir / "mean_std_metrics.csv", index=False)
    pretty.to_csv(out_dir / "mean_std_metrics_pretty.csv", index=False)

    sizes = per_seed[["n_eval", "n_pos"]].drop_duplicates()
    print(f"[rebuild] {len(per_seed)} rows, {per_seed['seed'].nunique()} seeds, evaluation set(s): "
          + ", ".join(f"{e:,} users / {p} anomalies" for e, p in sizes.itertuples(index=False)))
    print(f"[rebuild] Wrote per-seed, mean/std and pretty tables to {out_dir}")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
