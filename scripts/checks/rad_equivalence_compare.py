#!/usr/bin/env python3
"""Compare runs of scripts/checks/rad_equivalence_run.py, with no tolerance.

    python scripts/checks/rad_equivalence_compare.py same RUN_A RUN_B [--out REPORT.json]
    python scripts/checks/rad_equivalence_compare.py timing RUN [RUN ...]

`same` checks, for every arm the two runs share (both must have the same arms):
- stdout: every line is identical, in order, after these normalizations only:
  - the run folder and the code checkout path are replaced by <OUT> and <ROOT>;
  - "Done in N seconds." (relbench's load timer) has its number replaced;
  - lines that only one version of the code prints are dropped from both runs: the deterministic
    wrapper's mode line and the refactor's rad_hp:, rad_inputs_fp:, rad_best_fit_metric: and
    "[rules] rule table matches" lines.
- raw_results.pt (SAVE_RAW_RESULTS_PATH): every entry is identical. Tensors and arrays must have
  the same dtype, shape and bytes (so NaNs in the same places compare equal), floats must be ==,
  and strings are compared after the same path normalization.
  One exception, rf_detector_scores (the supervised RF detector baseline, scored with sklearn
  n_jobs=-1, whose per-tree summation order varies between runs by ~1e-16): it must have the same
  dtype and shape, an identical row ranking, and bit-identical AUROC and AUPRC against the run's
  test_labels; its max abs difference is reported. It is not a Table 4 row and nothing downstream
  reads it.
  Keys in NEW_ONLY_KEYS (the per-row feature/structure error terms, saved only by the refactored
  code) may be present in only one run; when both runs have one, it is compared bitwise.
- the rule table (rules/customer_rule_table.pt) has the same md5.
Prints the first differences and exits 1 if anything differs.

`timing` prints, per run and arm, the median and mean seconds per SUP and RAD epoch from
epoch_times.tsv. The first epoch of each kind is left out, since its time includes setup.
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

DROPPED_PREFIXES = ("deterministic_mode: true", "rad_hp: ", "rad_inputs_fp: ", "rad_best_fit_metric: ",
                    "[rules] rule table matches ")
_DONE_IN = re.compile(r"^Done in \d+(\.\d+)? seconds\.$")
MAX_SHOWN = 20
RANK_ONLY_KEYS = ("rf_detector_scores",)  # see the docstring
NEW_ONLY_KEYS = ("test_feat_err", "test_struct_err", "orig_test_feat_err", "orig_test_struct_err")


def _config(run: Path) -> dict:
    return json.loads((run / "run_config.json").read_text())


def normalize(text: str, out_dir: str, code_root: str) -> list[str]:
    lines = []
    for line in text.replace(out_dir, "<OUT>").replace(code_root, "<ROOT>").splitlines():
        if line.startswith(DROPPED_PREFIXES):
            continue
        lines.append("Done in <N> seconds." if _DONE_IN.match(line) else line)
    return lines


def compare_lines(a: list[str], b: list[str]) -> list[str]:
    diffs = [f"line {i + 1}: {x!r} != {y!r}" for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(a) != len(b):
        diffs.append(f"{len(a)} lines != {len(b)} lines; first extra: "
                     f"{(a[len(b)] if len(a) > len(b) else b[len(a)])!r}")
    return diffs


def compare_objects(a, b, path: str, norm_a, norm_b) -> list[str]:
    import numpy as np
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return [f"{path}: {type(a).__name__} != {type(b).__name__}"]
        a, b = a.detach().cpu().contiguous().numpy(), b.detach().cpu().contiguous().numpy()
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: {a.dtype}{a.shape} != {b.dtype}{b.shape}"]
        if a.dtype == object:
            return [d for i, (x, y) in enumerate(zip(a.ravel(), b.ravel()))
                    for d in compare_objects(x, y, f"{path}[{i}]", norm_a, norm_b)]
        if a.tobytes() != b.tobytes():
            n = int((np.ascontiguousarray(a).view(np.uint8).reshape(a.size, -1)
                     != np.ascontiguousarray(b).view(np.uint8).reshape(b.size, -1)).any(axis=1).sum()) if a.size else 0
            return [f"{path}: {n} of {a.size} values differ"]
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = [f"{path}: keys {sorted(map(str, set(a) ^ set(b)))} in only one run"] if set(a) != set(b) else []
        return diffs + [d for k in sorted(set(a) & set(b), key=str)
                        for d in compare_objects(a[k], b[k], f"{path}.{k}", norm_a, norm_b)]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{path}: {type(a).__name__}[{len(a)}] != {type(b).__name__}[{len(b)}]"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in compare_objects(x, y, f"{path}[{i}]", norm_a, norm_b)]
    if isinstance(a, str) and isinstance(b, str):
        return [] if norm_a(a) == norm_b(b) else [f"{path}: {a!r} != {b!r}"]
    if type(a) is not type(b):
        return [f"{path}: {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, float):
        same = a == b or (a != a and b != b)  # NaN == NaN
        return [] if same else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def compare_rank_only(a, b, labels, path: str) -> tuple[list[str], dict]:
    """Same dtype/shape, identical ranking (ties included), bit-identical AUROC and AUPRC."""
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.metrics import average_precision_score, roc_auc_score

    a, b, y = np.asarray(a), np.asarray(b), np.asarray(labels)
    if a.dtype != b.dtype or a.shape != b.shape:
        return [f"{path}: {a.dtype}{a.shape} != {b.dtype}{b.shape}"], {}
    info = {"n_values_differ": int((a != b).sum()), "max_abs_diff": float(np.abs(a.astype(float) - b.astype(float)).max()),
            "ranking_identical": bool(np.array_equal(rankdata(a), rankdata(b))),
            "auroc": [float(roc_auc_score(y, a)), float(roc_auc_score(y, b))],
            "auprc": [float(average_precision_score(y, a)), float(average_precision_score(y, b))]}
    diffs = [] if info["ranking_identical"] else [f"{path}: row ranking differs"]
    diffs += [f"{path}: {m} {info[m][0]!r} != {info[m][1]!r}" for m in ("auroc", "auprc") if info[m][0] != info[m][1]]
    return diffs, info


def _md5(path: Path) -> str | None:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def compare_runs(run_a: Path, run_b: Path) -> dict:
    import torch

    ca, cb = _config(run_a), _config(run_b)
    if ca["arms"] != cb["arms"] or ca["dataset"] != cb["dataset"] or ca["seed"] != cb["seed"]:
        raise RuntimeError(f"runs differ in dataset/seed/arms: {ca['dataset']} {ca['seed']} {ca['arms']} vs "
                           f"{cb['dataset']} {cb['seed']} {cb['arms']}")
    norm = {r: (lambda s, r=r, c=c: s.replace(str(r.resolve()), "<OUT>").replace(c["code_root"], "<ROOT>"))
            for r, c in ((run_a, ca), (run_b, cb))}
    report = {"run_a": str(run_a), "run_b": str(run_b), "arms": {}}
    for arm in ca["arms"]:
        res = {}
        la = normalize((run_a / arm / "stdout.log").read_text(), str(run_a.resolve()), ca["code_root"])
        lb = normalize((run_b / arm / "stdout.log").read_text(), str(run_b.resolve()), cb["code_root"])
        res["stdout_lines"] = [len(la), len(lb)]
        res["stdout_diffs"] = compare_lines(la, lb)
        ra = torch.load(run_a / arm / "raw_results.pt", map_location="cpu", weights_only=False)
        rb = torch.load(run_b / arm / "raw_results.pt", map_location="cpu", weights_only=False)
        res["raw_keys"] = sorted(map(str, ra)) if isinstance(ra, dict) else type(ra).__name__
        res["rank_only"], res["new_only_keys_in_one_run"] = {}, []
        if isinstance(ra, dict) and isinstance(rb, dict):
            ra, rb = dict(ra), dict(rb)
            for k in NEW_ONLY_KEYS:
                if (k in ra) != (k in rb):
                    res["new_only_keys_in_one_run"].append(k)
                    ra.pop(k, None)
                    rb.pop(k, None)
            for k in RANK_ONLY_KEYS:
                if k in ra and k in rb:  # a key in only one run is still reported below
                    d, info = compare_rank_only(ra.pop(k), rb.pop(k), ra["test_labels"], f"raw.{k}")
                    res["rank_only"][k] = {"diffs": d, **info}
        res["raw_diffs"] = compare_objects(ra, rb, "raw", norm[run_a], norm[run_b]) + [
            d for r in res["rank_only"].values() for d in r["diffs"]]
        ma, mb = _md5(run_a / arm / "rules" / "customer_rule_table.pt"), _md5(run_b / arm / "rules" / "customer_rule_table.pt")
        res["rule_table_md5"] = [ma, mb]
        res["identical"] = not res["stdout_diffs"] and not res["raw_diffs"] and ma == mb and ma is not None
        report["arms"][arm] = res
    report["identical"] = all(r["identical"] for r in report["arms"].values())
    return report


def epoch_seconds(tsv: Path) -> dict[str, list[float]]:
    rows = [line.split("\t") for line in tsv.read_text().splitlines()[1:]]
    out = {}
    for kind in ("SUP", "RAD"):
        t = [float(r[2]) for r in rows if r[0] == kind]
        out[kind] = [y - x for x, y in zip(t, t[1:])]  # epoch 2 onwards
    return out


def cmd_same(a) -> int:
    report = compare_runs(a.run_a, a.run_b)
    for arm, res in report["arms"].items():
        print(f"[{arm}] identical={res['identical']} stdout lines={res['stdout_lines']} "
              f"stdout diffs={len(res['stdout_diffs'])} raw diffs={len(res['raw_diffs'])} "
              f"rule tables={res['rule_table_md5']}")
        for d in (res["stdout_diffs"] + res["raw_diffs"])[:MAX_SHOWN]:
            print("    " + d)
        if res["new_only_keys_in_one_run"]:
            print(f"    in one run only (allowed): {res['new_only_keys_in_one_run']}")
        for k, r in res["rank_only"].items():
            if r.get("auroc"):
                print(f"    {k} (rank-only check): {r['n_values_differ']} values differ, max abs diff "
                      f"{r['max_abs_diff']:.3g}, ranking identical={r['ranking_identical']}, "
                      f"auroc {r['auroc'][0]!r}/{r['auroc'][1]!r}, auprc {r['auprc'][0]!r}/{r['auprc'][1]!r}")
    print(f"\n{'IDENTICAL' if report['identical'] else 'DIFFERENT'}: {a.run_a} vs {a.run_b}")
    if a.out:
        if a.out.exists():
            raise RuntimeError(f"{a.out} exists; refusing to overwrite it.")
        a.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["identical"] else 1


def cmd_timing(a) -> int:
    for run in a.runs:
        c = _config(run)
        for arm in c["arms"]:
            secs = epoch_seconds(run / arm / "epoch_times.tsv")
            parts = [f"{k}: n={len(v)} median={statistics.median(v):.1f}s mean={statistics.mean(v):.1f}s"
                     if v else f"{k}: n=0" for k, v in secs.items()]
            print(f"{run} [{c['dataset']} {arm} deterministic={c['deterministic_mode']}] " + " | ".join(parts))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("same")
    s.add_argument("run_a", type=Path)
    s.add_argument("run_b", type=Path)
    s.add_argument("--out", type=Path)
    t = sub.add_parser("timing")
    t.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    return cmd_same(a) if a.cmd == "same" else cmd_timing(a)


if __name__ == "__main__":
    sys.exit(main())
