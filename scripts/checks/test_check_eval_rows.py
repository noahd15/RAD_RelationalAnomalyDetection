"""Synthetic tests for check_eval_rows.py: python scripts/checks/test_check_eval_rows.py"""
import os
import subprocess
import sys
import tempfile

CHECKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_eval_rows.py")

CSVS = {
    # LANL style: one global eval set, scored counts present.
    "ok.csv": "variant,seed,auroc,test_rows,test_pos,eval_fp,n_eval,n_pos\n"
              "a,0,0.9,63627.0,127.0,ad800208e39b,63627,127\n"
              "b,0,0.8,63627.0,127.0,ad800208e39b,63627,127\n",
    # Same fingerprint, but the model scored one positive fewer.
    "badpos.csv": "variant,seed,auroc,test_rows,test_pos,eval_fp,n_eval,n_pos\n"
                  "a,0,0.9,63627.0,126.0,ad800208e39b,63627,127\n",
    "noscored.csv": "model,seed,auroc,eval_fp,n_eval,n_pos\n"
                    "IF,0,0.5,ad800208e39b,63627,127\n",
    "nofp.csv": "seed,auroc,test_rows,test_pos\n0,0.9,63627,127\n",
    "nullfp.csv": "model,seed,auroc,eval_fp,n_eval,n_pos\nIF,0,0.5,,63627,127\n",
    # H&M/Amazon style: the eval set differs between seeds and agrees within each seed.
    "ps_ok.csv": "model,seed,auroc,eval_fp,n_eval,n_pos\n"
                 "RAD,0,0.6,aaa,5694,85\nIF,0,0.5,aaa,5694,85\n"
                 "RAD,1,0.6,bbb,5701,85\nIF,1,0.5,bbb,5701,85\n",
    # A second CSV for the same seeds, scored on the same rows (e.g. the RDL runner).
    "ps_ok_rdl.csv": "model,seed,auroc,test_rows,test_pos,eval_fp,n_eval,n_pos\n"
                     "RDL,0,0.6,5694,85,aaa,5694,85\nRDL,1,0.6,5701,85,bbb,5701,85\n",
    # Seed 0 has a model scored on other rows (the re-split RDL bug).
    "ps_bad.csv": "model,seed,auroc,eval_fp,n_eval,n_pos\n"
                  "RAD,0,0.6,aaa,5694,85\nRDL,0,0.5,ccc,2272,34\nRAD,1,0.6,bbb,5701,85\n",
    "ps_noseed.csv": "model,auroc,eval_fp,n_eval,n_pos\nRAD,0.6,aaa,5694,85\n",
}

CASES = [  # (arguments, expected exit code)
    (["ok.csv"], 0),
    (["--require-scored-counts", "ok.csv"], 0),
    (["ok.csv", "badpos.csv"], 1),
    (["--require-scored-counts", "noscored.csv"], 1),
    (["noscored.csv"], 0),
    (["nofp.csv"], 1),
    (["nullfp.csv"], 1),
    (["ps_ok.csv"], 1),                      # global mode rejects per-seed eval sets
    (["--per-seed", "ps_ok.csv"], 0),
    (["--per-seed", "ps_ok.csv", "ps_ok_rdl.csv"], 0),
    (["--per-seed", "ps_bad.csv"], 1),
    (["--per-seed", "ps_ok.csv", "ps_bad.csv"], 1),
    (["--per-seed", "ps_noseed.csv"], 1),
]


def main():
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        for name, text in CSVS.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        for args, want in CASES:
            rc = subprocess.run([sys.executable, CHECKER, *args], cwd=d, capture_output=True).returncode
            ok = rc == want
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} rc={rc} want={want}: {' '.join(args)}")
    if failed:
        sys.exit(f"{failed} case(s) failed")
    print(f"all {len(CASES)} cases passed")


if __name__ == "__main__":
    main()
