"""
Reproduce every table and figure in the manuscript, in one command.

Runs each analysis script with this interpreter and a non-interactive matplotlib
backend, reporting PASS/FAIL and wall-clock time. Tables are printed by the
analysis scripts; figures are (re)written to plots/. Use --quick to skip the
slowest scripts (nested CV, the full grouped-split / RF sweeps).

    .venv/Scripts/python.exe scripts/reproduce_all.py [--quick]
"""

import os
import sys
import time
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# (script, slow?) — slow ones are skipped under --quick
SCRIPTS = [
    ("cv_evaluation.py", False),          # Tables 1, 2 (non-zero CV)
    ("cv_rf_baseline.py", True),          # Table 2b (RF baseline)
    ("nested_cv.py", True),               # Table 6 (nested CV + significance)
    ("feature_ablation.py", False),       # Table 7 (ablation + error analysis)
    ("cv_grouped_splits.py", True),       # Table 5 (grouped splits)
    ("double_perovskite_analysis.py", False),
    ("better_model.py", False),           # negative-result experiments
    ("learning_curve.py", False),         # Figure 3
    ("feature_importance.py", False),     # Figure 4 (SHAP)
    ("expt_gap_parity.py", False),        # Figure 1 (expt_gap parity)
    ("make_generalization_figure.py", False),   # Figure 2
]


def main():
    quick = "--quick" in sys.argv
    env = dict(os.environ, MPLBACKEND="Agg")
    results = []
    for script, slow in SCRIPTS:
        if quick and slow:
            results.append((script, "SKIP", 0.0))
            continue
        print(f"\n{'=' * 70}\nRunning {script} ...")
        t0 = time.time()
        proc = subprocess.run([PY, str(HERE / script)], env=env,
                              capture_output=True, text=True)
        dt = time.time() - t0
        ok = proc.returncode == 0
        results.append((script, "PASS" if ok else "FAIL", dt))
        # echo the script's stdout minus progress bars
        for line in proc.stdout.splitlines():
            if "it/s" not in line and "it]" not in line and line.strip():
                print("   " + line)
        if not ok:
            print("   !! STDERR:", proc.stderr.strip()[-500:])

    print(f"\n{'=' * 70}\nSUMMARY")
    for script, status, dt in results:
        print(f"  {status:5}  {script:<32} {dt:6.1f}s")
    failed = [s for s, st, _ in results if st == "FAIL"]
    print("\nAll reproduced." if not failed else f"\nFAILURES: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
