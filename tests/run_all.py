"""
Runs every test module in tests/. Run with:
    python3 tests/run_all.py
"""
import subprocess
import sys
from pathlib import Path

TEST_FILES = [
    "test_dates_logic.py",
    "test_orchestrator_logic.py",
    "test_jobs_logic.py",
    "test_review_fixes.py",
]

def main():
    here = Path(__file__).resolve().parent
    failures = []
    for f in TEST_FILES:
        print(f"\n{'='*70}\nRunning {f}\n{'='*70}")
        result = subprocess.run([sys.executable, str(here / f)])
        if result.returncode != 0:
            failures.append(f)

    print(f"\n{'='*70}")
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"ALL {len(TEST_FILES)} TEST MODULES PASSED")


if __name__ == "__main__":
    main()
