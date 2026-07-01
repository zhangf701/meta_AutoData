"""Quick verification script for WeakSolver with local Ollama."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from data_generation.solvers.weak_solver import WeakSolver
from data_generation.config import WEAK_MODEL, OLLAMA_BASE_URL


def main():
    print("=" * 60)
    print("WeakSolver Verification")
    print("=" * 60)
    print(f"  Model:    {WEAK_MODEL}")
    print(f"  Base URL: {OLLAMA_BASE_URL}")

    ws = WeakSolver()

    test_queries = [
        "电力系统暂态稳定的定义是什么？请用一句话回答。",
        "Explain transient stability in power systems in one sentence.",
        "发电机转子运动方程中，加速功率等于什么？",
    ]

    all_pass = True
    for i, q in enumerate(test_queries):
        print(f"\n--- Test {i+1}: {q[:40]}... ---")
        try:
            result = ws.solve(q)
            ok = len(result) > 10 and "Error" not in result
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{status}] Response ({len(result)} chars): {result[:150]}")
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED - WeakSolver is configured and working.")
    else:
        print("SOME TESTS FAILED - Check Ollama server status.")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
