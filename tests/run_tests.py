"""Standalone test runner discovering and executing all test_* functions in tests/."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import time
from pathlib import Path

# Add clauselens to path
TESTS_DIR = Path(__file__).resolve().parent
CLAUSELENS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(CLAUSELENS_DIR))


def run_all_tests() -> int:
    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    passed = 0
    failed = 0
    errors = []

    print(f"\n==================== Running ClauseLens Test Suite ====================")
    start_time = time.time()

    for tf in test_files:
        module_name = tf.stem
        spec = importlib.util.spec_from_file_location(module_name, tf)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod

        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"ERROR importing {tf.name}: {e}")
            failed += 1
            errors.append((tf.name, "ImportError", str(e)))
            continue

        test_funcs = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        for name, func in test_funcs:
            t_sub = time.time()
            try:
                func()
                elapsed = time.time() - t_sub
                print(f"PASSED {tf.name}::{name} ({elapsed:.3f}s)")
                passed += 1
            except Exception as e:
                elapsed = time.time() - t_sub
                print(f"FAILED {tf.name}::{name} ({elapsed:.3f}s) -> {e}")
                failed += 1
                errors.append((tf.name, name, str(e)))

    total_time = time.time() - start_time
    print(f"\n==================== Summary: {passed} passed, {failed} failed in {total_time:.2f}s ====================")

    if errors:
        print("\nFailures:")
        for file_name, func_name, err in errors:
            print(f"  - {file_name}::{func_name}: {err}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())

