"""Smoke test: verify the packages the project needs import correctly."""
import sys


def test_imports():
    import numpy as np
    print(f"numpy {np.__version__}")

    import numba
    print(f"numba {numba.__version__}")

    import cv2
    print(f"opencv {cv2.__version__}")

    try:
        import cupy
        # conftest substitutes a MagicMock for cupy on CPU-only machines
        print(f"cupy {getattr(cupy, '__version__', '(not installed — mocked)')}")
    except ImportError:
        print("cupy not installed (only the GMM_CUPY_* models need it)")

    print(f"\nPython {sys.version}")
    print("All imports OK")


def test_numba_jit():
    import numpy as np
    from numba import njit

    @njit(cache=True)
    def add(a, b):
        return a + b

    assert np.allclose(add(np.ones(100, np.float32), np.ones(100, np.float32)), 2.0)
    print("Numba JIT compilation OK")


def test_cuda_available():
    try:
        from numba import cuda
        if cuda.is_available():
            from pipeline import gpu_name
            print(f"CUDA available: {gpu_name()}")
        else:
            print("CUDA not available (expected on macOS — "
                  "use NUMBA_ENABLE_CUDASIM=1 locally, Colab for real timings)")
    except Exception as e:
        print(f"CUDA check skipped: {e}")


if __name__ == "__main__":
    test_imports()
    test_numba_jit()
    test_cuda_available()
