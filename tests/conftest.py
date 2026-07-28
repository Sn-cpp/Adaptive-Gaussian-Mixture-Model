import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock cupy before any gmm imports so CPU-only environments can load the module tree.
# The mock satisfies `import cupy as cp` in utils/timer.py, utils/gpu_warmup.py,
# and gmm/gpu/*.py without needing a real GPU.
try:
    import cupy
except ImportError:
    sys.modules['cupy'] = MagicMock()

import pytest
import numpy as np

from gmm import GMM_CPU, GMM_CPU_NUMBA


def _cupy_available():
    try:
        import cupy as cp
        if isinstance(cp, MagicMock):
            return False
        cp.zeros(1)
        return True
    except Exception:
        return False


requires_cupy = pytest.mark.skipif(
    not _cupy_available(),
    reason="CuPy/CUDA not available on this platform"
)


@pytest.fixture
def small_dims():
    return (60, 80)


@pytest.fixture
def default_params():
    return {
        'n_components': 5,
        'match_threshold': np.float32(3.5),
        'bg_threshold': np.float32(0.7),
        'alpha': np.float32(0.01),
    }
