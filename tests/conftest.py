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


@pytest.fixture
def small_dims():
    return (60, 80)
