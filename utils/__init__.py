from .timer import cpu_timer, gpu_timer
from .post_processing import (mask_refiner, background_subtractor,
                              blur_ksize_for, close_ksize_for, fill_holes)
from .iou import compute_iou
from .metric_monitor import Monitor

# cupy is optional — everything above must work on a machine without a GPU,
# so the CuPy-only helpers degrade to None instead of breaking the import.
try:
    from .gpu_warmup import cp_gpu_warmup
except ImportError:                     # pragma: no cover - cupy not installed
    cp_gpu_warmup = None
