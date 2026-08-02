"""End-to-end background-blur pipeline built on the MOG2 models.

    frame -> MOG2 step (mask) -> morphological open -> separable blur ⨝ composite

`Pipeline` drives any CPU MOG2 model. `CUDAPipeline` keeps the whole thing on
the GPU and uses pinned host buffers plus two streams, so the upload of frame
*i* overlaps the compute of frame *i-1*.
"""
import os
import platform
import sys
import time

import numpy as np

from gmm.mog2_common import create_gaussian_kernel_1d, to_planar
from settings import BLUR_KSIZE, BLUR_SIGMA


def detect_platform():
    info = {
        'os': platform.system(),
        'arch': platform.machine(),
        'is_colab': 'google.colab' in sys.modules,
        'has_cuda': False,
        'cpu_count': os.cpu_count(),
    }
    try:
        from numba import cuda
        info['has_cuda'] = cuda.is_available()
        if info['has_cuda']:
            info['gpu_name'] = gpu_name()
    except Exception:
        pass
    return info


def gpu_name():
    """Device name as str — numba returns bytes on some versions, str on others."""
    from numba import cuda
    name = cuda.get_current_device().name
    return name.decode() if isinstance(name, bytes) else name


def available_models():
    """The MOG2 models that can actually run on this machine."""
    from gmm.cpu.GMM_cpu_mog2 import GMM_CPU_MOG2
    from gmm.cpu.GMM_cpu_numba_mog2 import GMM_CPU_NUMBA_MOG2
    models = {'sequential': GMM_CPU_MOG2, 'numba_cpu': GMM_CPU_NUMBA_MOG2}
    try:
        from gmm.gpu.GMM_cuda_mog2 import GMM_CUDA_MOG2, is_available
        if is_available() or os.environ.get('NUMBA_ENABLE_CUDASIM') == '1':
            models['cuda'] = GMM_CUDA_MOG2
    except Exception:
        pass
    return models


def best_model(models):
    for name in ('cuda', 'numba_cpu', 'sequential'):
        if name in models:
            return name
    return 'sequential'


class Pipeline:
    """CPU pipeline. Owns the model plus every scratch buffer; no reallocation."""

    NAME = 'CPU'

    def __init__(self, model_cls, first_frame, n_components=5, color=True,
                 morphology=True, ksize=BLUR_KSIZE, sigma=BLUR_SIGMA, **model_kw):
        from utils import blur_numba
        self.blur = blur_numba
        blur_numba.warmup()

        self.height, self.width = first_frame.shape[:2]
        self.color = color
        self.morphology = morphology
        self.model = model_cls(first_frame, n_components=n_components,
                               color=color, **model_kw)
        self.NAME = model_cls.__name__
        self.k1d = create_gaussian_kernel_1d(ksize, sigma)

        H, W = self.height, self.width
        self.mask_tmp = np.zeros((H, W), np.uint8)
        self.mask_clean = np.zeros((H, W), np.uint8)
        self.tmp = np.zeros((H, W, 3), np.float32)
        self.out = np.zeros((H, W, 3), np.uint8)

    def process(self, frame_bgr, update_alpha=-1.0):
        t = {}

        t0 = time.perf_counter()
        planar = to_planar(frame_bgr, self.color)
        t['convert'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mask = self.model.step(planar, update_alpha)
        t['gmm'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.morphology:
            mask = self.blur.morph_open(mask, self.mask_tmp, self.mask_clean)
        t['morph'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.blur.blur_h(frame_bgr, self.tmp, self.k1d)
        self.blur.blur_v_composite(self.tmp, frame_bgr, mask, self.out, self.k1d)
        t['blur_composite'] = time.perf_counter() - t0

        t['total'] = sum(t.values())
        t['fps'] = 1.0 / max(t['total'], 1e-9)
        return self.out, mask, t


class _Slot:
    """One pipeline stage: pinned host buffers + device buffers + a stream."""

    def __init__(self, H, W, C):
        from numba import cuda
        self.stream = cuda.stream()
        self.h_frame = cuda.pinned_array((H, W, 3), dtype=np.uint8)
        self.h_planar = cuda.pinned_array((C, H, W), dtype=np.float32)
        self.h_out = cuda.pinned_array((H, W, 3), dtype=np.uint8)
        self.h_mask = cuda.pinned_array((H, W), dtype=np.uint8)
        self.d_frame = cuda.device_array((H, W, 3), dtype=np.uint8, stream=self.stream)
        self.d_planar = cuda.device_array((C, H, W), dtype=np.float32, stream=self.stream)
        self.d_tmp = cuda.device_array((H, W, 3), dtype=np.float32, stream=self.stream)
        self.d_out = cuda.device_array((H, W, 3), dtype=np.uint8, stream=self.stream)
        self.event = cuda.event()


class CUDAPipeline:
    """Same `process(frame_bgr)` contract as `Pipeline`, plus a streamed variant."""

    NAME = 'GMM_CUDA_MOG2'

    def __init__(self, model_cls, first_frame, n_components=5, color=True,
                 morphology=True, ksize=BLUR_KSIZE, sigma=BLUR_SIGMA,
                 n_slots=2, **model_kw):
        from numba import cuda
        from utils import blur_cuda
        if ksize != BLUR_KSIZE:
            raise ValueError(
                f"the CUDA blur is compiled for ksize={BLUR_KSIZE}; change "
                "settings.BLUR_KSIZE and reimport to use another size")
        self.cuda = cuda
        self.bc = blur_cuda

        self.height, self.width = first_frame.shape[:2]
        self.color = color
        self.morphology = morphology
        self.model = model_cls(first_frame, n_components=n_components,
                               color=color, **model_kw)
        self.NAME = model_cls.__name__

        H, W, C = self.height, self.width, self.model.n_channels
        self.d_k1d = cuda.to_device(create_gaussian_kernel_1d(ksize, sigma))
        self.d_mask_tmp = cuda.device_array((H, W), dtype=np.uint8)
        self.d_mask_clean = cuda.device_array((H, W), dtype=np.uint8)
        self.slots = [_Slot(H, W, C) for _ in range(n_slots)]
        self.prev_event = None

        self.block = (blur_cuda.TILE_X, blur_cuda.TILE_Y)
        self.grid = ((W + blur_cuda.TILE_X - 1) // blur_cuda.TILE_X,
                     (H + blur_cuda.TILE_Y - 1) // blur_cuda.TILE_Y)

    def _enqueue(self, slot, frame_bgr, update_alpha):
        """Queue the upload, every kernel, and the download on this slot's stream."""
        args = self.model.next_args(update_alpha)
        st = slot.stream
        # the colour conversion stays on the host so every backend sees
        # bit-identical model input; both copies are async on this stream
        slot.h_frame[:] = frame_bgr
        slot.h_planar[:] = to_planar(frame_bgr, self.color)
        slot.d_frame.copy_to_device(slot.h_frame, stream=st)
        slot.d_planar.copy_to_device(slot.h_planar, stream=st)

        # the GMM state is shared, so serialise the model updates across slots
        if self.prev_event is not None:
            self.prev_event.wait(stream=st)

        mask = self.model.step_device(slot.d_planar, args, stream=st)

        if self.morphology:
            self.bc.erode_kernel[self.grid, self.block, st](mask, self.d_mask_tmp)
            self.bc.dilate_kernel[self.grid, self.block, st](
                self.d_mask_tmp, self.d_mask_clean)
            mask = self.d_mask_clean

        self.bc.blur_h_kernel[self.grid, self.block, st](
            slot.d_frame, slot.d_tmp, self.d_k1d)
        self.bc.blur_v_composite_kernel[self.grid, self.block, st](
            slot.d_tmp, slot.d_frame, mask, slot.d_out, self.d_k1d)

        slot.d_out.copy_to_host(slot.h_out, stream=st)
        mask.copy_to_host(slot.h_mask, stream=st)
        slot.event.record(stream=st)
        self.prev_event = slot.event

    def process(self, frame_bgr, update_alpha=-1.0):
        slot = self.slots[0]
        t0 = time.perf_counter()
        self._enqueue(slot, frame_bgr, update_alpha)
        slot.stream.synchronize()
        total = time.perf_counter() - t0
        return slot.h_out, slot.h_mask, {'total': total, 'fps': 1.0 / max(total, 1e-9)}

    def process_stream(self, frames, update_alpha=-1.0):
        """Pipelined path: the upload of frame i overlaps the compute of frame i-1."""
        pending = []
        for frame in frames:
            if len(pending) == len(self.slots):
                slot = pending.pop(0)
                slot.stream.synchronize()
                yield slot.h_out.copy(), slot.h_mask.copy()
            else:
                slot = self.slots[len(pending)]
            self._enqueue(slot, frame, update_alpha)
            pending.append(slot)
        for slot in pending:
            slot.stream.synchronize()
            yield slot.h_out.copy(), slot.h_mask.copy()


def make_pipeline(model_cls, first_frame, **kw):
    """Pick the CUDA pipeline automatically for the CUDA model."""
    if model_cls.__name__ == 'GMM_CUDA_MOG2':
        return CUDAPipeline(model_cls, first_frame, **kw)
    return Pipeline(model_cls, first_frame, **kw)
