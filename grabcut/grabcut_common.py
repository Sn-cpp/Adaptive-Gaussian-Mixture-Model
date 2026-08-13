import numpy as np
from time import perf_counter

from settings import GAMMA, BLUR_KSIZE

class GrabCut_Base:
    def __init__(self, H, W, bg_model, fg_model):
        self.H = H
        self.W = W
        N = H * W

        self._bg_gmm  = bg_model
        self._fg_gmm  = fg_model

        # GC label map
        self._gc_mask = np.zeros((H, W), dtype=np.uint8)

        self._nlp_bg  = np.zeros((H, W), dtype=np.float32)
        self._nlp_fg  = np.zeros((H, W), dtype=np.float32)

        # N-weight scratch (float32 — calc_nweights signature)
        self._leftW    = np.zeros((H, W), dtype=np.float32)
        self._upleftW  = np.zeros((H, W), dtype=np.float32)
        self._upW      = np.zeros((H, W), dtype=np.float32)
        self._uprightW = np.zeros((H, W), dtype=np.float32)

        # Flat graph capacity arrays
        self._cap_src   = np.zeros(N, dtype=np.float32)
        self._cap_snk   = np.zeros(N, dtype=np.float32)
        self._cap_right = np.zeros(N, dtype=np.float32)
        self._cap_down  = np.zeros(N, dtype=np.float32)

        # Output + morph scratch
        self._final_mask = np.zeros((H, W), dtype=np.uint8)
        self._morph_tmp1 = np.zeros((H, W), dtype=np.uint8)
        self._morph_tmp2 = np.zeros((H, W), dtype=np.uint8)
        self._blurred    = np.zeros((H, W, 3), dtype=np.uint8)
        self._composite  = np.zeros((H, W, 3), dtype=np.uint8)

        self.gamma = GAMMA
        self.blur_ks = BLUR_KSIZE

    def apply(self, frame, bg_prob, to_host=True, profiling=False):
        t0 = perf_counter()
        mask, composite_frame = self._step_kernel(frame, bg_prob, to_host, profiling)
        return mask, composite_frame, perf_counter() - t0

    def _step_kernel(self, frame, bg_prob, to_host, profiling):
        raise NotImplementedError
        