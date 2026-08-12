import numpy as np
from settings import MOG2_N_COMPONENTS

class GMM_EM_Base:
    def __init__(self, height: int, width: int):
        self.H = height
        self.W = width
        self.K = MOG2_N_COMPONENTS

        self.model    = np.zeros((self.K, 13), dtype=np.float32)
        self.inv_covs = np.zeros((self.K, 3, 3), dtype=np.float32)
        self.cov_dets = np.zeros(self.K, dtype=np.float32)

        # Sufficient-statistics buffers (no alloc in hot path)
        self._sums   = np.zeros((self.K, 3),    dtype=np.float32)
        self._prods  = np.zeros((self.K, 3, 3), dtype=np.float32)
        self._counts = np.zeros(self.K,         dtype=np.int64)
        self._total  = np.zeros(1,         dtype=np.int64)
        self._comp   = np.zeros((self.H, self.W), dtype=np.int32)

        # Seed: uniform coefs so the first frame has a valid model
        for ci in range(self.K):
            self.model[ci, 0] = 1.0 / self.K

    
    def fit(self, frame: np.ndarray, label_map: np.ndarray, is_fg: bool):
        """Refit GMM from pixels of this class in the current frame.

        frame         : (H, W, 3) float32
        label_map     : (H, W) uint8  — GC label map (0/1/2/3)
        is_fg         : bool
        """
        raise NotImplementedError


    def neg_log_prob(self, frame_bgr_f32, out):
        """-log P(color | GMM) for every pixel → written into `out` (H,W) float32."""
        raise NotImplementedError
    