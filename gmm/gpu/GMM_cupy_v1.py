import numpy as np
import cupy as cp

from settings import INIT_VAR, REINIT_WEIGHT

with open("gmm/gpu/kernels/update_kernel_cp_v1.cu", "r", encoding="utf-8") as f_update: 
    UPDATE_KERNEL = f_update.read()

with open("gmm/gpu/kernels/predict_kernel_cp_v1.cu", "r", encoding="utf-8") as f_predict: 
    PREDICT_KERNEL = f_predict.read()


class GMM_CUPY_V1:
    def __init__(self, first_frame: np.ndarray, n_components: int, block_size: int=256, *arg, **kwargs):        
        self.height, self.width, _ = first_frame.shape
        
        self.k_comps = np.int32(n_components)

        # First component mean with the first frame
        self.means = np.ones(shape=(self.k_comps, 3, self.height, self.width), dtype=np.float32)
        self.means[0, :, :, :] = first_frame.transpose(2, 0, 1)
        self.means = cp.asarray(self.means)
    
        # All variances to a fixed value
        self.vars = cp.full(shape=(self.k_comps, self.height, self.width), fill_value=INIT_VAR, dtype=cp.float32)
        
        # Weight of the first component of each pixel is 1.0, the others are 0.0
        self.weights = cp.zeros(shape=(self.k_comps, self.height, self.width), dtype=cp.float32)
        self.weights[0, :, :] = 1.0

        self.update_kernel = cp.RawKernel(UPDATE_KERNEL, "update_gmm", options=("-lineinfo",))
        self.predict_kernel = cp.RawKernel(PREDICT_KERNEL, "predict_gmm", options=("-lineinfo",))

        # GPU constants
        self.num_pixels = np.int32(first_frame.shape[0] * first_frame.shape[1])
        self.C = np.int32(first_frame.shape[2])
        self.block_size = block_size
        self.grid_size = (self.num_pixels + block_size - 1) // block_size

    def update(self, frame: cp.ndarray, diff_square_sum: cp.ndarray, match_threshold: np.float32, update_alpha: np.float32):
        self.update_kernel((self.grid_size,), (self.block_size,), (
            frame, diff_square_sum, self.means, self.vars, self.weights,
            match_threshold, update_alpha,
            INIT_VAR, REINIT_WEIGHT, self.num_pixels, self.C, self.k_comps
        ))

    def predict(self, frame: cp.ndarray, match_threshold: np.float32, weight_threshold: np.float32):
        mask = cp.zeros(shape=(frame.shape[1], frame.shape[2]), dtype=cp.uint8)
        diff_square_sum = cp.zeros_like(self.weights, dtype=cp.float32)

        self.predict_kernel((self.grid_size,), (self.block_size,), (
            frame, diff_square_sum, mask,
            self.means, self.vars, self.weights,
            match_threshold, weight_threshold,
            self.num_pixels, self.C, self.k_comps
        ))

        return mask, diff_square_sum
