import os
import numpy as np
import cupy as cp

from gmm_mask.gmm_mask_common import GMM_Mask_Base
from utils import to_planar

from settings import FLT_EPSILON

# Resolve the .cu files against this module, not the current working directory,
# so importing gmm works from anywhere (notebooks/, tests/, ...).
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_kernels")

with open(os.path.join(KERNEL_DIR, "step_kernel_cp_v1.cu"), "r", encoding="utf-8") as f_update:
    STEP_KERNEL = f_update.read()

TILE_X = 32
TILE_Y = 8
MAX_C = 3

class GMM_Mask_CuPy(GMM_Mask_Base):
    def __init__(self, height: int, width: int, *args, **kwargs):
        super().__init__(height, width, *args, **kwargs)

        self.bg_prob = np.zeros((self.H, self.W), dtype=np.float32)
        
        self.d_means = cp.asarray(self.means)
        self.d_vars = cp.asarray(self.vars)
        self.d_weights = cp.asarray(self.weights)
        self.d_modes = cp.asarray(self.modes)
        self.d_mask = cp.asarray(self.mask)
        self.d_bg_prob = cp.asarray(self.bg_prob)



        self.block = (TILE_X, TILE_Y)
        self.grid = ((self.W + TILE_X - 1) // TILE_X,
                    (self.H + TILE_Y - 1) // TILE_Y)

        self.kernel = cp.RawKernel(STEP_KERNEL, "step_gmm", options=("-lineinfo",))


    def step_device(self, d_frame, args, stream=0):
        self.kernel(self.grid, self.block, (
            d_frame,
            self.d_weights,
            self.d_means,
            self.d_vars,
            self.d_modes,
            self.d_mask,
            self.d_bg_prob,
            FLT_EPSILON,
            self.H, self.W, self.C, self.K,
            *args
        ))
        return self.d_mask, self.d_bg_prob

    def _step_kernel(self, frame, to_host, args):
        d_frame = cp.asarray(frame)
        self.step_device(d_frame, args)
        cp.cuda.Device().synchronize()

        if to_host:
            self.mask = self.d_mask.get()
            self.bg_prob = self.d_bg_prob.get()
            return self.mask, self.bg_prob
        else:
            return self.d_mask, self.d_bg_prob

    def sync_state(self):
        self.weights = self.d_weights.get()
        self.means = self.d_means.get()
        self.vars = self.d_vars.get()
        self.modes = self.d_modes.get()

    def background_image(self):
        self.sync_state()
        return super().background_image()