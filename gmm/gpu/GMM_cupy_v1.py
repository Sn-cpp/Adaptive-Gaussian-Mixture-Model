import os

import numpy as np
import cupy as cp

from settings import FLT_EPSILON
from gmm.mog2_common import MOG2Base

# Resolve the .cu files against this module, not the current working directory,
# so importing gmm works from anywhere (notebooks/, tests/, ...).
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

with open(os.path.join(KERNEL_DIR, "step_kernel_cp_v1.cu"), "r", encoding="utf-8") as f_update:
    STEP_KERNEL = f_update.read()

TILE_X = 32
TILE_Y = 8
MAX_C = 3

class GMM_CUPY_V1(MOG2Base):
    def __init__(self, first_frame: np.ndarray, n_components: int, *arg, **kwargs):
        super().__init__(first_frame, n_components, *arg, **kwargs)

        self.d_means = cp.asarray(self.means)
        self.d_vars = cp.asarray(self.vars)
        self.d_weights = cp.asarray(self.weights)
        self.d_modes = cp.asarray(self.modes)
        self.d_mask = cp.asarray(self.mask)

        self.num_pixels = self.height * self.width

        self.block = (TILE_X, TILE_Y)
        self.grid = ((self.width + TILE_X - 1) // TILE_X,
                     (self.height + TILE_Y - 1) // TILE_Y)

        self.kernel = cp.RawKernel(STEP_KERNEL, "step_gmm", options=("-lineinfo",))        

    def step_device(self, d_frame, args, stream=0):
        self.kernel(self.grid, self.block, (
            d_frame,
            self.d_weights,
            self.d_means,
            self.d_vars,
            self.d_modes,
            self.d_mask,
            FLT_EPSILON,
            self.num_pixels, self.n_channels, self.n_comps,
            *args
        ))
        return self.d_mask

    def _step_kernel(self, frame, args):
        d_frame = cp.asarray(np.ascontiguousarray(frame))

        print(d_frame.shape)
        print(d_frame.flags['C_CONTIGUOUS'])
        print(d_frame.strides)

        raise Exception("Test")
        self.step_device(d_frame, args)
        cp.cuda.Device().synchronize()
        self.mask = self.d_mask.get()

    def sync_state(self):
        self.weights = self.d_weights.get()
        self.means = self.d_means.get()
        self.vars = self.d_vars.get()
        self.modes = self.d_modes.get()

    def background_image(self):
        self.sync_state()
        return super().background_image()