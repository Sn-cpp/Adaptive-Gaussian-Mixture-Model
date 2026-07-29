import numpy as np
import cupy as cp

from settings import INIT_VAR, REINIT_WEIGHT

class GMM_CUPY_V0:
    def __init__(self, first_frame: np.ndarray, n_components: int, *arg, **kwargs):        
        self.height, self.width, _ = first_frame.shape
        
        self.k_comps = n_components

        # First component mean with the first frame
        self.means = np.ones(shape=(self.k_comps, 3, self.height, self.width), dtype=np.float32)
        self.means[0, :, :, :] = first_frame.transpose(2, 0, 1)
        self.means = cp.asarray(self.means)
    
        # All variances to a fixed value
        self.vars = cp.full(shape=(self.k_comps, self.height, self.width), fill_value=INIT_VAR, dtype=cp.float32)
        
        # Weight of the first component of each pixel is 1.0, the others are 0.0
        self.weights = cp.zeros(shape=(self.k_comps, self.height, self.width), dtype=cp.float32)
        self.weights[0, :, :] = 1.0

    def update(self, frame: cp.ndarray, diff_square_sum: cp.ndarray, match_threshold: np.float32, update_alpha: np.float32):

        # Components satisfying matching threshold
        valid_diff = diff_square_sum < (match_threshold ** 2) * self.vars
        matched_pixels = cp.any(valid_diff, axis=0)                       

        # Best valid component (minimum error)
        large_value = cp.finfo(diff_square_sum.dtype).max
        masked_error = diff_square_sum + (~valid_diff) * large_value

        min_err_comps = masked_error.argmin(axis=0)                       

        matches = matched_pixels[None, :, :] & (
            cp.arange(self.k_comps)[:, None, None] == min_err_comps[None, :, :]
        )                                                                  

        # Update weights
        self.weights *= (1.0 - update_alpha)
        self.weights[matches] += update_alpha

        # Update means
        k, rows, cols = cp.where(matches)
        self.means[k, :, rows, cols] = (1-update_alpha)*self.means[k, :, rows, cols]\
                            + update_alpha*frame[:, rows, cols].T
        
        # Update variances
        self.vars[matches] = (1 - update_alpha) * self.vars[matches]\
                            +update_alpha * diff_square_sum[matches]
        

        # Replace weakest component where no match exists
        wk_rows, wk_cols = cp.where(~matched_pixels)

        if len(wk_rows) > 0:
            weakest = self.weights.argmin(axis=0)         
            weakest_comp = weakest[wk_rows, wk_cols]

            self.means[weakest_comp, :, wk_rows, wk_cols] = frame[:, wk_rows, wk_cols].T
            self.vars[weakest_comp, wk_rows, wk_cols] = INIT_VAR
            self.weights[weakest_comp, wk_rows, wk_cols] = REINIT_WEIGHT

        # Normalize weights
        self.weights /= self.weights.sum(axis=0, keepdims=True)

    def predict(self, frame: cp.ndarray, match_threshold: np.float32, weight_threshold: np.float32):
        
        # Squared L2 (without variance)
        diff = frame[None, :, :, :] - self.means         
        diff_square_sum = cp.sum(diff * diff, axis=1)  

        # Rank components by w / sigma
        rank = self.weights / (cp.sqrt(self.vars) + 1e-6)    
        order = cp.argsort(rank, axis=0)[::-1, :, :]    

        # Sort weights
        sorted_weights = cp.take_along_axis(self.weights, order, axis=0)

        # Cumulative weight
        cumulative_weights = cp.cumsum(sorted_weights, axis=0)

        # Select background components
        background_components = cumulative_weights <= weight_threshold

        # Ensure the first component is always included
        background_components[0] = True

        # Match test
        matches = diff_square_sum < (match_threshold ** 2) * self.vars

        # Reorder matches to match ranking
        sorted_matches = cp.take_along_axis(matches, order, axis=0)

        # Background if matching any selected background component
        background_mask = cp.any(sorted_matches & background_components, axis=0)

        foreground_mask = (~background_mask).astype(cp.uint8) * 255

        return foreground_mask, diff_square_sum
