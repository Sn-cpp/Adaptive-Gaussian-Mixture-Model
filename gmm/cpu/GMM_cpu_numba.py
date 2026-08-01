import numpy as np
from numba import njit, prange
from utils import cpu_timer

from settings import INIT_VAR, REINIT_WEIGHT


@njit(cache=True, parallel=True)
def update_numba_parallel(frame: np.ndarray, mask: np.ndarray, diff_square_sum: np.ndarray,
                match_threshold: float, update_alpha: float, comp_gen_threshold: float, 
                k_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):

    sqr_threshold = match_threshold ** 2
    
    for i in prange(frame.shape[1]):
        for j in range(frame.shape[2]):
            p0 = frame[0, i, j]
            p1 = frame[1, i, j]
            p2 = frame[2, i, j]
            
            min_err_comp = -1
            min_err_val = np.inf
            min_dist = np.inf

            # Find the best matching component
            for k in range(k_comps):
                diff_sqr = diff_square_sum[k, i, j] / (vars[k, i, j] + 1e-12)
                
                if diff_sqr < sqr_threshold:
                    if diff_sqr < min_err_val:
                        min_err_val = diff_sqr
                        min_err_comp = k

                if diff_sqr < min_dist:
                    min_dist = diff_sqr

            # Determine alpha based on current classification (not previous mask)
            # Use the mask from the prediction step
            is_foreground = (mask[i, j] > 0)
            if is_foreground:
                alpha = 0.01  # Extremely slow for foreground persistence
            else:
                alpha = update_alpha
            
            one_minus_alpha = 1.0 - alpha

            for k in range(k_comps):
                weights[k, i, j] *= one_minus_alpha

            if min_err_comp != -1:
                # Use rho = alpha (simplified) or clamp the exponential
                rho = max(alpha * 0.3, alpha * np.exp(-0.5 * min_err_val))
                
                # Weight update for matched component
                weights[min_err_comp, i, j] += alpha

                # Update mean
                means[min_err_comp, 0, i, j] += rho * (p0 - means[min_err_comp, 0, i, j])
                means[min_err_comp, 1, i, j] += rho * (p1 - means[min_err_comp, 1, i, j])
                means[min_err_comp, 2, i, j] += rho * (p2 - means[min_err_comp, 2, i, j])

                # Correct variance update
                vars[min_err_comp, i, j] = (1 - rho) * vars[min_err_comp, i, j] + rho * min_err_val

            else:
                # No match found
                if min_dist > comp_gen_threshold:
                    weakest_comp = 0
                    min_weight = weights[0, i, j]
                    for k in range(1, k_comps):
                        if weights[k, i, j] < min_weight:
                            min_weight = weights[k, i, j]
                            weakest_comp = k
                
                    means[weakest_comp, 0, i, j] = p0
                    means[weakest_comp, 1, i, j] = p1
                    means[weakest_comp, 2, i, j] = p2
                    vars[weakest_comp, i, j] = INIT_VAR
                    weights[weakest_comp, i, j] = REINIT_WEIGHT

            # Normalize weights
            sum_w = 0.0
            for k in range(k_comps):
                sum_w += weights[k, i, j]
                
            if sum_w > 0:
                for k in range(k_comps):
                    weights[k, i, j] /= sum_w

@njit(cache=True, parallel=True)
def predict_numba_parallel(frame: np.ndarray, match_threshold: float, weight_threshold: float,
                  k_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):
    
    diff_square_sum = np.zeros(shape=(k_comps, frame.shape[1], frame.shape[2]))
    foreground_mask = np.zeros(shape=(frame.shape[1], frame.shape[2]), dtype=np.uint8)
    sqr_match = match_threshold ** 2
    
    for i in prange(frame.shape[1]):
        for j in range(frame.shape[2]):
            p0 = frame[0, i, j]
            p1 = frame[1, i, j]
            p2 = frame[2, i, j]

            ranks = np.zeros(k_comps, dtype=np.float32)
            order = np.arange(k_comps)
            matches = np.zeros(k_comps, dtype=np.bool_)
            
            # Pre-compute diff_square_sum for update step
            for k in range(k_comps):
                diff_0 = p0 - means[k, 0, i, j]
                diff_1 = p1 - means[k, 1, i, j]
                diff_2 = p2 - means[k, 2, i, j]
                
                err = diff_0**2 + diff_1**2 + diff_2**2
                diff_square_sum[k, i, j] = err
                
                # Store match status
                matches[k] = err < (sqr_match * vars[k, i, j])
                ranks[k] = weights[k, i, j] / (np.sqrt(vars[k, i, j]) + 1e-6)

            # Sort ranks (bubble sort - consider using argsort for faster)
            for m in range(k_comps - 1):
                for n in range(0, k_comps - m - 1):
                    if ranks[n] < ranks[n + 1]:
                        ranks[n], ranks[n + 1] = ranks[n + 1], ranks[n]
                        order[n], order[n + 1] = order[n + 1], order[n]

            # Correct background classification
            is_background = False
            cumulative_weight = 0.0

            for k in range(k_comps):
                idx = order[k]
                cumulative_weight += weights[idx, i, j]
                
                if cumulative_weight > weight_threshold:
                    break
                
                if matches[idx]:
                    is_background = True
                    break
                
            foreground_mask[i, j] = 0 if is_background else 255

    return foreground_mask, diff_square_sum

class GMM_CPU_NUMBA:
    def __init__(self, first_frame: np.ndarray, n_components: int, parallel=False, *arg, **kwargs):

        if parallel:
            self.update_func = update_numba_parallel
            self.predict_func = predict_numba_parallel
        # else:
        #     self.update_func = update_numba
        #     self.predict_func = predict_numba
        
        self.height, self.width, _ = first_frame.shape
        
        self.k_comps = n_components

        # First component mean with the first frame
        self.means = np.ones(shape=(self.k_comps, 3, self.height, self.width), dtype=np.float32)
        self.means[0, :, :, :] = first_frame.transpose(2, 0, 1)
    
        # All variances to a fixed value
        self.vars = np.full(shape=(self.k_comps, self.height, self.width), fill_value=INIT_VAR, dtype=np.float32)
        
        # Weight of the first component of each pixel is 1.0, the others are 0.0
        self.weights = np.zeros(shape=(self.k_comps, self.height, self.width), dtype=np.float32)
        self.weights[0, :, :] = 1.0

    def update(self, frame: np.ndarray, mask: np.ndarray, diff_square_sum: np.ndarray, match_threshold: np.float32, update_alpha: np.float32, comp_gen_threshold: np.float32):
        self.update_func(frame, mask, diff_square_sum, match_threshold, update_alpha, comp_gen_threshold, self.k_comps, self.means, self.vars, self.weights)

    def predict(self, frame: np.ndarray, match_threshold: np.float32, weight_threshold: np.float32):
        mask, diff_square_sum = self.predict_func(frame, match_threshold, weight_threshold, self.k_comps, self.means, self.vars, self.weights)
        
        return mask, diff_square_sum

    def step(self, frame: np.ndarray, match_threshold: np.float32, update_alpha: np.float32, weight_threshold: np.float32, comp_gen_threshold: np.float32):
        # Predict step 
        (mask, diff_square_sum), predict_cost = cpu_timer(self.predict, frame=frame, match_threshold=match_threshold, weight_threshold=weight_threshold)

        # Update step
        _, update_cost = cpu_timer(self.update, frame=frame, mask=mask, diff_square_sum=diff_square_sum, match_threshold=match_threshold, update_alpha=update_alpha, comp_gen_threshold=comp_gen_threshold)

        # Refine mask
        # TODO

        return mask, predict_cost + update_cost