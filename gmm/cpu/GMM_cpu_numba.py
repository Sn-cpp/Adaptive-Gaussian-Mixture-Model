import numpy as np
from numba import njit, prange

from settings import INIT_VAR, REINIT_WEIGHT

@njit(cache=True)
def update_numba(frame: np.ndarray, diff_square_sum: np.ndarray, match_threshold: float, update_alpha: float, 
                 n_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):

    # Pre-compute constants    
    sqr_threshold = match_threshold ** 2
    one_minus_alpha = 1.0 - update_alpha

    # Decay the weights
    weights *= one_minus_alpha

    for i in range(frame.shape[1]):
        for j in range(frame.shape[2]):
            p0 = frame[0, i, j]
            p1 = frame[1, i, j]
            p2 = frame[2, i, j]

            min_err_comp = -1
            min_err_val = np.inf

            # Loop through components to find the best matching component
            for k in range(n_comps):        
                diff_sqr = diff_square_sum[k, i, j]
                
                # Check if it matches the current component's variance threshold
                if diff_sqr < (sqr_threshold * vars[k, i, j]):
                    if diff_sqr < min_err_val:
                        min_err_val = diff_sqr
                        min_err_comp = k

            # Perform statistics update based on match result
            if min_err_comp != -1:
                # Increase weight
                weights[min_err_comp, i, j] += update_alpha

                # Update mean for all 3 channels
                means[min_err_comp, 0, i, j] = one_minus_alpha * means[min_err_comp, 0, i, j] + update_alpha * p0
                means[min_err_comp, 1, i, j] = one_minus_alpha * means[min_err_comp, 1, i, j] + update_alpha * p1
                means[min_err_comp, 2, i, j] = one_minus_alpha * means[min_err_comp, 2, i, j] + update_alpha * p2

                # Update variance
                vars[min_err_comp, i, j] = one_minus_alpha * vars[min_err_comp, i, j] + update_alpha * min_err_val

            else:
                # No match found: find weakest component manually
                weakest_comp = 0
                min_weight = weights[0, i, j]
                for k in range(1, n_comps):
                    if weights[k, i, j] < min_weight:
                        min_weight = weights[k, i, j]
                        weakest_comp = k
            
                # Replace mean channels
                means[weakest_comp, 0, i, j] = p0
                means[weakest_comp, 1, i, j] = p1
                means[weakest_comp, 2, i, j] = p2

                # Re-init variance and weight
                vars[weakest_comp, i, j] = INIT_VAR
                weights[weakest_comp, i, j] = REINIT_WEIGHT

            # Normalize weights manually for this pixel
            sum_w = 0.0
            for k in range(n_comps):
                sum_w += weights[k, i, j]
                
            if sum_w > 0:
                for k in range(n_comps):
                    weights[k, i, j] /= sum_w

@njit(cache=True)
def predict_numba(frame: np.ndarray, match_threshold: float, background_threshold: float,
                  n_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):
    
    diff_square_sum = np.zeros(shape=(n_comps, frame.shape[1], frame.shape[2]))

    foreground_mask = np.zeros(shape=(frame.shape[1], frame.shape[2]), dtype=np.uint8)

    sqr_match = match_threshold ** 2
    
    # Iterate over rows
    for i in range(frame.shape[1]):
        # Iterate over columns
        for j in range(frame.shape[2]):
            p0 = frame[0, i, j]
            p1 = frame[1, i, j]
            p2 = frame[2, i, j]

            ranks = np.zeros(n_comps, dtype=np.float32)
            order = np.arange(n_comps)
            matches = np.zeros(n_comps, dtype=np.bool)

            for k in range(n_comps):
                # Manually compute Euclidean distance between pixel and component k
                diff_0 = p0 - means[k, 0, i, j]
                diff_1 = p1 - means[k, 1, i, j]
                diff_2 = p2 - means[k, 2, i, j]

                err = diff_0**2 + diff_1**2 + diff_2**2
                diff_square_sum[k, i, j] = err
                
                # Check if this component is matched
                matches[k] = err < (sqr_match * vars[k, i, j])
                
                # Compute rank = w / sigma
                ranks[k] = weights[k, i, j] / (np.sqrt(vars[k, i, j]) + 1e-6)

            # Utilize a sort algorithm
            for m in range(n_comps - 1):
                for n in range(0, n_comps - m - 1):
                    if ranks[n] < ranks[n + 1]:
                        # Swap rank
                        temp_rank = ranks[n]
                        ranks[n] = ranks[n + 1]
                        ranks[n + 1] = temp_rank
                        # Swap order
                        temp_idx = order[n]
                        order[n] = order[n + 1]
                        order[n + 1] = temp_idx

            # Determine background/foreground
            is_background = False
            cumulative_weight = 0.0

            for k in range(n_comps):
                idx = order[k]
                cumulative_weight += weights[idx, i, j]
                
                # Use cumulative relative weight and threshold to determine the background
                included = (k == 0 or (cumulative_weight <= background_threshold))
                if not included:
                    break # Exceeding threshold

                if matches[idx]: # Must be a matched component
                    is_background = True
                    break # Force-stop to save time
                    

            foreground_mask[i, j] = 0 if is_background else 255

    return foreground_mask, diff_square_sum

@njit(cache=True, parallel=True)
def update_numba_parallel(frame: np.ndarray, diff_square_sum: np.ndarray, match_threshold: float, update_alpha: float, 
                 n_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):

    # Pre-compute constants    
    sqr_threshold = match_threshold ** 2
    one_minus_alpha = 1.0 - update_alpha

    # Decay the weights
    weights *= one_minus_alpha

    # Use prange for parallel loop optimization
    for i in prange(frame.shape[1]):
        for j in range(frame.shape[2]):
            p0 = frame[0, i, j]
            p1 = frame[1, i, j]
            p2 = frame[2, i, j]

            min_err_comp = -1
            min_err_val = np.inf

            # Loop through components to find the best matching component
            for k in range(n_comps):        
                diff_sqr = diff_square_sum[k, i, j]
                
                # Check if it matches the current component's variance threshold
                if diff_sqr < (sqr_threshold * vars[k, i, j]):
                    if diff_sqr < min_err_val:
                        min_err_val = diff_sqr
                        min_err_comp = k

            # Perform statistics update based on match result
            if min_err_comp != -1:
                # Increase weight
                weights[min_err_comp, i, j] += update_alpha

                # Update mean for all 3 channels
                means[min_err_comp, 0, i, j] = one_minus_alpha * means[min_err_comp, 0, i, j] + update_alpha * p0
                means[min_err_comp, 1, i, j] = one_minus_alpha * means[min_err_comp, 1, i, j] + update_alpha * p1
                means[min_err_comp, 2, i, j] = one_minus_alpha * means[min_err_comp, 2, i, j] + update_alpha * p2

                # Update variance
                vars[min_err_comp, i, j] = one_minus_alpha * vars[min_err_comp, i, j] + update_alpha * min_err_val

            else:
                # No match found: find weakest component manually
                weakest_comp = 0
                min_weight = weights[0, i, j]
                for k in range(1, n_comps):
                    if weights[k, i, j] < min_weight:
                        min_weight = weights[k, i, j]
                        weakest_comp = k
            
                # Replace mean channels
                means[weakest_comp, 0, i, j] = p0
                means[weakest_comp, 1, i, j] = p1
                means[weakest_comp, 2, i, j] = p2

                # Re-init variance and weight
                vars[weakest_comp, i, j] = INIT_VAR
                weights[weakest_comp, i, j] = REINIT_WEIGHT

            # Normalize weights manually for this pixel
            sum_w = 0.0
            for k in range(n_comps):
                sum_w += weights[k, i, j]
                
            if sum_w > 0:
                for k in range(n_comps):
                    weights[k, i, j] /= sum_w

@njit(cache=True, parallel=True)
def predict_numba_parallel(frame: np.ndarray, match_threshold: float, background_threshold: float,
                  n_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):
    
    diff_square_sum = np.zeros(shape=(n_comps, frame.shape[1], frame.shape[2]))

    foreground_mask = np.zeros(shape=(frame.shape[1], frame.shape[2]), dtype=np.uint8)

    sqr_match = match_threshold ** 2
    
    # Iterate over rows
    for i in prange(frame.shape[1]):
        # Iterate over columns
        for j in range(frame.shape[2]):
            p0 = frame[0, i, j]
            p1 = frame[1, i, j]
            p2 = frame[2, i, j]

            ranks = np.zeros(n_comps, dtype=np.float32)
            order = np.arange(n_comps)
            matches = np.zeros(n_comps, dtype=np.bool)

            for k in range(n_comps):
                # Manually compute Euclidean distance between pixel and component k
                diff_0 = p0 - means[k, 0, i, j]
                diff_1 = p1 - means[k, 1, i, j]
                diff_2 = p2 - means[k, 2, i, j]

                err = diff_0**2 + diff_1**2 + diff_2**2
                diff_square_sum[k, i, j] = err
                
                # Check if this component is matched
                matches[k] = err < (sqr_match * vars[k, i, j])
                
                # Compute rank = w / sigma
                ranks[k] = weights[k, i, j] / (np.sqrt(vars[k, i, j]) + 1e-6)

            # Utilize a sort algorithm
            for m in range(n_comps - 1):
                for n in range(0, n_comps - m - 1):
                    if ranks[n] < ranks[n + 1]:
                        # Swap rank
                        temp_rank = ranks[n]
                        ranks[n] = ranks[n + 1]
                        ranks[n + 1] = temp_rank
                        # Swap order
                        temp_idx = order[n]
                        order[n] = order[n + 1]
                        order[n + 1] = temp_idx

            # Determine background/foreground
            is_background = False
            cumulative_weight = 0.0

            for k in range(n_comps):
                idx = order[k]
                cumulative_weight += weights[idx, i, j]

                # Use cumulative relative weight and threshold to determine the background
                included = (k == 0 or (cumulative_weight <= background_threshold))
                if not included:
                    break # Exceeding threshold

                if matches[idx]: # Must be a matched component
                    is_background = True
                    break # Force-stop to save time
                
            foreground_mask[i, j] = 0 if is_background else 255

    return foreground_mask, diff_square_sum

class GMM_CPU_NUMBA:
    def __init__(self, first_frame: np.ndarray, n_components: int, parallel=False, *arg, **kwargs):

        if parallel:
            self.update_func = update_numba_parallel
            self.predict_func = predict_numba_parallel
        else:
            self.update_func = update_numba
            self.predict_func = predict_numba
        
        self.height, self.width, _ = first_frame.shape
        
        self.n_comps = n_components

        # First component mean with the first frame
        self.means = np.ones(shape=(self.n_comps, 3, self.height, self.width), dtype=np.float32)
        self.means[0, :, :, :] = first_frame.transpose(2, 0, 1)
    
        # All variances to a fixed value
        self.vars = np.full(shape=(self.n_comps, self.height, self.width), fill_value=INIT_VAR, dtype=np.float32)
        
        # Weight of the first component of each pixel is 1.0, the others are 0.0
        self.weights = np.zeros(shape=(self.n_comps, self.height, self.width), dtype=np.float32)
        self.weights[0, :, :] = 1.0

    def update(self, frame: np.ndarray, diff_square_sum: np.ndarray, match_threshold: np.float32, update_alpha: np.float32):
        self.update_func(frame, diff_square_sum, match_threshold, update_alpha, self.n_comps, self.means, self.vars, self.weights)

    def predict(self, frame: np.ndarray, match_threshold: np.float32, background_threshold: np.float32):
        mask, diff_square_sum = self.predict_func(frame, match_threshold, background_threshold, self.n_comps, self.means, self.vars, self.weights)
        
        return mask, diff_square_sum