import numpy as np
from numba import njit, prange

from settings import INIT_VAR, REINIT_WEIGHT
from utils.timer import cpu_timer

@njit(cache=True)
def update_numba(frame: np.ndarray, match_threshold: float, update_alpha: float, 
                 n_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):
    # Compute error between pixel and each component's mean on 3 channels (R, G, B)
    diff = frame[:, :, None, :] - means
    diff_square_sum = np.sum(diff**2, axis=-1) # Square each channel error and sum all together
    
    valid_diff = diff_square_sum < (match_threshold**2)*vars # Use threshold to filter components with large error
        
    # Decay the weights
    weights *= (1.0 - update_alpha)

    # Iterate over frame rows
    for i in range(valid_diff.shape[0]):
        # Iterate over frame collumns
        for j in range(valid_diff.shape[1]):

            min_err_comp = -1
            weakest_comp = None

            # Iterate over Gaussian components
            for k in range(valid_diff.shape[2]):
                if valid_diff[i, j, k]:                    
                    # Update reference to the better component (if any)
                    if min_err_comp == -1 or (diff_square_sum[i, j, k] < diff_square_sum[i, j, min_err_comp]):
                        min_err_comp = k

            # Perform statistics update for the best matched component
            if min_err_comp:
                # Increase weight
                weights[i, k, min_err_comp] += update_alpha

                # Update mean and variance
                means[i, j, min_err_comp] = (1-update_alpha)*means[i, j, min_err_comp] + update_alpha*frame[i, j]

                vars[i, j, min_err_comp] = (1-update_alpha)*vars[i, j, min_err_comp] + update_alpha*diff_square_sum[i, j, min_err_comp]

            # If there is no matched one, replace the weakest component
            else:
                weakest_comp = np.argmin(weights[i, j])
            
                # Replace mean
                means[i, j, weakest_comp] = frame[i, j]

                # Re-init variance
                vars[i, j, weakest_comp] = INIT_VAR

                # Re-init weight
                weights[i, j, weakest_comp] = REINIT_WEIGHT

            # Normalize weights
            weights[i, j] /= weights[i, j].sum() 

@njit(cache=True)
def predict_numba(frame: np.ndarray, match_threshold: float, background_threshold: float,
                  n_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):
    
    # Distance to all components
    diff = frame[:, :, None, :] - means
    diff_square_sum = np.sum(diff ** 2, axis=-1)
    matches = diff_square_sum < (match_threshold**2) * vars

    foreground_mask = np.zeros(shape=diff.shape[:-1], dtype=np.uint8)

    # Iterate over rows
    rank = weights / (np.sqrt(vars) + 1e-6) # Normalize components weights by w / sigma
    for i in range(rank.shape[0]):

        # Iterate over columns
        for j in range(rank.shape[1]):
            
            # Sort components in descending order
            order = np.argsort(rank[i, j])[::-1]

            # Apply sorted indices
            sorted_weights = np.take_along_axis(weights[i, j], order, axis=0)

            # Compute the cumulative relative weights, which shows the top-K components contribution 
            cumulative_weights = np.cumsum(sorted_weights)

            # Background components
            background_components = cumulative_weights <= background_threshold

            # Ensure first component is always included
            background_components[0] = True

            # Reorder match mask using same ordering
            sorted_matches = np.take_along_axis(matches[i, j], order, axis=0)

            # Combine masks to determine background pixel
            is_background = np.any(sorted_matches & background_components)

            foreground_mask[i, j] = 0 if is_background else 255

    return foreground_mask

@njit(cache=True, parallel=True)
def update_numba_parallel(frame: np.ndarray, match_threshold: float, update_alpha: float, 
                 n_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):

    # Pre-compute constants    
    sqr_threshold = match_threshold ** 2
    one_minus_alpha = 1.0 - update_alpha

    # Decay the weights
    weights *= (1.0 - update_alpha)

    # Use prange for parallel loop optimization
    for i in prange(frame.shape[0]):
        for j in range(frame.shape[1]):

            pixel = frame[i, j]
            min_err_comp = -1
            min_err_val = np.inf

            # Loop through components to find matches
            for k in range(n_comps):
                # Manual 3-channel Euclidean distance calculation
                diff_0 = pixel[0] - means[i, j, k, 0]
                diff_1 = pixel[1] - means[i, j, k, 1]
                diff_2 = pixel[2] - means[i, j, k, 2]
                
                diff_square_sum = diff_0**2 + diff_1**2 + diff_2**2
                
                # Check if it matches the current component's variance threshold
                if diff_square_sum < (sqr_threshold * vars[i, j, k]):
                    if diff_square_sum < min_err_val:
                        min_err_val = diff_square_sum
                        min_err_comp = k

            # Perform statistics update based on match result
            if min_err_comp != -1:  # Bug fix: check for valid index (index 0 is valid!)
                # Increase weight
                weights[i, j, min_err_comp] += update_alpha

                # Update mean for all 3 channels
                means[i, j, min_err_comp, 0] = one_minus_alpha * means[i, j, min_err_comp, 0] + update_alpha * pixel[0]
                means[i, j, min_err_comp, 1] = one_minus_alpha * means[i, j, min_err_comp, 1] + update_alpha * pixel[1]
                means[i, j, min_err_comp, 2] = one_minus_alpha * means[i, j, min_err_comp, 2] + update_alpha * pixel[2]

                # Update variance
                vars[i, j, min_err_comp] = one_minus_alpha * vars[i, j, min_err_comp] + update_alpha * min_err_val

            else:
                # No match found: find weakest component manually
                weakest_comp = 0
                min_weight = weights[i, j, 0]
                for k in range(1, n_comps):
                    if weights[i, j, k] < min_weight:
                        min_weight = weights[i, j, k]
                        weakest_comp = k
            
                # Replace mean channels
                means[i, j, weakest_comp, 0] = pixel[0]
                means[i, j, weakest_comp, 1] = pixel[1]
                means[i, j, weakest_comp, 2] = pixel[2]

                # Re-init variance and weight
                vars[i, j, weakest_comp] = INIT_VAR
                weights[i, j, weakest_comp] = REINIT_WEIGHT

            # Normalize weights manually for this pixel
            sum_w = 0.0
            for k in range(n_comps):
                sum_w += weights[i, j, k]
                
            if sum_w > 0:
                for k in range(n_comps):
                    weights[i, j, k] /= sum_w

@njit(cache=True, parallel=True)
def predict_numba_parallel(frame: np.ndarray, match_threshold: float, background_threshold: float,
                  n_comps: int, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):
    
    foreground_mask = np.zeros(shape=frame.shape[:-1], dtype=np.uint8)

    sqr_match = match_threshold ** 2
    

    # Iterate over rows
    for i in prange(frame.shape[0]):
        # Iterate over columns
        for j in range(frame.shape[1]):
            pixel = frame[i, j]

            ranks = np.zeros(n_comps, dtype=np.float64)
            order = np.arange(n_comps)
            matches = np.zeros(n_comps, dtype=np.bool_)

            for k in range(n_comps):
                # Manually compute Euclidean distance between pixel and component k
                diff_0 = pixel[0] - means[i, j, k, 0]
                diff_1 = pixel[1] - means[i, j, k, 1]
                diff_2 = pixel[2] - means[i, j, k, 2]
                diff_square_sum = diff_0**2 + diff_1**2 + diff_2**2
                
                # Check if this component is matched
                matches[k] = diff_square_sum < (sqr_match * vars[i, j, k])
                
                # Compute rank = w / sigma
                ranks[k] = weights[i, j, k] / (np.sqrt(vars[i, j, k]) + 1e-6)

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
                current_weight = weights[i, j, idx]
                
                # Use cumulative relative weight and threshold to determine the background
                if k == 0 or (cumulative_weight <= background_threshold):
                    if matches[idx]: # Must be a matched component
                        is_background = True
                        break # Force-stop to save time
                else: 
                    break # Exceeding threshold
                    
                cumulative_weight += current_weight

            if is_background:
                foreground_mask[i, j] = 0
            else:
                foreground_mask[i, j] = 255

    return foreground_mask

class GMM_CPU_NUMBA:
    def __init__(self, first_frame: np.ndarray, n_components: int=1, parallel=False):        
        if parallel:
            self.update_func = update_numba_parallel
            self.predict_func = predict_numba_parallel
        else:
            self.update_func = update_numba
            self.predict_func = predict_numba
        
        self.height, self.width, _ = first_frame.shape
        
        self.n_comps = n_components

        # First component mean with the first frame
        self.means = np.ones(shape=(self.height, self.width, self.n_comps, 3), dtype=np.float32)
        self.means[:, :, 0, :] = first_frame

        # All variances to a fixed value: 400
        self.vars = np.full(shape=(self.height, self.width, self.n_comps), fill_value=INIT_VAR, dtype=np.float32)
        
        # Weight of the first component of each pixel is 1.0, the others are 0.0
        self.weights = np.zeros(shape=(self.height, self.width, self.n_comps), dtype=np.float32)
        self.weights[:, :, 0] = 1.0

    def step(self, frame: np.ndarray, match_threshold: float=2.5, background_threshold: float=0.7, update_alpha: float=0.01):
        mask = self.predict_func(frame, match_threshold, background_threshold, self.n_comps, self.means, self.vars, self.weights)
        self.update_func(frame, mask, match_threshold, update_alpha, self.n_comps, self.means, self.vars, self.weights)
        return mask

    # def step_profiler(self, frame: np.ndarray, match_threshold: float=2.5, background_threshold: float=0.7, update_alpha: float=0.01):
    #     update_profile = self.update_profiler(frame, match_threshold, update_alpha)
        
    #     mask, predict_profile = self.predict_profiler(frame, match_threshold, background_threshold)

    #     return mask, update_profile, predict_profile
    
    # def update_profiler(self, frame: np.ndarray, match_threshold: float=2.5, update_alpha: float=0.01):
        
    #     time_dict = dict()

    #     # Compute error between pixel and each component's mean on 3 channels (R, G, B)
    #     diff, time_dict['diff'] =  cpu_timer(lambda: frame[:, :, None, :] - self.means)
    #     diff_square_sum, time_dict['diff_square_sum'] = cpu_timer(lambda: np.sum(diff**2, axis=-1)) # Square each channel error and sum all together
        
    #     valid_diff, time_dict['valid_diff'] = cpu_timer(lambda: diff_square_sum < (match_threshold**2)*self.vars) # Use threshold to filter components with large error
    #     matched_pixels, time_dict['matched_pixels'] = cpu_timer(lambda: np.any(valid_diff, axis=2))


    #     # Find the best matched and valid component for each pixel (with lowest error)
    #     large_value, time_dict['large_value'] = cpu_timer(lambda: np.finfo(diff_square_sum.dtype).max) # Upper-bound of datatype

    #     masked_error, time_dict['masked_error'] = cpu_timer(lambda: diff_square_sum + (~valid_diff) * large_value) # Push invalid components error to max
    #     min_err_comps, time_dict['min_err_comps'] = cpu_timer(lambda: masked_error.argmin(axis=2)) # Use argmin to get the minimum error component for each pixel
    #     matches, time_dict['matches'] = cpu_timer(lambda: matched_pixels[:, :, None] & (np.arange(self.n_comps) == min_err_comps[:, :, None]))
                    

    #     # Update weights of components
    #     self.weights, time_dict['weight_decay'] = cpu_timer(lambda: self.weights * (1.0 - update_alpha)) # Decay the weights
    #     self.weights[matches], time_dict['weight_update'] = cpu_timer(lambda: self.weights[matches] + update_alpha) # Update weights of matched components only

    #     # Update means and variances of matched components
    #     self.means[matches], time_dict['mean_update'] = cpu_timer(lambda: (1-update_alpha)*self.means[matches]\
    #                             + update_alpha*frame[matched_pixels])

    #     self.vars[matches], time_dict['var_update'] = cpu_timer(lambda: (1-update_alpha)*self.vars[matches]\
    #                             + update_alpha*diff_square_sum[matches])
        

    #     # Replace weakest component of each pixel
    #     (rows, cols), time_dict['weakest_comps_idx'] = cpu_timer(lambda: np.where(~matched_pixels))

    #     time_dict['weakest'] = 0.0
    #     time_dict['weakest_comp'] = 0.0
    #     time_dict['mean_replace'] = 0.0
    #     time_dict['var_replace'] = 0.0
    #     time_dict['weight_re_init'] = 0.0
     
    #     if len(rows) > 0:
    #         weakest, time_dict['weakest'] = cpu_timer(lambda: self.weights.argmin(axis=2))
    #         weakest_comp, time_dict['weakest_comp'] = cpu_timer(lambda: weakest[rows, cols])

    #         # Replace mean
    #         self.means[rows, cols, weakest_comp], time_dict['mean_replace'] = cpu_timer(lambda: frame[rows, cols])

    #         # Re-init variance
    #         self.vars[rows, cols, weakest_comp], time_dict['var_replace'] = cpu_timer(lambda: 400.0)

    #         # Re-init weight
    #         self.weights[rows, cols, weakest_comp], time_dict['weight_re_init'] = cpu_timer(lambda: 0.01)

    #     # Normalize weights
    #     self.weights, time_dict['weight_normalize'] = cpu_timer(lambda: self.weights / self.weights.sum(axis=2, keepdims=True))

    #     return time_dict


    # def predict_profiler(self, frame: np.ndarray, match_threshold: float = 2.5, background_threshold: float = 0.7):
        # time_dict = dict()


        # # Distance to all components
        # diff, time_dict['diff'] = cpu_timer(lambda: frame[:, :, None, :] - self.means)
        # diff_square_sum, time_dict['diff_square_sum'] = cpu_timer(lambda: np.sum(diff ** 2, axis=-1))

        # # Sort components by w / sigma
        # rank, time_dict['rank'] = cpu_timer(lambda: self.weights / (np.sqrt(self.vars) + 1e-6))
        # order, time_dict['order'] = cpu_timer(lambda: np.argsort(rank, axis=2)[:, :, ::-1])

        # sorted_weights, time_dict['sort_weight'] = cpu_timer(lambda: np.take_along_axis(self.weights, order, axis=2))

        # cumulative_weights, time_dict['weight_accumulate'] = cpu_timer(lambda: np.cumsum(sorted_weights, axis=2))

        # # Background components
        # background_components, time_dict['background_comps'] = cpu_timer(lambda: cumulative_weights <= background_threshold)
    
        # # Ensure first component is always included
        # background_components[:, :, 0], time_dict['first_comp'] = cpu_timer(lambda: True)

        # # Reorder match mask using same ordering
        # matches, time_dict['matches'] = cpu_timer(lambda: diff_square_sum < (match_threshold**2) * self.vars)

        # sorted_matches, time_dict['sort_matches'] = cpu_timer(lambda: np.take_along_axis(matches, order, axis=2))

        # # Pixel is background if it matches any selected background component
        # background_mask, time_dict['background_mask'] = cpu_timer(lambda: np.any(sorted_matches & background_components, axis=2))

        # # Foreground mask
        # foreground_mask, time_dict['foreground_mask'] = cpu_timer(lambda: (~background_mask).astype(np.uint8) * 255)

        # return foreground_mask, time_dict