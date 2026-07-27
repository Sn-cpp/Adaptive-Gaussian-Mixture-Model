import numpy as np

from settings import INIT_VAR, REINIT_WEIGHT
from utils.timer import cpu_timer

class GMM_CPU:
    def __init__(self, first_frame: np.ndarray, n_components: int, *arg, **kwargs):        
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

        # Components satisfying matching threshold
        valid_diff = diff_square_sum < (match_threshold ** 2) * self.vars
        matched_pixels = np.any(valid_diff, axis=0)                       

        # Best valid component (minimum error)
        large_value = np.finfo(diff_square_sum.dtype).max
        masked_error = diff_square_sum + (~valid_diff) * large_value

        min_err_comps = masked_error.argmin(axis=0)                       

        matches = matched_pixels[None, :, :] & (
            np.arange(self.n_comps)[:, None, None] == min_err_comps[None, :, :]
        )                                                                  

        # Decay weights
        self.weights *= (1.0 - update_alpha)

        # Update weights
        self.weights[matches] += update_alpha

        # Update means
        k, rows, cols = np.where(matches)
        self.means[k, :, rows, cols] = (1-update_alpha)*self.means[k, :, rows, cols]\
                            + update_alpha*frame[:, rows, cols].T
        
        # Update variances
        self.vars[matches] = (1 - update_alpha) * self.vars[matches]\
                            +update_alpha * diff_square_sum[matches]
        

        # Replace weakest component where no match exists
        wk_rows, wk_cols = np.where(~matched_pixels)

        if len(wk_rows) > 0:
            weakest = self.weights.argmin(axis=0)         
            weakest_comp = weakest[wk_rows, wk_cols]

            self.means[weakest_comp, :, wk_rows, wk_cols] = frame[:, wk_rows, wk_cols].T
            self.vars[weakest_comp, wk_rows, wk_cols] = INIT_VAR
            self.weights[weakest_comp, wk_rows, wk_cols] = REINIT_WEIGHT

        # Normalize weights
        self.weights /= self.weights.sum(axis=0, keepdims=True)

    def predict(self, frame: np.ndarray, match_threshold: np.float32, background_threshold: np.float32):
        
        # Squared L2 (without variance)
        diff = frame[None, :, :, :] - self.means         
        diff_square_sum = np.sum(diff * diff, axis=1)  

        # Rank components by w / sigma
        rank = self.weights / (np.sqrt(self.vars) + 1e-6)    
        order = np.argsort(rank, axis=0)[::-1, :, :]    

        # Sort weights
        sorted_weights = np.take_along_axis(self.weights, order, axis=0)

        # Cumulative weight
        cumulative_weights = np.cumsum(sorted_weights, axis=0)

        # Select background components
        background_components = cumulative_weights <= background_threshold

        # Ensure the first component is always included
        background_components[0] = True

        # Match test
        matches = diff_square_sum < (match_threshold ** 2) * self.vars

        # Reorder matches to match ranking
        sorted_matches = np.take_along_axis(matches, order, axis=0)

        # Background if matching any selected background component
        background_mask = np.any(sorted_matches & background_components, axis=0)

        foreground_mask = (~background_mask).astype(np.uint8) * 255

        return foreground_mask, diff_square_sum

    def step_profiler(self, frame: np.ndarray, match_threshold: np.float32, background_threshold: np.float32, update_alpha: np.float32):

        mask, diff_square_sum, predict_profile = self.predict_profiler(frame, match_threshold, background_threshold)

        update_profile = self.update_profiler(frame, diff_square_sum, match_threshold, update_alpha)

        return mask, update_profile, predict_profile
    
    def update_profiler(self, frame: np.ndarray, diff_square_sum: np.ndarray, match_threshold: np.float32, update_alpha: np.float32):
        
        time_dict = dict()
   
        valid_diff, time_dict['valid_diff'] = cpu_timer(lambda: diff_square_sum < (match_threshold**2)*self.vars) # Use threshold to filter components with large error
        matched_pixels, time_dict['matched_pixels'] = cpu_timer(lambda: np.any(valid_diff, axis=0))


        # Find the best matched and valid component for each pixel (with lowest error)
        large_value, time_dict['large_value'] = cpu_timer(lambda: np.finfo(diff_square_sum.dtype).max) # Upper-bound of datatype

        masked_error, time_dict['masked_error'] = cpu_timer(lambda: diff_square_sum + (~valid_diff) * large_value) # Push invalid components error to max
        min_err_comps, time_dict['min_err_comps'] = cpu_timer(lambda: masked_error.argmin(axis=0)) # Use argmin to get the minimum error component for each pixel
        matches, time_dict['matches'] = cpu_timer(lambda: matched_pixels[None, :, :] & (np.arange(self.n_comps)[:, None, None] == min_err_comps[None, :, :]))
                    

        # Update weights of components
        self.weights, time_dict['weight_decay'] = cpu_timer(lambda: self.weights * (1.0 - update_alpha)) # Decay the weights
        self.weights[matches], time_dict['weight_update'] = cpu_timer(lambda: self.weights[matches] + update_alpha) # Update weights of matched components only

        # Update means and variances of matched components
        (k, rows, cols), time_dict['coordinate_retrieval'] = cpu_timer(lambda: np.where(matches))
        self.means[k, :, rows, cols], time_dict['mean_update'] = cpu_timer(lambda: (1-update_alpha)*self.means[k, :, rows, cols]\
                                + update_alpha*frame[:, rows, cols].T)

        self.vars[matches], time_dict['var_update'] = cpu_timer(lambda: (1-update_alpha)*self.vars[matches]\
                                + update_alpha*diff_square_sum[matches])
        

        # Replace weakest component of each pixel
        (wk_rows, wk_cols), time_dict['weakest_comps_idx'] = cpu_timer(lambda: np.where(~matched_pixels))

        time_dict['weakest'] = 0.0
        time_dict['weakest_comp'] = 0.0
        time_dict['mean_replace'] = 0.0
        time_dict['var_replace'] = 0.0
        time_dict['weight_re_init'] = 0.0
     
        if len(wk_rows) > 0:
            weakest, time_dict['weakest'] = cpu_timer(lambda: self.weights.argmin(axis=0))
            weakest_comp, time_dict['weakest_comp'] = cpu_timer(lambda: weakest[wk_rows, wk_cols])

            # Replace mean
            self.means[weakest_comp, :, wk_rows, wk_cols], time_dict['mean_replace'] = cpu_timer(lambda: frame[:, wk_rows, wk_cols].T)

            # Re-init variance
            self.vars[weakest_comp, wk_rows, wk_cols], time_dict['var_replace'] = cpu_timer(lambda: 400.0)

            # Re-init weight
            self.weights[weakest_comp, wk_rows, wk_cols], time_dict['weight_re_init'] = cpu_timer(lambda: 0.01)

        # Normalize weights
        self.weights, time_dict['weight_normalize'] = cpu_timer(lambda: self.weights / self.weights.sum(axis=0, keepdims=True))

        return time_dict

    def predict_profiler(self, frame: np.ndarray, match_threshold: np.float32, background_threshold: np.float32):
        time_dict = dict()

        # Compute error between pixel and each component's mean on 3 channels (R, G, B)
        diff, time_dict['diff'] =  cpu_timer(lambda: frame[None, :, :, :] - self.means)
        diff_square_sum, time_dict['diff_square_sum'] = cpu_timer(lambda: np.sum(diff * diff, axis=1)) # Square each channel error and sum all together


        # Sort components by w / sigma
        rank, time_dict['rank'] = cpu_timer(lambda: self.weights / (np.sqrt(self.vars) + 1e-6))
        order, time_dict['order'] = cpu_timer(lambda: np.argsort(rank, axis=0)[::-1, :, :])

        sorted_weights, time_dict['sort_weight'] = cpu_timer(lambda: np.take_along_axis(self.weights, order, axis=0))

        cumulative_weights, time_dict['weight_accumulate'] = cpu_timer(lambda: np.cumsum(sorted_weights, axis=0))

        # Background components
        background_components, time_dict['background_comps'] = cpu_timer(lambda: cumulative_weights <= background_threshold)
    
        # Ensure first component is always included
        background_components[0], time_dict['first_comp'] = cpu_timer(lambda: True)

        # Reorder match mask using same ordering
        matches, time_dict['matches'] = cpu_timer(lambda: diff_square_sum < (match_threshold**2) * self.vars)

        sorted_matches, time_dict['sort_matches'] = cpu_timer(lambda: np.take_along_axis(matches, order, axis=0))

        # Pixel is background if it matches any selected background component
        background_mask, time_dict['background_mask'] = cpu_timer(lambda: np.any(sorted_matches & background_components, axis=0))

        # Foreground mask
        foreground_mask, time_dict['foreground_mask'] = cpu_timer(lambda: (~background_mask).astype(np.uint8) * 255)

        return foreground_mask, diff_square_sum, time_dict