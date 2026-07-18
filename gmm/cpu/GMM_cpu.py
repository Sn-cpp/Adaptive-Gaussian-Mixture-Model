import numpy as np

from settings import INIT_VAR, REINIT_WEIGHT
from utils.timer import cpu_timer

class GMM_CPU:
    def __init__(self, first_frame: np.ndarray, n_components: int=1):        
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


    def update(self, frame: np.ndarray, match_threshold: float=2.5, update_alpha: float=0.01):
        
        # Compute error between pixel and each component's mean on 3 channels (R, G, B)
        diff = frame[:, :, None, :] - self.means
        diff_square_sum = np.sum(diff**2, axis=-1) # Square each channel error and sum all together
        
        valid_diff = diff_square_sum < (match_threshold**2)*self.vars # Use threshold to filter components with large error
        matched_pixels = np.any(valid_diff, axis=2)


        # Find the best matched and valid component for each pixel (with lowest error)
        large_value = np.finfo(diff_square_sum.dtype).max # Upper-bound of datatype

        masked_error = diff_square_sum + (~valid_diff) * large_value # Push invalid components error to max
        min_err_comps = masked_error.argmin(axis=2) # Use argmin to get the minimum error component for each pixel
        matches = matched_pixels[:, :, None] & (np.arange(self.n_comps) == min_err_comps[:, :, None])
                    

        # Update weights of components
        self.weights *= (1.0 - update_alpha) # Decay the weights
        self.weights[matches] += update_alpha # Update weights of matched components only

        # Update means and variances of matched components
        self.means[matches] = (1-update_alpha)*self.means[matches]\
                                + update_alpha*frame[matched_pixels]

        self.vars[matches] = (1-update_alpha)*self.vars[matches]\
                                + update_alpha*diff_square_sum[matches]
        

        # Replace weakest component of each pixel
        rows, cols = np.where(~matched_pixels)

        if len(rows) > 0:
            weakest = self.weights.argmin(axis=2)
            weakest_comp = weakest[rows, cols]

            # Replace mean
            self.means[rows, cols, weakest_comp] = frame[rows, cols]

            # Re-init variance
            self.vars[rows, cols, weakest_comp] = INIT_VAR

            # Re-init weight
            self.weights[rows, cols, weakest_comp] = REINIT_WEIGHT

        # Normalize weights
        self.weights /= self.weights.sum(axis=2, keepdims=True) 

    def predict(self, frame: np.ndarray, match_threshold: float = 2.5, background_threshold: float = 0.7):
        
        # Distance to all components
        diff = frame[:, :, None, :] - self.means
        diff_square_sum = np.sum(diff ** 2, axis=-1)

        # Sort components by w / sigma
        rank = self.weights / (np.sqrt(self.vars) + 1e-6)
        order = np.argsort(rank, axis=2)[:, :, ::-1]

        sorted_weights = np.take_along_axis(self.weights, order, axis=2)

        cumulative_weights = np.cumsum(sorted_weights, axis=2)

        # Background components
        background_components = cumulative_weights <= background_threshold
    
        # Ensure first component is always included
        background_components[:, :, 0] = True

        # Reorder match mask using same ordering
        matches = diff_square_sum < (match_threshold**2) * self.vars

        sorted_matches = np.take_along_axis(matches, order, axis=2)

        # Pixel is background if it matches any selected background component
        background_mask = np.any(sorted_matches & background_components, axis=2)

        # Foreground mask
        foreground_mask = (~background_mask).astype(np.uint8) * 255

        return foreground_mask

    def post_mask_process(self, mask: np.ndarray):
        pass

    def step(self, frame: np.ndarray, match_threshold: float=2.5, background_threshold: float=0.7, update_alpha: float=0.01):
        self.update(frame, match_threshold, update_alpha)
        return self.predict(frame, match_threshold, background_threshold)

    def step_profiler(self, frame: np.ndarray, match_threshold: float=2.5, background_threshold: float=0.7, update_alpha: float=0.01):
        update_profile = self.update_profiler(frame, match_threshold, update_alpha)
        
        mask, predict_profile = self.predict_profiler(frame, match_threshold, background_threshold)

        return mask, update_profile, predict_profile
    
    def update_profiler(self, frame: np.ndarray, match_threshold: float=2.5, update_alpha: float=0.01):
        
        time_dict = dict()

        # Compute error between pixel and each component's mean on 3 channels (R, G, B)
        diff, time_dict['diff'] =  cpu_timer(lambda: frame[:, :, None, :] - self.means)
        diff_square_sum, time_dict['diff_square_sum'] = cpu_timer(lambda: np.sum(diff**2, axis=-1)) # Square each channel error and sum all together
        
        valid_diff, time_dict['valid_diff'] = cpu_timer(lambda: diff_square_sum < (match_threshold**2)*self.vars) # Use threshold to filter components with large error
        matched_pixels, time_dict['matched_pixels'] = cpu_timer(lambda: np.any(valid_diff, axis=2))


        # Find the best matched and valid component for each pixel (with lowest error)
        large_value, time_dict['large_value'] = cpu_timer(lambda: np.finfo(diff_square_sum.dtype).max) # Upper-bound of datatype

        masked_error, time_dict['masked_error'] = cpu_timer(lambda: diff_square_sum + (~valid_diff) * large_value) # Push invalid components error to max
        min_err_comps, time_dict['min_err_comps'] = cpu_timer(lambda: masked_error.argmin(axis=2)) # Use argmin to get the minimum error component for each pixel
        matches, time_dict['matches'] = cpu_timer(lambda: matched_pixels[:, :, None] & (np.arange(self.n_comps) == min_err_comps[:, :, None]))
                    

        # Update weights of components
        self.weights, time_dict['weight_decay'] = cpu_timer(lambda: self.weights * (1.0 - update_alpha)) # Decay the weights
        self.weights[matches], time_dict['weight_update'] = cpu_timer(lambda: self.weights[matches] + update_alpha) # Update weights of matched components only

        # Update means and variances of matched components
        self.means[matches], time_dict['mean_update'] = cpu_timer(lambda: (1-update_alpha)*self.means[matches]\
                                + update_alpha*frame[matched_pixels])

        self.vars[matches], time_dict['var_update'] = cpu_timer(lambda: (1-update_alpha)*self.vars[matches]\
                                + update_alpha*diff_square_sum[matches])
        

        # Replace weakest component of each pixel
        (rows, cols), time_dict['weakest_comps_idx'] = cpu_timer(lambda: np.where(~matched_pixels))

        time_dict['weakest'] = 0.0
        time_dict['weakest_comp'] = 0.0
        time_dict['mean_replace'] = 0.0
        time_dict['var_replace'] = 0.0
        time_dict['weight_re_init'] = 0.0
     
        if len(rows) > 0:
            weakest, time_dict['weakest'] = cpu_timer(lambda: self.weights.argmin(axis=2))
            weakest_comp, time_dict['weakest_comp'] = cpu_timer(lambda: weakest[rows, cols])

            # Replace mean
            self.means[rows, cols, weakest_comp], time_dict['mean_replace'] = cpu_timer(lambda: frame[rows, cols])

            # Re-init variance
            self.vars[rows, cols, weakest_comp], time_dict['var_replace'] = cpu_timer(lambda: 400.0)

            # Re-init weight
            self.weights[rows, cols, weakest_comp], time_dict['weight_re_init'] = cpu_timer(lambda: 0.01)

        # Normalize weights
        self.weights, time_dict['weight_normalize'] = cpu_timer(lambda: self.weights / self.weights.sum(axis=2, keepdims=True))

        return time_dict


    def predict_profiler(self, frame: np.ndarray, match_threshold: float = 2.5, background_threshold: float = 0.7):
        time_dict = dict()


        # Distance to all components
        diff, time_dict['diff'] = cpu_timer(lambda: frame[:, :, None, :] - self.means)
        diff_square_sum, time_dict['diff_square_sum'] = cpu_timer(lambda: np.sum(diff ** 2, axis=-1))

        # Sort components by w / sigma
        rank, time_dict['rank'] = cpu_timer(lambda: self.weights / (np.sqrt(self.vars) + 1e-6))
        order, time_dict['order'] = cpu_timer(lambda: np.argsort(rank, axis=2)[:, :, ::-1])

        sorted_weights, time_dict['sort_weight'] = cpu_timer(lambda: np.take_along_axis(self.weights, order, axis=2))

        cumulative_weights, time_dict['weight_accumulate'] = cpu_timer(lambda: np.cumsum(sorted_weights, axis=2))

        # Background components
        background_components, time_dict['background_comps'] = cpu_timer(lambda: cumulative_weights <= background_threshold)
    
        # Ensure first component is always included
        background_components[:, :, 0], time_dict['first_comp'] = cpu_timer(lambda: True)

        # Reorder match mask using same ordering
        matches, time_dict['matches'] = cpu_timer(lambda: diff_square_sum < (match_threshold**2) * self.vars)

        sorted_matches, time_dict['sort_matches'] = cpu_timer(lambda: np.take_along_axis(matches, order, axis=2))

        # Pixel is background if it matches any selected background component
        background_mask, time_dict['background_mask'] = cpu_timer(lambda: np.any(sorted_matches & background_components, axis=2))

        # Foreground mask
        foreground_mask, time_dict['foreground_mask'] = cpu_timer(lambda: (~background_mask).astype(np.uint8) * 255)

        return foreground_mask, time_dict