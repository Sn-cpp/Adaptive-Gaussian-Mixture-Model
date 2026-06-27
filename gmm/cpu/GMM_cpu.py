import numpy as np

np.random.seed(42)

class GMM_CPU:
    def __init__(self, first_frame: np.ndarray, n_components: int=1):        
        self.height, self.width, _ = first_frame.shape
        
        self.n_comps = n_components

        # First component mean with the first frame
        self.means = np.ones(shape=(self.height, self.width, self.n_comps, 3), dtype=np.float32)
        self.means[:, :, 0, :] = first_frame

        # All variances to a fixed value: 400
        self.vars = np.full(shape=(self.height, self.width, self.n_comps), fill_value=400, dtype=np.float32)
        
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
        unmatched_pixels = ~matched_pixels
        rows, cols = np.where(unmatched_pixels)

        if len(rows) > 0:
            weakest = self.weights.argmin(axis=2)
            weakest_comp = weakest[rows, cols]

            # Replace mean
            self.means[rows, cols, weakest_comp] = frame[rows, cols]

            # Re-init variance
            self.vars[rows, cols, weakest_comp] = 400.0

            # Re-init weight
            self.weights[rows, cols, weakest_comp] = 0.01

            # Normalize weights
            self.weights /= self.weights.sum(axis=2, keepdims=True) 

    def predict(self, frame: np.ndarray, match_threshold: float = 2.5, background_threshold: float = 0.7):
        
        # Distance to all components
        diff = frame[:, :, None, :] - self.means
        dist2 = np.sum(diff ** 2, axis=-1)

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
        matches = dist2 < (match_threshold**2) * self.vars

        sorted_matches = np.take_along_axis(matches, order, axis=2)

        # Pixel is background if it matches any selected background component
        background_mask = np.any(sorted_matches & background_components, axis=2)

        return background_mask.astype(np.uint8) * 255
        # Foreground mask
        foreground_mask = ~background_mask

        return foreground_mask.astype(np.uint8) * 255

    def step(self, frame: np.ndarray, match_threshold: float=2.5, background_threshold: float=0.7, update_alpha: float=0.01):
        self.update(frame, match_threshold, update_alpha)
        return self.predict(frame, match_threshold, background_threshold)

