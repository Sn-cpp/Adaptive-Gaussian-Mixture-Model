import numpy as np

from post_processing.post_processing_common import PostProcessingBase

def median_blur(img: np.ndarray, ksize: int):
    H, W  = img.shape
    pad = ksize // 2
    padded = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img, dtype=img.dtype)

    for r in range(H):
        for c in range(W):
            window = padded[r:r+ksize, c:c+ksize]
            out[r, c] = np.median(window)

    return out

def convolve2d(img: np.ndarray, kernel: np.ndarray):
    H, W = img.shape
    K    = kernel.shape[0]
    half = K // 2
    padded = np.pad(img, half, mode='constant', constant_values=0.0)
    out = np.zeros_like(img, dtype=img.dtype)
    for r in range(H):
        for c in range(W):
            out[r, c] = np.sum(padded[r:r+K, c:c+K] * kernel)

    return out

def dilate2d(img: np.ndarray, kernel: np.ndarray):
    H, W = img.shape
    K    = kernel.shape[0]
    half = K // 2
    
    padded = np.pad(img, half, mode='constant', constant_values=0.0)
    out = np.zeros_like(img, dtype=img.dtype)
        
    morph_kernel = np.where(kernel == 1, 0.0, -np.inf)
    
    for r in range(H):
        for c in range(W):
            window = padded[r:r+K, c:c+K]
            out[r, c] = np.max(window + morph_kernel)

    return out

def erode2d(img: np.ndarray, kernel: np.ndarray):
    H, W = img.shape
    K    = kernel.shape[0]
    half = K // 2

    max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
    padded = np.pad(img, half, mode='constant', constant_values=max_val)
    
    out = np.zeros_like(img, dtype=img.dtype)
    flipped_kernel = np.flip(kernel)
    
    morph_kernel = np.where(flipped_kernel == 1, 0.0, np.inf)
    
    for r in range(H):
        for c in range(W):
            window = padded[r:r+K, c:c+K]
            out[r, c] = np.min(window - morph_kernel)

    return out


class PostProcessingCPU(PostProcessingBase):
    def _apply_kernel(self, frame, mask, args):
        median_mask = median_blur(mask, 5)

        morph_mask = median_mask
        for i in range(2):
            morph_mask = erode2d(morph_mask, self.kernel_open)
            morph_mask = dilate2d(morph_mask, self.kernel_open)

        morph_mask = dilate2d(morph_mask, self.kernel_close)
        morph_mask = erode2d(morph_mask, self.kernel_close)

        morph_mask = dilate2d(morph_mask, self.kernel_dilate)

        self.processed_frame = convolve2d(frame, self.kernel_gauss)
        boolean_mask = (morph_mask > 0)[: , :, np.newaxis]
        np.copyto(self.processed_frame, frame, where=boolean_mask)

        
