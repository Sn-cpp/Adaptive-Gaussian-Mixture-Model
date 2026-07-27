import numpy as np
import cupy as cp

def compare_diff_square_sum(cpu: np.ndarray,
                            gpu: np.ndarray,
                            atol: float = 1e-5,
                            max_diffs: int = 10):
    """
    Compare CPU and GPU diff_square_sum tensors.

    Parameters
    ----------
    cpu : np.ndarray
        CPU tensor of shape (K, H, W)

    gpu : cp.ndarray
        GPU tensor of shape (K, H, W)

    atol : float
        Absolute tolerance.

    max_diffs : int
        Maximum number of differing entries to print.
    """

    gpu = cp.asnumpy(gpu)

    # print("\n")

    # print(f"CPU shape : {cpu.shape}, dtype={cpu.dtype}")
    # print(f"GPU shape : {gpu.shape}, dtype={gpu.dtype}")

    if cpu.shape != gpu.shape:
        raise ValueError("Shape mismatch")

    abs_err = np.abs(cpu - gpu)

    # print(f"Max abs error : {abs_err.max():.6f}")
    # print(f"Mean abs error: {abs_err.mean():.6f}")

    bad = np.argwhere(abs_err > atol)

    print(f"Different entries (> {atol}): {len(bad)} / {cpu.size}")

    # if len(bad) == 0:
    #     print("✓ Tensors match.")
    #     print("--------------------------------------------------------------------------")
    #     return

    # print("--------------------------------------------------------------------------")