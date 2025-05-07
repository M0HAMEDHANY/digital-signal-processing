import numpy as np


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate PSNR between two images or videos (frame‐by‐frame if 3D).
    """
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))
