"""Decoding audio back to an image, and image/spectral quality metrics."""
import numpy as np
from scipy.stats import rankdata
from skimage.metrics import structural_similarity
from skimage.transform import resize

from .stft import stft


def spectral_convergence(Y: np.ndarray, S: np.ndarray) -> float:
    r"""SC = \| |Y| - S \|_F / \| S \|_F  for a complex STFT Y and target magnitude S."""
    return float(np.linalg.norm(np.abs(Y) - S) / np.linalg.norm(S))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    r"""Mean SSIM (Wang et al. 2004) on images in [0, 1]:
    SSIM = \frac{(2\mu_a\mu_b + C_1)(2\sigma_{ab} + C_2)}{(\mu_a^2+\mu_b^2+C_1)(\sigma_a^2+\sigma_b^2+C_2)}.
    """
    return float(structural_similarity(a, b, data_range=1.0))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    r"""PSNR = 10 \log_{10}(1 / \mathrm{MSE}) dB for images in [0, 1]."""
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))


def auc(target: np.ndarray, rec: np.ndarray) -> float:
    r"""Stroke/background ROC-AUC of the decoded image, as the Mann-Whitney statistic:
    A = P(\hat{I}_s > \hat{I}_b) + \tfrac12 P(\hat{I}_s = \hat{I}_b), for s a stroke pixel (T > 0.5)
    and b a background pixel. Depends only on the order of pixel values, so any increasing map of
    \hat{I} leaves it unchanged. A = 1 is perfect separation; A = 0.5 is none.
    """
    stroke = target.ravel() > 0.5
    n1, n0 = stroke.sum(), (~stroke).sum()
    return float((rankdata(rec.ravel())[stroke].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recover(x: np.ndarray, n_fft: int, hop: int, win: np.ndarray,
            band: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    r"""Decoded image: \hat{I} = |STFT(x)|[band] / max, flipped so row 0 = highest frequency,
    resampled to `shape` (the target grid)."""
    Y = np.abs(stft(x, n_fft, hop, win))[band]
    Y = Y / Y.max()
    return resize(Y[::-1], shape, anti_aliasing=True)
