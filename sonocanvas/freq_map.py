"""Map an image onto a target STFT magnitude."""
import numpy as np
from skimage.transform import resize


def mel(f: np.ndarray) -> np.ndarray:
    r"""m(f) = 2595 \log_{10}(1 + f/700)  (Stevens, Volkmann & Newman 1937; O'Shaughnessy form)."""
    return 2595 * np.log10(1 + f / 700)


def image_to_mag(img: np.ndarray, n_fft: int, hop: int, dur: float, f_lo: float, f_hi: float,
                 sr: int, scale: str = "linear") -> tuple[np.ndarray, np.ndarray]:
    r"""Target magnitude S, shape (N/2+1, 1 + \lfloor dur\cdot sr \rfloor / H), and band mask.

    Columns are resampled to frames; image row 0 (top) maps to f_hi. Bin k at f_k = k\,sr/N
    has normalised position p_k = (f_k - f_lo)/(f_hi - f_lo) (linear) or
    p_k = (m(f_k) - m(f_lo))/(m(f_hi) - m(f_lo)) (mel); S[k, m] = interp of the flipped
    column at p_k for p_k \in [0, 1], else 0.
    """
    n_bins = n_fft // 2 + 1
    n_frames = 1 + int(dur * sr) // hop
    freqs = np.arange(n_bins) * sr / n_fft
    S = np.zeros((n_bins, n_frames))
    im = resize(img, (img.shape[0], n_frames), anti_aliasing=True)
    im = im[::-1]  # row 0 of image = top = highest frequency
    rows = np.linspace(0, 1, im.shape[0])
    if scale == "mel":
        pos = (mel(freqs) - mel(f_lo)) / (mel(f_hi) - mel(f_lo))
    elif scale == "linear":
        pos = (freqs - f_lo) / (f_hi - f_lo)
    else:
        raise ValueError(f"unknown scale {scale!r}")
    band = (pos >= 0) & (pos <= 1)
    for j in range(n_frames):
        S[band, j] = np.interp(pos[band], rows, im[:, j])
    return S, band
