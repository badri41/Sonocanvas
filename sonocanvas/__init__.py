r"""SonoCanvas: embed an image in the STFT magnitude of an audio signal.

encode: image -> S (freq_map) -> Griffin-Lim phase -> weighted OLA -> x.
decode: x -> |STFT(x)| on the band -> image on the target grid.
"""
import numpy as np

from .freq_map import image_to_mag
from .image_prep import load, preprocess, render_text
from .metrics import psnr, recover, spectral_convergence, ssim
from .phase import griffin_lim
from .stft import istft, stft, window
from .stft import window as _make_window

__all__ = ["encode", "decode", "load", "render_text", "preprocess", "image_to_mag",
           "stft", "istft", "window", "griffin_lim", "recover", "ssim", "psnr",
           "spectral_convergence"]


def encode(img: np.ndarray, *, n_fft: int = 1024, hop: int = 256, dur: float = 5.0,
           f_lo: float = 1000, f_hi: float = 8000, sr: int = 22050, scale: str = "linear",
           window: str = "hann", n_iter: int = 60, init: str = "random",
           seed: int = 0) -> tuple[np.ndarray, dict]:
    r"""Image -> waveform x with |STFT(x)| \approx S(img); x is peak-normalised: x \leftarrow 0.9\,x/\max|x|.

    info: S (full target magnitude), band (bin mask), target (S[band] flipped, / max,
    the grid decode() returns), errs (Griffin-Lim spectral convergence), sr.
    """
    win = _make_window(window, n_fft)
    S, band = image_to_mag(img, n_fft, hop, dur, f_lo, f_hi, sr, scale)
    x, errs = griffin_lim(S, n_fft, hop, win, n_iter, init, seed)
    x = 0.9 * x / np.abs(x).max()
    target = S[band][::-1]
    target = target / target.max()
    return x, {"S": S, "band": band, "target": target, "errs": errs, "sr": sr}


def decode(x: np.ndarray, *, n_fft: int = 1024, hop: int = 256, dur: float = 5.0,
           f_lo: float = 1000, f_hi: float = 8000, sr: int = 22050, scale: str = "linear",
           window: str = "hann") -> np.ndarray:
    r"""Waveform -> image \hat{I} = |STFT(x)|[band] / max, on the same (bins x frames) grid as
    encode's target. Uses only the shared parameters, not encoder state."""
    win = _make_window(window, n_fft)
    n_frames = 1 + int(dur * sr) // hop
    _, band = image_to_mag(np.zeros((2, 2)), n_fft, hop, dur, f_lo, f_hi, sr, scale)
    return recover(x, n_fft, hop, win, band, (int(band.sum()), n_frames))
