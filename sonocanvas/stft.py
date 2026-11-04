"""Analysis/synthesis operators: windows, STFT, weighted overlap-add ISTFT (from scratch)."""
import numpy as np


def window(name: str, N: int) -> np.ndarray:
    r"""Periodic (DFT-even) window of length N, n = 0..N-1.

    hann:     w[n] = 1/2 - 1/2 cos(2\pi n/N)
    hamming:  w[n] = 0.54 - 0.46 cos(2\pi n/N)
    blackman: w[n] = 0.42 - 0.5 cos(2\pi n/N) + 0.08 cos(4\pi n/N)
    rect:     w[n] = 1
    """
    t = 2 * np.pi * np.arange(N) / N
    if name == "hann":
        return 0.5 - 0.5 * np.cos(t)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(t)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(t) + 0.08 * np.cos(2 * t)
    if name == "rect":
        return np.ones(N)
    raise ValueError(f"unknown window {name!r}")


def stft(x: np.ndarray, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    r"""Centre-padded STFT, shape (n_fft//2 + 1, n_frames).

    x is zero-padded by N/2 on both sides, then
    X[k, m] = \sum_{n=0}^{N-1} x[n + mH] w[n] e^{-j 2\pi k n / N},  k = 0..N/2.
    Phase is frame-relative: X[k,m] = <x, T_{mH}(w e_k)>, which differs from the
    absolute-time convention by e^{-j 2\pi k m H / N}; |X| is identical.
    """
    pad = n_fft // 2
    x = np.pad(x, pad)
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n_frames)], 1)
    return np.fft.rfft(frames, axis=0)


def istft(X: np.ndarray, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    r"""Weighted overlap-add (least-squares ISTFT, Griffin & Lim 1984).

    y_m = IDFT(X[:, m]);  \hat{x}[n] = \sum_m w[n - mH] y_m[n - mH] / \sum_m w^2[n - mH].
    Returns H (n_frames - 1) samples after removing the N/2 centre padding, i.e.
    H \lfloor L/H \rfloor samples for an STFT of a length-L signal.
    """
    frames = np.fft.irfft(X, n=n_fft, axis=0) * win[:, None]
    n = n_fft + hop * (X.shape[1] - 1)
    y = np.zeros(n)
    wsum = np.zeros(n)
    for m in range(X.shape[1]):
        y[m * hop:m * hop + n_fft] += frames[:, m]
        wsum[m * hop:m * hop + n_fft] += win ** 2
    y /= np.maximum(wsum, 1e-8)
    pad = n_fft // 2
    return y[pad:-pad]
