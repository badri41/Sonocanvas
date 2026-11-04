"""Phase recovery by Griffin-Lim alternating projections."""
import numpy as np

from .metrics import spectral_convergence
from .stft import istft, stft


def griffin_lim(S: np.ndarray, n_fft: int, hop: int, win: np.ndarray, n_iter: int = 60,
                init: str = "random", seed: int = 0) -> tuple[np.ndarray, list[float]]:
    r"""Griffin-Lim (1984): X^{(i+1)} = S \cdot e^{j \angle STFT(ISTFT(X^{(i)}))}.

    init='random': X^{(0)} = S e^{j 2\pi U}, U ~ Uniform[0,1) (seeded);  init='zero': X^{(0)} = S.
    errs[i] = spectral convergence of STFT(ISTFT(X^{(i)})) against S.
    Returns (ISTFT(X^{(n_iter)}), errs).
    """
    if init == "random":
        rng = np.random.default_rng(seed)
        X = S * np.exp(2j * np.pi * rng.random(S.shape))
    elif init == "zero":
        X = S.astype(complex)
    else:
        raise ValueError(f"unknown init {init!r}")
    errs = []
    for _ in range(n_iter):
        x = istft(X, n_fft, hop, win)
        Y = stft(x, n_fft, hop, win)[:, :S.shape[1]]
        errs.append(spectral_convergence(Y, S))
        X = S * np.exp(1j * np.angle(Y))
    return istft(X, n_fft, hop, win), errs
