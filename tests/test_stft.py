import numpy as np

from sonocanvas import istft, stft, window

N, H, SR = 1024, 256, 22050


def test_roundtrip_noise():
    # Length is a multiple of H: istft returns H*floor(L/H) samples, so the tail would be cut otherwise.
    x = np.random.default_rng(0).standard_normal(200 * H)
    w = window("hann", N)
    y = istft(stft(x, N, H, w), N, H, w)
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-8


def test_hann_cola_at_quarter_hop():
    # sum_m w^2[n - mH] for Hann at H = N/4 equals 3N/(8H) = 1.5 away from the edges.
    w = window("hann", N)
    M = 40
    acc = np.zeros(N + H * (M - 1))
    for m in range(M):
        acc[m * H:m * H + N] += w ** 2
    interior = acc[N:-N]
    assert np.ptp(interior) < 1e-12
    assert np.allclose(interior, 3 * N / (8 * H))


def test_tone_peak_bin():
    f0 = 3000.0
    t = np.arange(SR) / SR
    mag = np.abs(stft(np.sin(2 * np.pi * f0 * t), N, H, window("hann", N))).mean(axis=1)
    k = int(np.argmax(mag))
    assert abs(k * SR / N - f0) <= SR / N
