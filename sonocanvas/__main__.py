"""CLI: python -m sonocanvas "HELLO" -o hello.wav [--png hello.png] [--mode edges|raw]"""
import argparse

import numpy as np
from scipy.io import wavfile

from . import decode, encode, preprocess, psnr, render_text, ssim


def figure(target: np.ndarray, rec: np.ndarray, info: dict, s: float, p: float,
           dur: float = 5.0, f_lo: float = 1000, f_hi: float = 8000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [0, dur, f_lo / 1000, f_hi / 1000]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    ax[0].imshow(target, cmap="magma", aspect="auto", extent=ext)
    ax[0].set_title("target magnitude $S$")
    ax[1].imshow(rec, cmap="magma", aspect="auto", extent=ext)
    ax[1].set_title(f"decoded from audio: SSIM {s:.3f}, PSNR {p:.1f} dB, "
                    f"spectral convergence {info['errs'][-1]:.3f}")
    for a in ax:
        a.set_ylabel("frequency (kHz)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m sonocanvas",
                                 description="Hide text in the spectrogram of a WAV file.")
    ap.add_argument("text")
    ap.add_argument("-o", "--out", required=True, help="output WAV path")
    ap.add_argument("--png", help="optional target-vs-decoded spectrogram image")
    ap.add_argument("--mode", choices=["edges", "raw"], default="edges")
    a = ap.parse_args()

    x, info = encode(preprocess(render_text(a.text), mode=a.mode))
    rec = decode(x)
    s, p = ssim(info["target"], rec), psnr(info["target"], rec)
    wavfile.write(a.out, info["sr"], (x * 32767).astype(np.int16))
    print(f"wrote {a.out} ({len(x) / info['sr']:.2f} s @ {info['sr']} Hz)  "
          f"SSIM={s:.4f}  PSNR={p:.2f} dB  SC={info['errs'][-1]:.4f}")
    if a.png:
        figure(info["target"], rec, info, s, p).savefig(a.png, dpi=120)
        print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
