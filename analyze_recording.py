r"""Score a real recording of a demo WAV: align it, decode it, and report AUC and SSIM.

    python analyze_recording.py take.wav --word DA321 --dist 2.0 --device phone [--video]

Takes a WAV, or any format ffmpeg can read (.m4a from a phone, for example). Writes a row to
results/acoustic.csv, the images to figures/acoustic/, and with --video also data/acoustic_video.npz
for scene 7. Re-running a take replaces its row.

Alignment: the delay is the peak of the band-limited cross-correlation with the encoded signal,
d^* = \arg\max_d |\sum_n r[n+d]\,x[n]|, computed by FFT. decode() then sees exactly the 5 s clip.
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt

import sonocanvas as sc

SR, F_LO, F_HI = 22050, 1000, 8000
CSV = Path("results/acoustic.csv")
HEADER = ["word", "dist_m", "device", "take", "auc", "ssim", "psnr_db", "delay_s", "peak_ratio", "snr_db"]
MIN_PEAK_RATIO = 40.0  # correct takes gave 130-158 in simulation, the wrong word 19


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    """Mono float audio. Anything other than WAV goes through ffmpeg first."""
    if path.suffix.lower() != ".wav":
        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg is needed to read this format; convert the file to WAV first")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", str(out)], check=True)
            return read_audio(out)
    sr, a = wavfile.read(path)
    a = a.astype(float)
    if a.ndim > 1:
        a = a.mean(1)
    return sr, a / max(np.abs(a).max(), 1e-12)


def to_rate(a: np.ndarray, sr: int) -> np.ndarray:
    g = gcd(sr, SR)
    return resample_poly(a, SR // g, sr // g) if sr != SR else a


def align(r: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, int, float]:
    """The len(x) samples of r matching x, the delay in samples, and peak / median of |correlation|."""
    sos = butter(6, [F_LO, F_HI], "bandpass", fs=SR, output="sos")
    c = np.abs(correlate(sosfiltfilt(sos, r), x, mode="valid", method="fft"))
    d = int(np.argmax(c))
    return r[d:d + len(x)], d, float(c[d] / np.median(c))


def snr_db(r: np.ndarray, d: int, n: int) -> float:
    r"""In-band SNR from the room noise before and after the clip: 10\log_{10}(P_{clip}/P_{noise} - 1)."""
    sos = butter(6, [F_LO, F_HI], "bandpass", fs=SR, output="sos")
    b = sosfiltfilt(sos, r)
    guard = SR // 10  # skip the clip's onset and any tail
    quiet = np.r_[b[:max(d - guard, 0)], b[d + n + guard:]]
    if len(quiet) < SR // 2:
        return float("nan")
    return float(10 * np.log10(max(np.mean(b[d:d + n] ** 2) / np.mean(quiet ** 2) - 1, 1e-6)))


def save_row(row: list) -> None:
    rows = []
    if CSV.exists():
        with open(CSV, newline="") as f:
            rows = [r for r in list(csv.reader(f))[1:] if r[:4] != [str(v) for v in row[:4]]]
    rows.append([str(v) for v in row])
    rows.sort(key=lambda r: (r[0], float(r[1]), r[2], r[3]))
    with open(CSV, "w", newline="") as f:
        csv.writer(f).writerows([HEADER] + rows)


def figure(target: np.ndarray, rec: np.ndarray, title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True, layout="constrained")
    for a, im, t in ((ax[0], target, "target"), (ax[1], rec, title)):
        a.imshow(im, cmap="magma", aspect="auto", vmin=0, vmax=1, extent=[0, 5, 1, 8])
        a.set(title=t, ylabel="frequency (kHz)")
    ax[1].set_xlabel("time (s)")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--word", default="DA321", help="the word in the WAV that was played")
    ap.add_argument("--dist", type=float, required=True, help="speaker-to-microphone distance in metres")
    ap.add_argument("--device", default="phone", help="recording device, e.g. phone or mac")
    ap.add_argument("--take", type=int, default=1)
    ap.add_argument("--video", action="store_true", help="use this take in scene 7 of the video")
    a = ap.parse_args()

    x, info = sc.encode(sc.preprocess(sc.render_text(a.word), mode="edges"))  # the WAVs in demo/ exactly
    sr, r = read_audio(a.recording)
    r = to_rate(r, sr)
    if len(r) < len(x):
        sys.exit(f"recording is {len(r) / SR:.1f} s but the clip is {len(x) / SR:.1f} s: it cannot contain it")
    y, d, ratio = align(r, x)
    if ratio < MIN_PEAK_RATIO:
        print(f"WARNING: weak correlation peak (ratio {ratio:.1f} < {MIN_PEAK_RATIO}). Check that the word and "
              f"file match; these scores may be meaningless.")
    rec = sc.decode(y)
    T = info["target"]
    u, s, p = sc.auc(T, rec), sc.ssim(T, rec), sc.psnr(T, rec)
    snr = snr_db(r, d, len(x))
    save_row([a.word, a.dist, a.device, a.take, f"{u:.4f}", f"{s:.4f}", f"{p:.2f}", f"{d / SR:.3f}", f"{ratio:.1f}", f"{snr:.1f}"])
    name = f"{a.word}_{a.dist:g}m_{a.device}_{a.take}"
    Path("figures/acoustic").mkdir(parents=True, exist_ok=True)
    figure(T, rec, f"recorded at {a.dist:g} m, {a.device}: AUC {u:.4f}, SSIM {s:.3f}", Path(f"figures/acoustic/{name}.png"))
    print(f"{name}: delay {d / SR:.3f} s, peak ratio {ratio:.1f}, in-band SNR {snr:.1f} dB, "
          f"AUC {u:.4f}, SSIM {s:.3f}, PSNR {p:.1f} dB")
    print(f"wrote {CSV} and figures/acoustic/{name}.png")
    if a.video:
        np.savez_compressed("data/acoustic_video.npz", x=y.astype(np.float32), rec=rec.astype(np.float32),
                            target=T.astype(np.float32), dist=a.dist)
        print("wrote data/acoustic_video.npz: scene 7 will use this take")


if __name__ == "__main__":
    main()
