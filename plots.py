"""Draws figures/<name>.{pdf,png} from data/<name>.npz, which bin/experiments wrote.

    python plots.py                    # all of them
    python plots.py noise_robustness   # just one

I kept this in Python because matplotlib has no comparable C++ library. It only draws; it computes nothing.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = ("text", "line", "photo")
LABEL = {"text": "text", "line": "line drawing", "photo": "photo"}
MODE = {"text": "edges", "line": "edges", "photo": "raw"}
FIG, DATA = Path("figures"), Path("data")
COLOR = {"text": "#2a78d6", "line": "#eb6834", "photo": "#1baf7a"}
MARK = {"text": "o", "line": "s", "photo": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4,
    "legend.frameon": False, "savefig.dpi": 220,
})


def show(a: plt.Axes, img: np.ndarray, title: str) -> None:
    """One picture in magma with a title and no axes."""
    a.imshow(img, cmap="magma", aspect="auto", vmin=0, vmax=1)
    a.set_title(title, pad=2)
    a.set(xticks=[], yticks=[])
    a.grid(False)
    a.spines[:].set_visible(False)


def plot_edges_vs_filled(d) -> plt.Figure:
    """Experiment 1: target and decoded, filled vs edges, for all three pictures."""
    fig, ax = plt.subplots(3, 4, figsize=(7.16, 3.5), layout="constrained")
    for i, n in enumerate(NAMES):
        for j, mode in enumerate(("raw", "edges")):
            for a, key, title in ((ax[i, 2 * j], "target", f"{LABEL[n]}: {mode}, target"),
                                  (ax[i, 2 * j + 1], "rec", f"AUC {d[f'{n}_{mode}_auc']:.4f}, SSIM {d[f'{n}_{mode}_ssim']:.3f}")):
                show(a, d[f"{n}_{mode}_{key}"], title)
    return fig


def plot_sweep_window_len(d) -> plt.Figure:
    """Experiment 2: scores against window length N, plus the decoded text at a few N."""
    fig = plt.figure(figsize=(3.5, 3.9), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 3.3])
    ax = top.subplots(1, 3)
    for a, N in zip(ax, (128, 1024, 4096)):
        show(a, d[f"text_rec_{N}"], f"$N={N}$")
    a, b = bot.subplots(2, 1, sharex=True)
    L = d["lengths"]
    for n in NAMES:
        a.plot(L, 1 - d[f"{n}_image_auc"].mean(1), color=COLOR[n], marker=MARK[n], label=LABEL[n])
        b.plot(L, d[f"{n}_image"].mean(1), color=COLOR[n], marker=MARK[n], label=LABEL[n])
    own, = b.plot(L, d["text_own"].mean(1), color=COLOR["text"], marker=MARK["text"], ls=":", mfc="white",
                  label="text, own grid")
    a.set_yscale("log")
    b.set_xscale("log", base=2)
    b.set_xticks(L, [str(v) for v in L])
    b.minorticks_off()
    a.set(ylabel="$1-$AUC")
    b.set(xlabel="window length $N$ (hop $N/4$)", ylabel="SSIM", ylim=(0.3, 1))
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    b.legend(handles=[own], loc="lower right")
    return fig


WINDOWS = ("rect", "hann", "hamming", "blackman")


def plot_sweep_window_type(d) -> plt.Figure:
    """Experiment 3: what each window looks like in frequency, and how it scores."""
    fig, (a, c, b) = plt.subplots(1, 3, figsize=(7.16, 2.3), layout="constrained", width_ratios=[1.5, 1, 1])
    for w, col, ls in zip(WINDOWS, ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6"), ("-", "-", "--", ":")):
        a.plot(d[f"{w}_bins"], d[f"{w}_db"], color=col, ls=ls, lw=1.1,
               label=f"{w} ({float(d[f'{w}_sidelobe_db']):.0f} dB)")
    a.set(xlabel="offset from centre (DFT bins)", ylabel="$|W|$ (dB re peak)", ylim=(-100, 3), xlim=(0, 12))
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    x = np.arange(len(WINDOWS))
    for i, n in enumerate(NAMES):
        off = x + (i - 1) * 0.1
        c.plot(off, [1 - d[f"{n}_{w}_auc"].mean() for w in WINDOWS], color=COLOR[n], marker=MARK[n], ls="", label=LABEL[n])
        b.plot(off, [d[f"{n}_{w}_ssim"].mean() for w in WINDOWS], color=COLOR[n], marker=MARK[n], ls="", label=LABEL[n])
    for p in (b, c):
        p.set_xticks(x, ["rect", "hann", "hamm.", "black."])
        p.set_xlim(-0.4, 3.4)
    c.set_yscale("log")
    c.set(ylabel="$1-$AUC")
    b.set(ylabel="SSIM", ylim=(0, 1))
    c.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    return fig


def plot_gl_iterations(d) -> plt.Figure:
    """Experiment 4: Griffin-Lim error and scores against the number of iterations."""
    fig, (a, c, b) = plt.subplots(3, 1, figsize=(3.5, 4.0), sharex=True, layout="constrained")
    it = np.arange(len(d["text_errs_zero"]))
    for n in NAMES:
        a.plot(it + 1, d[f"{n}_errs_random"].mean(0), color=COLOR[n], label=f"{LABEL[n]}, random")
        a.plot(it + 1, d[f"{n}_errs_zero"], color=COLOR[n], ls=":", label=f"{LABEL[n]}, zero")
        c.plot(d["iters"] + 1, 1 - d[f"{n}_auc_random"].mean(1), color=COLOR[n], marker=MARK[n])
        b.plot(d["iters"] + 1, d[f"{n}_ssim_random"].mean(1), color=COLOR[n], marker=MARK[n])
        b.plot(d["iters"] + 1, d[f"{n}_ssim_zero"], color=COLOR[n], marker=MARK[n], ls=":", mfc="white")
    a.set_xscale("log")
    c.set_yscale("log")
    ticks = [1, 2, 3, 6, 11, 21, 61, 101, 201]
    b.set_xticks(ticks, [str(t - 1) for t in ticks])
    b.minorticks_off()
    a.set(ylabel="spectral convergence")
    c.set(ylabel="$1-$AUC")
    b.set(xlabel="Griffin-Lim iteration", ylabel="SSIM")
    a.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=0.9)
    return fig


def plot_mel_vs_linear(d) -> plt.Figure:
    """Experiment 5: linear or mel encoding, viewed on a linear or mel axis."""
    fig, ax = plt.subplots(2, 2, figsize=(3.5, 2.2), layout="constrained")
    for i, enc in enumerate(("linear", "mel")):
        for j, vw in enumerate(("linear", "mel")):
            show(ax[i, j], d[f"text_{enc}_{vw}_img"],
                 f"{enc} $\\to$ {vw}: AUC {float(d[f'text_{enc}_{vw}_auc']):.3f}")
    fig.suptitle("encoding axis $\\to$ viewing axis", fontsize=8)
    return fig


def plot_noise_robustness(d) -> plt.Figure:
    """Experiment 6: scores against SNR, same SNR vs same noise floor."""
    fig = plt.figure(figsize=(3.5, 4.5), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 4.4])
    for a, snr in zip(top.subplots(1, 3), (20, 10, 0)):
        show(a, d[f"text_edges_rec_{snr}"], f"{snr} dB")
    a, m, b = bot.subplots(3, 1, sharex=True)
    s = d["snrs"]
    for n in NAMES:
        y, u = d[f"{n}_{MODE[n]}_ssim"], d[f"{n}_{MODE[n]}_auc"]
        a.plot(s, 1 - u.mean(1), color=COLOR[n], marker=MARK[n], label=LABEL[n])
        m.errorbar(s, y.mean(1), y.std(1), color=COLOR[n], marker=MARK[n], capsize=2, label=LABEL[n])
    a.set_yscale("log")
    a.set(ylabel="$1-$AUC")
    m.set(ylabel="SSIM", ylim=(0, 1))
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    for key, ls, tag in (("auc", "-", "same SNR"), ("floor_auc", ":", "same noise floor")):
        b.plot(s, 1 - d[f"text_raw_{key}"].mean(1), color=MUTED, marker="D", ls=ls, mfc="white", label=f"filled, {tag}")
        b.plot(s, 1 - d[f"text_edges_{key}"].mean(1), color=COLOR["text"], marker="o", ls=ls, label=f"edges, {tag}")
    b.set_yscale("log")
    b.set(xlabel="SNR (dB)", ylabel="text $1-$AUC")
    b.invert_xaxis()
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    return fig

PLOTS = {f.__name__[5:]: f for f in (plot_edges_vs_filled, plot_sweep_window_len, plot_sweep_window_type,
                                      plot_gl_iterations, plot_mel_vs_linear, plot_noise_robustness)}

if __name__ == "__main__":
    for name in sys.argv[1:] or PLOTS:
        fig = PLOTS[name](np.load(DATA / f"{name}.npz"))
        fig.savefig(FIG / f"{name}.pdf")
        fig.savefig(FIG / f"{name}.png")
        plt.close(fig)
        print(f"figures/{name}.pdf")
