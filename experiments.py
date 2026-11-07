r"""SonoCanvas controlled experiments (SPEC.md section 3).

Each experiment computes data/<name>.npz, then plots figures/<name>.{pdf,png} from that file only.
results/table.csv collects every (experiment, image, param, AUC, AUC std, SSIM, SSIM std, PSNR).
AUC is the legibility score: how well decoded intensity separates target pixels from background.
On the photo, 'target pixels' just means T > 0.5, the bright half of a continuous-tone image.

    python experiments.py                       # everything
    python experiments.py noise_robustness      # one experiment
    python experiments.py --plot-only           # redraw figures from data/*.npz

Test images: text = render_text('DA321'); line = skimage horse silhouette (CC0), whose 'edges' form
is the line drawing; photo = skimage chelsea (CC0). Unless an experiment varies it, text and line use
mode='edges' and photo uses mode='raw'. Means and std are over SEEDS (Griffin-Lim or noise seeds).
"""
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter
from skimage import data as skdata
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from skimage.transform import resize

import sonocanvas as sc

SEEDS = tuple(range(5))
SR, F_LO, F_HI, DUR = 22050, 1000, 8000, 5.0
NAMES = ("text", "line", "photo")
LABEL = {"text": "text", "line": "line drawing", "photo": "photo"}
MODE = {"text": "edges", "line": "edges", "photo": "raw"}
FIG, DATA, RES = Path("figures"), Path("data"), Path("results")
TABLE = RES / "table.csv"

# Validated categorical slots 1-3 (all-pairs CVD dE >= 9.2); entity keeps its colour in every chart.
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


def base_images() -> dict[str, np.ndarray]:
    return {"text": sc.render_text("DA321"),
            "line": (~skdata.horse()).astype(float),
            "photo": rgb2gray(skdata.chelsea())}


def trial(img: np.ndarray, seeds=SEEDS, refs: dict | None = None, **p) -> dict:
    r"""Encode and decode once per seed. Each `refs` entry also scores \hat{I} resized onto its grid."""
    dec = {k: v for k, v in p.items() if k not in ("n_iter", "init")}
    refs = refs or {}
    out = {"ssim": [], "psnr": [], "auc": []} | {f"{m}_{k}": [] for k in refs for m in ("ssim", "psnr", "auc")}
    for s in seeds:
        x, info = sc.encode(img, seed=s, **p)
        rec = sc.decode(x, **dec)
        out["ssim"].append(sc.ssim(info["target"], rec))
        out["psnr"].append(sc.psnr(info["target"], rec))
        out["auc"].append(sc.auc(info["target"], rec))
        for k, ref in refs.items():
            r2 = np.clip(resize(rec, ref.shape, anti_aliasing=True), 0, 1)
            out[f"ssim_{k}"].append(sc.ssim(ref, r2))
            out[f"psnr_{k}"].append(sc.psnr(ref, r2))
            out[f"auc_{k}"].append(sc.auc(ref, r2))
            if s == seeds[0]:
                out[f"rec_{k}"] = r2
        if s == seeds[0]:
            out.update(x=x, target=info["target"], rec=rec, errs=info["errs"], S=info["S"],
                       band=info["band"])
    return out


def show(a: plt.Axes, img: np.ndarray, title: str) -> None:
    a.imshow(img, cmap="magma", aspect="auto", vmin=0, vmax=1)
    a.set_title(title, pad=2)
    a.set(xticks=[], yticks=[])
    a.grid(False)
    a.spines[:].set_visible(False)


def row(exp: str, image: str, param: str, ssims, psnrs, aucs) -> list:
    return [exp, image, param, f"{np.mean(aucs):.4f}", f"{np.std(aucs):.4f}",
            f"{np.mean(ssims):.4f}", f"{np.std(ssims):.4f}", f"{np.mean(psnrs):.2f}"]


def save(name: str, **arrays) -> None:
    np.savez_compressed(DATA / f"{name}.npz", **{
        k: v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == float else v
        for k, v in arrays.items()})


def flat_stats(target: np.ndarray, rec: np.ndarray) -> dict:
    r"""What SSIM does over flat target regions: 7x7 windows with \sigma_T^2 < C_2.

    frac: share of such windows. var: mean \sigma_R^2. lum: mean luminance term l. ssim, rest: mean
    SSIM map on and off them. pred: mean of l C_2/(\sigma_R^2 + C_2), the flat-target approximation.
    """
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    _, smap = structural_similarity(target, rec, data_range=1.0, full=True)
    mean = lambda a: uniform_filter(a, 7)
    var = lambda a: (mean(a ** 2) - mean(a) ** 2) * 49 / 48  # sample variance, as in skimage
    mt, mr, vr = mean(target), mean(rec), var(rec)
    flat = var(target) < c2
    flat[:3], flat[-3:], flat[:, :3], flat[:, -3:] = False, False, False, False  # skimage crops the border
    body = np.zeros_like(flat)
    body[3:-3, 3:-3] = True
    lum = (2 * mt * mr + c1) / (mt ** 2 + mr ** 2 + c1)
    return {"flat_frac": flat.sum() / body.sum(), "flat_var": vr[flat].mean(), "flat_lum": lum[flat].mean(),
            "flat_ssim": smap[flat].mean(), "rest_ssim": smap[body & ~flat].mean(),
            "flat_pred": (lum * c2 / (vr + c2))[flat].mean()}


def diagnostics(target: np.ndarray, rec: np.ndarray) -> dict:
    r"""Separate what SSIM measures on a decoded image.

    support: mean SSIM map over the target's lit pixels dilated by 3 (excludes black background).
    ls_gain: SSIM after the least-squares gain g = <T,R>/<R,R>, undoing the max-normalisation bias.
    cv: \sigma/\mu of the decoded image on pixels with T > 0.5 (speckle strength).
    interior: mean SSIM map where the whole 7x7 SSIM window is lit, i.e. \sigma_T \approx 0. There
    SSIM \approx l \cdot C_2 / (\sigma_R^2 + C_2), so speckle is punished regardless of its size in
    the image; `interior_pred` evaluates that expression from the measured \mu_R and \sigma_R.
    NaN when no such window exists (thin edges).
    """
    _, smap = structural_similarity(target, rec, data_range=1.0, full=True)
    sup = binary_dilation(target > 0.01, iterations=3)
    lit = target > 0.5
    g = (rec * target).sum() / (rec * rec).sum()
    core = binary_erosion(target > 0.99, np.ones((7, 7)))
    interior, pred = np.nan, np.nan
    if core.any():
        interior = smap[core].mean()
        mu, c1, c2 = rec[core].mean(), 0.01 ** 2, 0.03 ** 2
        pred = (2 * mu + c1) / (1 + mu ** 2 + c1) * c2 / (rec[core].var() + c2)
    return {"support": smap[sup].mean(), "interior": interior, "interior_pred": pred, "gain": g,
            "ls_gain": sc.ssim(target, np.clip(g * rec, 0, 1)),
            "cv": rec[lit].std() / rec[lit].mean(), "lit_frac": lit.mean(),
            "support_frac": sup.mean()}


# 1 -----------------------------------------------------------------------------------------------
def edges_vs_filled(imgs: dict) -> list:
    rows, arrs = [], {}
    for n in NAMES:
        for mode in ("raw", "edges"):
            t = trial(sc.preprocess(imgs[n], mode))
            rows.append(row("edges_vs_filled", n, mode, t["ssim"], t["psnr"], t["auc"]))
            d = diagnostics(t["target"], t["rec"])
            for k in ("support", "ls_gain", "interior", "interior_pred"):  # seed 0 only: no std
                if np.isfinite(d[k]):
                    rows.append(["edges_vs_filled", n, f"{mode}/{k}", "", "", f"{d[k]:.4f}", "", ""])
            arrs |= {f"{n}_{mode}_target": t["target"], f"{n}_{mode}_rec": t["rec"],
                     f"{n}_{mode}_ssim": np.mean(t["ssim"]), f"{n}_{mode}_auc": np.mean(t["auc"]),
                     f"{n}_{mode}_energy": np.sum(t["x"] ** 2)}
            arrs |= {f"{n}_{mode}_{k}": v for k, v in d.items()}
    save("edges_vs_filled", **arrs)
    return rows


def plot_edges_vs_filled(d) -> plt.Figure:
    fig, ax = plt.subplots(3, 4, figsize=(7.16, 3.5), layout="constrained")
    for i, n in enumerate(NAMES):
        for j, mode in enumerate(("raw", "edges")):
            for a, key, title in ((ax[i, 2 * j], "target", f"{LABEL[n]}: {mode}, target"),
                                  (ax[i, 2 * j + 1], "rec", f"AUC {d[f'{n}_{mode}_auc']:.3f}, SSIM {d[f'{n}_{mode}_ssim']:.3f}")):
                show(a, d[f"{n}_{mode}_{key}"], title)
    return fig


# 2 -----------------------------------------------------------------------------------------------
LENGTHS = (128, 256, 512, 1024, 2048, 4096)


def sweep_window_len(imgs: dict) -> list:
    r"""Own-grid SSIM scores each N against a different target, so it is not comparable across N.

    The image-grid score compares every \hat{I} with the original image on one fixed grid. No N may
    keep its native grid: resampling smooths speckle, so an unresampled N would be penalised.
    'half' repeats this at half resolution to show that the ranking does not depend on the grid.
    """
    rows, arrs = [], {"lengths": np.array(LENGTHS)}
    for n in NAMES:
        img = sc.preprocess(imgs[n], MODE[n])
        full = resize(img, (260, 900), anti_aliasing=True)
        refs = {"image": full, "half": resize(img, (130, 450), anti_aliasing=True)}
        scores = {k: [] for k in ("own", "image", "half")}
        aucs = {k: [] for k in scores}
        for N in LENGTHS:
            t = trial(img, refs=refs, n_fft=N, hop=N // 4)
            for k in scores:
                sk, pk, ak = ("ssim", "psnr", "auc") if k == "own" else (f"ssim_{k}", f"psnr_{k}", f"auc_{k}")
                rows.append(row("sweep_window_len", n, f"N={N}/{k}", t[sk], t[pk], t[ak]))
                scores[k].append(t[sk])
                aucs[k].append(t[ak])
            arrs |= {f"{n}_{k}_{N}": v for k, v in flat_stats(full, t["rec_image"]).items()}
            if n == "text":
                arrs[f"text_rec_{N}"] = t["rec_image"]
        arrs |= {f"{n}_{k}": np.array(v) for k, v in scores.items()}
        arrs |= {f"{n}_{k}_auc": np.array(v) for k, v in aucs.items()}
    save("sweep_window_len", **arrs)
    return rows


def plot_sweep_window_len(d) -> plt.Figure:
    fig = plt.figure(figsize=(3.5, 3.5), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1.9])
    ax = top.subplots(1, 3)
    for a, N in zip(ax, (128, 1024, 4096)):
        show(a, d[f"text_rec_{N}"], f"$N={N}$")
    a = bot.subplots()
    L = d["lengths"]
    for n in NAMES:
        a.plot(L, d[f"{n}_image"].mean(1), color=COLOR[n], marker=MARK[n], label=LABEL[n])
    a.plot(L, d["text_own"].mean(1), color=COLOR["text"], marker=MARK["text"], ls=":", mfc="white",
           label="text, own-grid score")
    a.set_xscale("log", base=2)
    a.set_xticks(L, [str(v) for v in L])
    a.minorticks_off()
    a.set(xlabel="window length $N$ (hop $N/4$)", ylabel="SSIM")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    return fig


# 3 -----------------------------------------------------------------------------------------------
WINDOWS = ("rect", "hann", "hamming", "blackman")


def window_spectrum(name: str, N: int = 1024, pad: int = 32) -> tuple[np.ndarray, np.ndarray, float]:
    r"""|W(f)| in dB relative to its peak on bins \in [0, 12], and the highest sidelobe (dB)."""
    M = np.abs(np.fft.rfft(sc.window(name, N), N * pad))
    db = 20 * np.log10(np.maximum(M / M.max(), 1e-8))
    first_null = np.argmax(np.diff(db) > 0)  # end of the main lobe
    return np.arange(12 * pad) / pad, db[:12 * pad], float(db[first_null:].max())


def sweep_window_type(imgs: dict) -> list:
    rows, arrs = [], {}
    for w in WINDOWS:
        arrs[f"{w}_bins"], arrs[f"{w}_db"], arrs[f"{w}_sidelobe_db"] = window_spectrum(w)
        for n in NAMES:
            t = trial(sc.preprocess(imgs[n], MODE[n]), window=w)
            rows.append(row("sweep_window_type", n, w, t["ssim"], t["psnr"], t["auc"]))
            arrs[f"{n}_{w}_ssim"] = np.array(t["ssim"])
            arrs[f"{n}_{w}_auc"] = np.array(t["auc"])
            arrs |= {f"{n}_{w}_{k}": v for k, v in flat_stats(t["target"], t["rec"]).items()}
            if n == "text":
                arrs[f"text_{w}_rec"] = t["rec"]
    save("sweep_window_type", **arrs)
    return rows


def plot_sweep_window_type(d) -> plt.Figure:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.3), layout="constrained")
    for w, c, ls in zip(WINDOWS, ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6"), ("-", "-", "--", ":")):
        a.plot(d[f"{w}_bins"], d[f"{w}_db"], color=c, ls=ls, lw=1.1,
               label=f"{w} (sidelobe {float(d[f'{w}_sidelobe_db']):.0f} dB)")
    a.set(xlabel="offset from centre (DFT bins)", ylabel="$|W|$ (dB re peak)", ylim=(-100, 3), xlim=(0, 12))
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    x = np.arange(len(WINDOWS))
    for i, n in enumerate(NAMES):
        y = [d[f"{n}_{w}_ssim"].mean() for w in WINDOWS]
        b.plot(x + (i - 1) * 0.1, y, color=COLOR[n], marker=MARK[n], ls="", label=LABEL[n])
    b.set_xticks(x, WINDOWS)
    b.set(ylabel="SSIM", ylim=(0, 1), xlim=(-0.4, 3.4))
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    return fig


# 4 -----------------------------------------------------------------------------------------------
ITERS = (0, 1, 2, 5, 10, 20, 60, 100, 200)
CURVE_LEN = 201  # errs[i] is the error of X^(i), so i = 0..200 needs 201 passes


def gl_iterations(imgs: dict) -> list:
    r"""n_iter=0 is random phase only: x = ISTFT(S e^{j\phi}), with no projection step."""
    rows, arrs = [], {"iters": np.array(ITERS)}
    for n in NAMES:
        img = sc.preprocess(imgs[n], MODE[n])
        arrs[f"{n}_errs_random"] = np.array(
            [trial(img, seeds=(s,), n_iter=CURVE_LEN)["errs"] for s in SEEDS])
        arrs[f"{n}_errs_zero"] = np.array(trial(img, seeds=(0,), n_iter=CURVE_LEN, init="zero")["errs"])
        ssim_r, ssim_z, auc_r = [], [], []
        for k in ITERS:
            r = trial(img, n_iter=k)
            z = trial(img, seeds=(0,), n_iter=k, init="zero")
            rows.append(row("gl_iterations", n, f"random/iter={k}", r["ssim"], r["psnr"], r["auc"]))
            rows.append(row("gl_iterations", n, f"zero/iter={k}", z["ssim"], z["psnr"], z["auc"]))
            ssim_r.append(r["ssim"])
            auc_r.append(r["auc"])
            ssim_z.append(z["ssim"][0])
        arrs |= {f"{n}_ssim_random": np.array(ssim_r), f"{n}_ssim_zero": np.array(ssim_z),
                 f"{n}_auc_random": np.array(auc_r)}
    save("gl_iterations", **arrs)
    return rows


def plot_gl_iterations(d) -> plt.Figure:
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=True, layout="constrained")
    it = np.arange(CURVE_LEN)
    for n in NAMES:
        a.plot(it + 1, d[f"{n}_errs_random"].mean(0), color=COLOR[n], label=f"{LABEL[n]}, random")
        a.plot(it + 1, d[f"{n}_errs_zero"], color=COLOR[n], ls=":", label=f"{LABEL[n]}, zero")
        b.plot(d["iters"] + 1, d[f"{n}_ssim_random"].mean(1), color=COLOR[n], marker=MARK[n])
        b.plot(d["iters"] + 1, d[f"{n}_ssim_zero"], color=COLOR[n], marker=MARK[n], ls=":", mfc="white")
    a.set_xscale("log")
    ticks = [1, 2, 3, 6, 11, 21, 61, 101, 201]
    b.set_xticks(ticks, [str(t - 1) for t in ticks])
    b.minorticks_off()
    a.set(ylabel="spectral convergence")
    b.set(xlabel="Griffin-Lim iteration", ylabel="SSIM")
    a.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.28), handlelength=1.6, columnspacing=0.9)
    return fig


# 5 -----------------------------------------------------------------------------------------------
def mel_inv(m: np.ndarray) -> np.ndarray:
    r"""f(m) = 700 (10^{m/2595} - 1)."""
    return 700 * (10 ** (m / 2595) - 1)


def view(rec: np.ndarray, scale: str, n_fft: int = 1024) -> np.ndarray:
    r"""Redraw a bins-by-frames image on a viewer axis: row r \in [0,1] shows frequency f(r)."""
    freqs = np.arange(n_fft // 2 + 1) * SR / n_fft
    fb = freqs[(freqs >= F_LO) & (freqs <= F_HI)]
    r = np.linspace(0, 1, rec.shape[0])
    if scale == "mel":
        lo, hi = sc.freq_map.mel(np.array([F_LO, F_HI], float))
        f = mel_inv(lo + r * (hi - lo))
    else:
        f = F_LO + r * (F_HI - F_LO)
    y = rec[::-1]
    return np.clip(interp1d(fb, y, axis=0, bounds_error=False, fill_value=(y[0], y[-1]))(f)[::-1], 0, 1)


def mel_vs_linear(imgs: dict) -> list:
    r"""Encode on one axis and view on either; score against the undistorted image on the viewer grid."""
    rows, arrs = [], {}
    for n in NAMES:
        img = sc.preprocess(imgs[n], MODE[n])
        for enc in ("linear", "mel"):
            t = trial(img, scale=enc)
            ref = resize(img, t["target"].shape, anti_aliasing=True)
            rows.append(row("mel_vs_linear", n, f"encode={enc}/bin_grid", t["ssim"], t["psnr"], t["auc"]))
            for vw in ("linear", "mel"):
                v = view(t["rec"], vw)
                s, p, u = sc.ssim(ref, v), sc.psnr(ref, v), sc.auc(ref, v)
                rows.append(["mel_vs_linear", n, f"encode={enc}/view={vw}", f"{u:.4f}", "", f"{s:.4f}", "", f"{p:.2f}"])
                arrs |= {f"{n}_{enc}_{vw}_img": v, f"{n}_{enc}_{vw}_ssim": s, f"{n}_{enc}_{vw}_auc": u}
    save("mel_vs_linear", **arrs)
    return rows


def plot_mel_vs_linear(d) -> plt.Figure:
    fig, ax = plt.subplots(2, 2, figsize=(3.5, 2.2), layout="constrained")
    for i, enc in enumerate(("linear", "mel")):
        for j, vw in enumerate(("linear", "mel")):
            show(ax[i, j], d[f"text_{enc}_{vw}_img"],
                 f"{enc} $\\to$ {vw}: AUC {float(d[f'text_{enc}_{vw}_auc']):.3f}")
    fig.suptitle("encoding axis $\\to$ viewing axis", fontsize=8)
    return fig


# 6 -----------------------------------------------------------------------------------------------
SNRS = (30, 20, 10, 5, 0)


def noisy_scores(x: np.ndarray, target: np.ndarray, sigma: float) -> tuple[list, list, list]:
    ss, ps, us = [], [], []
    for s in SEEDS:
        y = x + sigma * np.random.default_rng(100 + s).standard_normal(x.size)
        rec = sc.decode(y)
        ss.append(sc.ssim(target, rec))
        ps.append(sc.psnr(target, rec))
        us.append(sc.auc(target, rec))
    return ss, ps, us


def noise_robustness(imgs: dict) -> list:
    r"""SNR = 10\log_{10}(\sum x^2 / \sum n^2).

    Relative noise holds SNR fixed per signal, i.e. fixed signal energy. A loudspeaker instead fixes
    peak amplitude and the room fixes the noise floor, so for text the 'floor' series gives raw and
    edges the same absolute \sigma, chosen so that raw sits at the nominal SNR.
    """
    rows, arrs = [], {"snrs": np.array(SNRS)}
    sig = lambda x, snr: np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))
    enc = {}
    for n in NAMES:
        for mode in ("raw", "edges"):
            enc[n, mode] = sc.encode(sc.preprocess(imgs[n], mode))
    for (n, mode), (x, info) in enc.items():
        res = [noisy_scores(x, info["target"], sig(x, s)) for s in SNRS]
        for s, (ss, ps, us) in zip(SNRS, res):
            rows.append(row("noise_robustness", n, f"{mode}/snr={s}", ss, ps, us))
        arrs[f"{n}_{mode}_ssim"] = np.array([r[0] for r in res])
        arrs[f"{n}_{mode}_auc"] = np.array([r[2] for r in res])
    x, info = enc["text", "edges"]
    for s in SNRS:
        y = x + sig(x, s) * np.random.default_rng(100).standard_normal(x.size)
        arrs[f"text_edges_rec_{s}"] = sc.decode(y)
    x_raw = enc["text", "raw"][0]
    for mode in ("raw", "edges"):
        x, info = enc["text", mode]
        res = [noisy_scores(x, info["target"], sig(x_raw, s)) for s in SNRS]
        for s, (ss, ps, us) in zip(SNRS, res):
            rows.append(row("noise_robustness", "text", f"{mode}/floor={s}", ss, ps, us))
        arrs[f"text_{mode}_floor_ssim"] = np.array([r[0] for r in res])
        arrs[f"text_{mode}_floor_auc"] = np.array([r[2] for r in res])
    save("noise_robustness", **arrs)
    return rows


def plot_noise_robustness(d) -> plt.Figure:
    fig = plt.figure(figsize=(3.5, 4.7), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 4.2])
    for a, snr in zip(top.subplots(1, 3), (20, 10, 0)):
        show(a, d[f"text_edges_rec_{snr}"], f"{snr} dB")
    a, b = bot.subplots(2, 1, sharex=True)
    s = d["snrs"]
    for n in NAMES:
        y = d[f"{n}_{MODE[n]}_ssim"]
        a.errorbar(s, y.mean(1), y.std(1), color=COLOR[n], marker=MARK[n], capsize=2, label=LABEL[n])
    a.set(ylabel="SSIM", ylim=(0, 1))
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    for key, ls, tag in (("ssim", "-", "same SNR"), ("floor_ssim", ":", "same noise floor")):
        b.plot(s, d[f"text_raw_{key}"].mean(1), color=MUTED, marker="D", ls=ls, mfc="white", label=f"raw, {tag}")
        b.plot(s, d[f"text_edges_{key}"].mean(1), color=COLOR["text"], marker="o", ls=ls, label=f"edges, {tag}")
    b.set(xlabel="SNR (dB)", ylabel="SSIM", ylim=(0, 1))
    b.invert_xaxis()
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    return fig


EXPERIMENTS = {f.__name__: (f, globals()[f"plot_{f.__name__}"]) for f in (
    edges_vs_filled, sweep_window_len, sweep_window_type, gl_iterations, mel_vs_linear,
    noise_robustness)}
HEADER = ["experiment", "image", "param", "auc", "auc_std", "ssim", "ssim_std", "psnr_db"]


def update_table(new: dict[str, list]) -> None:
    """Replace the rows of the experiments just run, keeping the file in experiment order."""
    old = []
    if TABLE.exists():
        with open(TABLE, newline="") as f:
            old = [r for r in list(csv.reader(f))[1:] if r[0] not in new]
    rows = old + [r for rs in new.values() for r in rs]
    rows.sort(key=lambda r: list(EXPERIMENTS).index(r[0]))  # stable: keeps param order
    with open(TABLE, "w", newline="") as f:
        csv.writer(f).writerows([HEADER] + rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", choices=[[], *EXPERIMENTS], help="default: all")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    for d in (FIG, DATA, RES):
        d.mkdir(exist_ok=True)
    names = a.names or list(EXPERIMENTS)
    imgs = base_images()
    new = {}
    for name in names:
        run, plot = EXPERIMENTS[name]
        if not a.plot_only:
            t0 = time.time()
            new[name] = run(imgs)
            print(f"{name}: {len(new[name])} rows in {time.time() - t0:.0f} s", flush=True)
        fig = plot(np.load(DATA / f"{name}.npz"))
        fig.savefig(FIG / f"{name}.pdf")
        fig.savefig(FIG / f"{name}.png")
        plt.close(fig)
    if new:
        update_table(new)


if __name__ == "__main__":
    main()
