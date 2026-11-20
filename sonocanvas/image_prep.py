"""Image loading, text rendering and point/edge preprocessing."""
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, sobel

# DejaVu Sans Bold ships with matplotlib, so the rendered text is identical on every machine.
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"


def load(path: str) -> np.ndarray:
    r"""Grayscale image as float in [0, 1]: I = L / 255 (Pillow 'L' = ITU-R 601 luma)."""
    return np.asarray(Image.open(path).convert("L"), float) / 255.0


def render_text(text: str, w: int = 900, h: int = 260, fit: bool = True) -> np.ndarray:
    r"""White bold text (DejaVu Sans Bold, 170 px) centred on a black w x h canvas, in [0, 1].

    Text wider than 0.92 w is shrunk to fit (fit=True), or the canvas is widened to
    w' = \lceil (b_2 - b_0)/0.92 \rceil so the font keeps its size (fit=False; encode it with
    dur = 5 w'/900 s to keep 900 px per 5 s, see text_duration).
    With glyph bbox (b_0, b_1, b_2, b_3): origin = ((w - (b_2 - b_0))/2 - b_0, (h - (b_3 - b_1))/2 - b_1).
    """
    f = ImageFont.truetype(str(FONT), 170)  # raises if missing; no silent fallback
    b0, b1, b2, b3 = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=f)
    if b2 - b0 > 0.92 * w:
        if fit:  # shrink long text to fit
            f = ImageFont.truetype(str(FONT), int(170 * 0.92 * w / (b2 - b0)))
            b0, b1, b2, b3 = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=f)
        else:  # widen the canvas instead
            w = int(np.ceil((b2 - b0) / 0.92))
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    d.text(((w - (b2 - b0)) / 2 - b0, (h - (b3 - b1)) / 2 - b1), text, fill=255, font=f)
    return np.asarray(img, float) / 255.0


def text_duration(img: np.ndarray, w: int = 900, dur: float = 5.0) -> float:
    r"""Clip length that keeps a widened text canvas at the default time scale: dur \cdot w'/w seconds."""
    return dur * max(1.0, img.shape[1] / w)


def preprocess(img: np.ndarray, mode: str = "raw", gamma: float = 1.0,
               dilate: int = 1) -> np.ndarray:
    r"""Point processing, then optional edge extraction.

    Contrast stretch I' = (I - \min I)/(\max I - \min I), gamma I'' = I'^\gamma.
    mode='edges': G = \sqrt{(S_x * I'')^2 + (S_y * I'')^2} (Sobel), E = [G > 0.25 \max G],
    then E \oplus B applied `dilate` times (3x3 cross B). Returns float in [0, 1].
    """
    lo, hi = img.min(), img.max()
    out = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img, float)
    out = out ** gamma
    if mode == "raw":
        return out
    if mode != "edges":
        raise ValueError(f"unknown mode {mode!r}")
    g = np.hypot(sobel(out, 0), sobel(out, 1))
    e = g > 0.25 * g.max()
    if dilate > 0:  # scipy treats iterations < 1 as "dilate until nothing changes"
        e = binary_dilation(e, iterations=dilate)
    return e.astype(float)
