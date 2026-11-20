"""
This handles a simple demo website for the project

All the work happens in the C++ program bin/sonocanvas (build it with `make`). This page only runs it,
plays the WAV it writes, and shows the answer-key PNG and the scores it reports.
Presentation mode (toggle, or open with ?present=1): three large presets, large text, and a fallback
that decodes the pre-rendered demo/<WORD>.wav if encoding takes longer than FALLBACK_S.
"""
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent
BIN = ROOT / "bin" / "sonocanvas"
DEMO = ROOT / "demo"
PRESETS = ("DA321", "HELLO", "IITG")
FALLBACK_S = float(os.environ.get("SONOCANVAS_FALLBACK_S", 4.0))


def sonocanvas(*args: str) -> dict:
    """Run bin/sonocanvas encode|decode with --png and --npz; return the WAV, the PNG and the scores."""
    with tempfile.TemporaryDirectory() as tmp:
        png, npz = f"{tmp}/key.png", f"{tmp}/scores.npz"
        cmd = [str(BIN), *args, "--png", png, "--npz", npz]
        if args[0] == "encode":
            cmd += ["-o", f"{tmp}/out.wav"]
        subprocess.run(cmd, check=True, capture_output=True)
        wav = Path(f"{tmp}/out.wav" if args[0] == "encode" else args[1]).read_bytes()
        scores = {k: float(v) for k, v in np.load(npz).items() if v.ndim == 0}
        return {"wav": wav, "png": Path(png).read_bytes(), **scores}


@st.cache_data(show_spinner="Running Griffin-Lim...")
def encode(text: str, mode: str, fixed: bool, image: bytes | None) -> dict:
    args = ["encode", text, "--mode", mode] + (["--fixed-font"] if fixed else [])
    if image is None:
        return sonocanvas(*args) | {"source": "encoded live"}
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(image)
        f.flush()
        return sonocanvas(*args, "--image", f.name) | {"source": "encoded live"}


def prerendered(word: str) -> dict:
    """The shipped demo/WORD.wav, decoded and scored now (decoding takes milliseconds)."""
    return sonocanvas("decode", str(DEMO / f"{word}.wav"), word) | {"source": f"pre-rendered demo/{word}.wav", "sc": -1}


@st.cache_resource
def pool() -> tuple[ThreadPoolExecutor, dict]:
    """One worker and its jobs, kept across reruns so a slow encode finishes in the background."""
    return ThreadPoolExecutor(max_workers=1), {}


def encode_presenting(word: str) -> dict:
    ex, jobs = pool()
    job = jobs.get(word) or jobs.setdefault(word, ex.submit(sonocanvas, "encode", word, "--mode", "edges"))
    try:
        return job.result(timeout=FALLBACK_S) | {"source": "encoded live"}
    except TimeoutError:
        if (DEMO / f"{word}.wav").exists():
            return prerendered(word)
        with st.spinner("Running Griffin-Lim..."):
            return job.result() | {"source": "encoded live"}


st.set_page_config(page_title="SonoCanvas", layout="centered")
if not BIN.exists():
    st.error("bin/sonocanvas is missing: run `make` in the project folder first.")
    st.stop()
present = st.toggle("Presentation mode", value=st.query_params.get("present") == "1")
if present:
    st.markdown("""<style>
    html, body, [class*="st-"], p, label, input { font-size: 1.45rem !important; }
    h1 { font-size: 3.6rem !important; }
    div.stButton > button { font-size: 2.2rem !important; padding: 0.6rem 0 !important; }
    [data-testid="stMetricValue"] { font-size: 3.2rem !important; }
    [data-testid="stMetricLabel"] p { font-size: 1.5rem !important; }
    </style>""", unsafe_allow_html=True)

st.title("SonoCanvas")
st.caption("Drawing with sound: the picture is hidden in the audio's spectrogram.")

st.session_state.setdefault("text", PRESETS[0])
for col, word in zip(st.columns(len(PRESETS)), PRESETS):
    if col.button(word, use_container_width=True):
        st.session_state.text = word
text = st.text_input("Word", key="text", max_chars=24).strip()
if present:
    upload, mode, fixed = None, "edges", False
else:
    fixed = st.toggle("Keep font size (longer audio)",
                      help="Long text normally shrinks to fit 5 s. On, the font stays full size and the clip grows.")
    upload = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    mode = st.radio("Preprocessing", ("edges", "raw"), horizontal=True,
                    help="Edges survive best. Raw keeps filled shapes, with visible speckle.")

if upload is None and not text:
    st.info("Type a word or upload an image.")
    st.stop()

try:
    if present:
        r = encode_presenting(text)
    else:
        r = encode(text, mode, fixed, upload.getvalue() if upload is not None else None)
except subprocess.CalledProcessError as e:
    st.warning(f"This input could not be encoded: {e.stderr.decode().strip()}.")
    st.stop()

st.audio(r["wav"], format="audio/wav")
st.download_button("Download WAV", r["wav"], file_name="sonocanvas.wav", mime="audio/wav")
c1, c2, c3 = st.columns(3)
c1.metric("AUC (legibility)", f"{r['auc']:.3f}")
c2.metric("SSIM", f"{r['ssim']:.3f}")
c3.metric("PSNR", f"{r['psnr']:.1f} dB")
st.image(r["png"], caption="target (top) and the picture decoded from the audio (bottom)")
sc_txt = f", spectral convergence {r['sc']:.3f}" if r["sc"] >= 0 else ""
st.caption(f"Source: {r['source']}{sc_txt}. Clip length {r['dur']:.1f} s. Phone app settings: linear frequency axis, "
           f"range up to at least 8 kHz, time window of at least {r['dur']:.0f} s.")
