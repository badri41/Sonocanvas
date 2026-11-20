"""SonoCanvas live demo: type a word or upload an image, then hear and see it.

Presentation mode (toggle, or open with ?present=1): three large presets, large text, and a fallback
that plays the pre-rendered demo/<WORD>.wav if live encoding takes longer than FALLBACK_S.
"""
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import streamlit as st
from scipy.io import wavfile

import sonocanvas as sc
from sonocanvas.__main__ import figure

PRESETS = ("DA321", "HELLO", "IITG")
DEMO = Path(__file__).parent / "demo"
FALLBACK_S = float(os.environ.get("SONOCANVAS_FALLBACK_S", 4.0))  # encoding takes about 2.4 s here


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, sr, (x * 32767).astype(np.int16))
    return buf.getvalue()


def scores(target: np.ndarray, rec: np.ndarray) -> dict:
    return {"rec": rec, "auc": sc.auc(target, rec), "ssim": sc.ssim(target, rec), "psnr": sc.psnr(target, rec)}


def encode(img: np.ndarray) -> dict:
    x, info = sc.encode(img)
    return {"wav": wav_bytes(x, info["sr"]), "target": info["target"], "info": info, "sc": info["errs"][-1],
            "source": "encoded live", **scores(info["target"], sc.decode(x))}


@st.cache_data(show_spinner="Running Griffin-Lim...")
def run(img: np.ndarray) -> dict:
    return encode(img)


@st.cache_resource
def pool() -> tuple[ThreadPoolExecutor, dict]:
    """One worker and its jobs, kept across reruns so a slow encode finishes in the background."""
    return ThreadPoolExecutor(max_workers=1), {}


def prerendered(word: str, img: np.ndarray) -> dict:
    """The shipped WAV, decoded now (decoding takes milliseconds). It is bit-identical to a fresh encode."""
    sr, y = wavfile.read(DEMO / f"{word}.wav")
    x = y / 32767
    S, band = sc.image_to_mag(img, 1024, 256, 5.0, 1000, 8000, sr, "linear")
    target = S[band][::-1] / S[band].max()
    return {"wav": (DEMO / f"{word}.wav").read_bytes(), "target": target, "info": {"errs": [np.nan]},
            "sc": None, "source": f"pre-rendered demo/{word}.wav", **scores(target, sc.decode(x))}


def run_presenting(key: str, word: str, img: np.ndarray) -> dict:
    ex, jobs = pool()
    job = jobs.get(key) or jobs.setdefault(key, ex.submit(encode, img))
    try:
        return job.result(timeout=FALLBACK_S)
    except TimeoutError:
        if (DEMO / f"{word}.wav").exists():
            return prerendered(word, img)
        with st.spinner("Running Griffin-Lim..."):
            return job.result()


st.set_page_config(page_title="SonoCanvas", layout="centered")
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
text = st.text_input("Word", key="text", max_chars=24)
if present:
    upload, mode = None, "edges"
else:
    upload = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    mode = st.radio("Preprocessing", ("edges", "raw"), horizontal=True,
                    help="Edges survive best. Raw keeps filled shapes, with visible speckle.")

if upload is None and not text.strip():
    st.info("Type a word or upload an image.")
    st.stop()

base = sc.load(upload) if upload is not None else sc.render_text(text)
img = sc.preprocess(base, mode=mode)
if img.max() == 0:
    st.warning("This image has no content to encode. Try the other preprocessing mode.")
    st.stop()

r = run_presenting(f"{text}|{mode}", text.strip(), img) if present else run(img)
st.audio(r["wav"], format="audio/wav")
st.download_button("Download WAV", r["wav"], file_name="sonocanvas.wav", mime="audio/wav")
c1, c2, c3 = st.columns(3)
c1.metric("AUC (legibility)", f"{r['auc']:.3f}")
c2.metric("SSIM", f"{r['ssim']:.3f}")
c3.metric("PSNR", f"{r['psnr']:.1f} dB")
fig = figure(r["target"], r["rec"], r["info"], r["ssim"], r["psnr"])
fig.axes[1].set_title(f"decoded from audio: AUC {r['auc']:.3f}, SSIM {r['ssim']:.3f}")
st.pyplot(fig)
sc_txt = f", spectral convergence {r['sc']:.3f}" if r["sc"] is not None else ""
st.caption(f"Source: {r['source']}{sc_txt}. Phone app settings: linear frequency axis, range up to at least 8 kHz.")
