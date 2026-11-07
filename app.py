"""SonoCanvas live demo: type a word or upload an image, then hear and see it."""
import io

import numpy as np
import streamlit as st
from scipy.io import wavfile

import sonocanvas as sc
from sonocanvas.__main__ import figure

PRESETS = ("DA321", "HELLO", "IITG")


@st.cache_data(show_spinner="Running Griffin-Lim...")
def run(img: np.ndarray) -> dict:
    x, info = sc.encode(img)
    rec = sc.decode(x)
    buf = io.BytesIO()
    wavfile.write(buf, info["sr"], (x * 32767).astype(np.int16))
    return {"wav": buf.getvalue(), "target": info["target"], "rec": rec, "info": info,
            "ssim": sc.ssim(info["target"], rec), "psnr": sc.psnr(info["target"], rec)}


st.set_page_config(page_title="SonoCanvas", layout="centered")
st.title("SonoCanvas")
st.caption("Drawing with sound: the picture is hidden in the audio's spectrogram.")

st.session_state.setdefault("text", PRESETS[0])
for col, word in zip(st.columns(len(PRESETS)), PRESETS):
    if col.button(word, use_container_width=True):
        st.session_state.text = word
text = st.text_input("Word", key="text", max_chars=24)
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

r = run(img)
st.audio(r["wav"], format="audio/wav")
st.download_button("Download WAV", r["wav"], file_name="sonocanvas.wav", mime="audio/wav")
c1, c2, c3 = st.columns(3)
c1.metric("SSIM", f"{r['ssim']:.3f}")
c2.metric("PSNR", f"{r['psnr']:.1f} dB")
c3.metric("Spectral convergence", f"{r['info']['errs'][-1]:.3f}")
st.pyplot(figure(r["target"], r["rec"], r["info"], r["ssim"], r["psnr"]))
st.caption("Phone app settings: linear frequency axis, range up to at least 8 kHz.")
