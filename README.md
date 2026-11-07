# SonoCanvas: Drawing with Sound

Encodes an image or typed word into audio, so that the audio's spectrogram shows the picture.
Individual project for DA321 Multimodal Data Processing, IIT Guwahati.

Encoder: image preparation, image-to-magnitude map, Griffin-Lim phase recovery, weighted overlap-add.
Decoder: STFT, band crop, recovered image, stroke/background AUC, SSIM and PSNR. The STFT, ISTFT and Griffin-Lim are written
from scratch in NumPy.

## Setup

Python 3.13 is recommended.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Manim needs cairo, pango and ffmpeg, and `MathTex` needs `latex` and `dvisvgm` on `PATH`. The paper
needs [tectonic](https://tectonic-typesetting.github.io).

## Usage

```sh
python -m sonocanvas "HELLO" -o hello.wav --png hello.png   # WAV plus target-vs-decoded image
python -m pytest -q                                         # tests
python experiments.py                                       # all six experiments
python experiments.py --plot-only                           # redraw figures from data/
streamlit run app.py                                        # live demo
python paper/make_numbers.py && (cd paper && tectonic main.tex)   # paper
PYTHONPATH=. python video/make_assets.py                    # real waveforms for the video
python video/timing.py                                      # scene times from the narration
video/render.sh m                                           # video at 720p (h for 1080p)
video/check_scene.sh <clip.mp4> <sheet.png>                 # contact sheet and holds over 3 s
python analyze_recording.py take.m4a --word DA321 --dist 2 --device phone   # score a recording
```

To view the result on a phone, use a spectrogram app with a **linear** frequency axis reaching at
least 8 kHz. `demo/` holds three pre-rendered WAVs.

## Layout

| Path | Contents |
|---|---|
| `sonocanvas/` | `stft.py`, `phase.py`, `freq_map.py`, `image_prep.py`, `metrics.py`, CLI |
| `tests/` | STFT round trip, COLA, tone bin, end-to-end SSIM, AUC, long text, app |
| `experiments.py` | Six experiments, writing `data/*.npz`, `figures/`, `results/table.csv` |
| `app.py` | Streamlit demo |
| `paper/` | IEEE paper; `make_numbers.py` generates every result macro and Table I |
| `video/` | Eight Manim scenes, `narration.md` (sets the timing), `render.sh` |
| `analyze_recording.py` | Aligns, decodes and scores a real recording; procedure in `ACOUSTIC_TEST.md` |
| `prototype/` | Original single-file prototype, kept for reference |

## Results

All results are in [`results/table.csv`](results/table.csv), and the paper states what they mean.
AUC is the legibility score and SSIM is reported beside it. SSIM understates legibility under noise
on sparse images. AUC saturates on clean text, so the figures plot 1 - AUC on a log axis.

Test images from scikit-image: horse silhouette and Chelsea the cat, both CC0.
