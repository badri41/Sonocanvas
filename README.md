# SonoCanvas: Drawing with Sound

Encodes an image or typed word into audio, so that the audio's spectrogram shows the picture.
Individual project for DA321 Multimodal Data Processing, IIT Guwahati.

Encoder: image preparation, image-to-magnitude map, Griffin-Lim phase recovery, weighted overlap-add.
Decoder: STFT, band crop, recovered image, SSIM and PSNR. The STFT, ISTFT and Griffin-Lim are written
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
video/render.sh m                                           # video at 720p (h for 1080p)
```

To view the result on a phone, use a spectrogram app with a **linear** frequency axis reaching at
least 8 kHz. `demo/` holds three pre-rendered WAVs.

## Layout

| Path | Contents |
|---|---|
| `sonocanvas/` | `stft.py`, `phase.py`, `freq_map.py`, `image_prep.py`, `metrics.py`, CLI |
| `tests/` | STFT round trip, COLA, tone bin, end-to-end SSIM, long text, app |
| `experiments.py` | Six experiments, writing `data/*.npz`, `figures/`, `results/table.csv` |
| `app.py` | Streamlit demo |
| `paper/` | IEEE paper; `make_numbers.py` generates every result macro and Table I |
| `video/` | Eight Manim scenes, `narration.md`, `render.sh` |
| `prototype/` | Original single-file prototype, kept for reference |

## Results

All results are in [`results/table.csv`](results/table.csv), and the paper states what they mean.
SSIM here should be read with care: on sparse images it is dominated by black background and it
understates legibility under noise. The paper's discussion section quantifies both effects.

Test images from scikit-image: horse silhouette and Chelsea the cat, both CC0.
