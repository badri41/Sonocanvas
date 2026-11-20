# SonoCanvas: Drawing with Sound

Encodes an image or typed word into audio, so that the audio's spectrogram shows the picture.
Individual project for DA321 Multimodal Data Processing, IIT Guwahati.

Encoder: image preparation, image-to-magnitude map, Griffin-Lim phase recovery, weighted overlap-add.
Decoder: STFT, band crop, recovered image, stroke/background AUC, SSIM and PSNR. The STFT, ISTFT and Griffin-Lim are written
from scratch in NumPy.

## Setup (one command)

```sh
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m pytest -q
```

That creates the environment and runs the tests. Activate it with `source .venv/bin/activate` for the
commands below. Rendering the video also needs cairo, pango and ffmpeg, and `MathTex` needs `latex` and
`dvisvgm` (TinyTeX; `video/render.sh` adds it to `PATH`). The paper needs
[tectonic](https://tectonic-typesetting.github.io), which downloads its TeX bundle on first use.

## Run

| What | Command | Notes |
|---|---|---|
| Tests | `python -m pytest -q` | core, metrics and app |
| Encode a word | `python -m sonocanvas "HELLO" -o hello.wav --png hello.png` | WAV plus target-vs-decoded image |
| Experiments | `python experiments.py` | six experiments; `gl_iterations` takes hours |
| Redraw figures | `python experiments.py --plot-only` | from `data/*.npz` |
| Paper | `python paper/make_numbers.py && (cd paper && tectonic main.tex)` | also refreshes the table below |
| Video | `video/render.sh h` | 1080p60 (`m` for 720p30); writes `video/sonocanvas_1080p60.mp4` |
| Scene timing | `python video/timing.py` | scene lengths from `video/narration.md` |
| Check a clip | `video/check_scene.sh clip.mp4 sheet.png` | contact sheet and holds over 3 s |
| Demo | `streamlit run app.py` | add `?present=1` to the URL for presentation mode |
| Score a recording | `python analyze_recording.py take.m4a --word DA321 --dist 2 --device phone` | see `ACOUSTIC_TEST.md` |

The demo runs without internet. Presentation mode shows the three presets in large text, and if live
encoding takes more than 4 s it plays the matching pre-rendered `demo/*.wav`, which is bit-identical to a
fresh encode. To view any WAV on a phone, use a spectrogram app with a **linear** frequency axis reaching at
least 8 kHz.

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
| `demo/` | Pre-rendered WAVs for the demo presets |
| `prototype/` | Original single-file prototype, kept for reference |

## Results

Headline rows of [`results/table.csv`](results/table.csv), written here by `paper/make_numbers.py`
(means over five seeds). Text and line use edge preprocessing and the photograph uses raw, except in the
first two rows. AUC is the legibility score and SSIM is reported beside it: SSIM understates legibility
under noise on sparse images, and AUC saturates on clean text.

<!-- results -->
| Configuration | Text AUC | Text SSIM | Line AUC | Line SSIM | Photo SSIM |
|---|---|---|---|---|---|
| Filled / raw | 1.0000 | 0.871 | 1.0000 | 0.698 | 0.377 |
| Sobel edges | 0.9998 | 0.952 | 0.9997 | 0.948 | 0.897 |
| N=128, image grid | 0.9571 | 0.805 | 0.9547 | 0.764 | 0.521 |
| N=1024, image grid | 0.9999 | 0.915 | 0.9995 | 0.937 | 0.510 |
| N=4096, image grid | 0.9893 | 0.759 | 0.9976 | 0.831 | 0.638 |
| Rectangular window | 0.9972 | 0.730 | 0.9998 | 0.747 | 0.513 |
| Random phase only | 0.9899 | 0.850 | 0.9918 | 0.856 | 0.114 |
| 200 iterations | 0.9999 | 0.955 | 0.9998 | 0.951 | 0.417 |
| Linear encode, mel view | 0.7839 | 0.705 | 0.7147 | 0.666 | 0.112 |
| 30 dB SNR | 0.9998 | 0.872 | 0.9998 | 0.845 | 0.377 |
| 10 dB SNR | 0.9998 | 0.212 | 0.9997 | 0.187 | 0.260 |
| 0 dB SNR | 0.9977 | 0.129 | 0.9957 | 0.114 | 0.100 |
<!-- /results -->

Test images from scikit-image: horse silhouette and Chelsea the cat, both CC0.
