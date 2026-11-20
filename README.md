# SONOCANVAS(1)

Drawing with Sound. Individual project for DA321 Multimodal Data Processing, IIT Guwahati.

## NAME

**sonocanvas**: encode an image or typed text into audio whose spectrogram shows the picture, and decode
it back.

## SYNOPSIS

```
python -m sonocanvas TEXT -o OUT.wav [--png OUT.png] [--mode edges|raw] [--fixed-font]
python experiments.py [NAME ...] [--plot-only]
python analyze_recording.py RECORDING --dist METRES [--word WORD] [--device NAME] [--take N] [--video]
python paper/make_numbers.py
streamlit run app.py                        # http://localhost:8501/?present=1 for presentation mode
video/render.sh [m|h]
python video/add_voice.py [--prepare]
./make_final.sh
python -m pytest -q
```

```python
import sonocanvas as sc
x, info = sc.encode(sc.preprocess(sc.render_text("HELLO"), mode="edges"))
rec = sc.decode(x)
sc.auc(info["target"], rec), sc.ssim(info["target"], rec)
```

## DESCRIPTION

An image $I$ is mapped onto a target STFT magnitude $S$. The time axis becomes STFT frames, and the
vertical axis becomes the bins between 1 and 8 kHz, with the top row at 8 kHz. The phase that makes
$S$ a valid spectrogram is unknown, so Griffin–Lim estimates it. The waveform is then synthesised by
weighted overlap-add. Decoding is the analysis STFT, cropped to the same band.

$$I \;\xrightarrow{\text{image\_to\_mag}}\; S \;\xrightarrow{\text{griffin\_lim}}\; x
\;\xrightarrow{\text{stft, crop}}\; \hat I$$

The STFT, ISTFT and Griffin–Lim are written from scratch in NumPy, without librosa.

**Defaults:** $f_s = 22050$ Hz, Hann window, $N = 1024$, hop $H = 256$, band 1000–8000 Hz, clip length
$d = 5$ s, 60 Griffin–Lim iterations.

With these defaults:
- the band is bins 47–371 (1012–7989 Hz), which is 325 bins;
- a 5 s clip has $M = 1 + \lfloor d f_s \rfloor / H = 431$ frames;
- the decoded image is 325 × 431;
- the WAV holds $H(M-1) = 110080$ samples (4.99 s), peak-normalised to 0.9.

**Text canvas.** `render_text` draws DejaVu Sans Bold at 170 px, centred on a 900 × 260 canvas. 900 px
of width become the 5 s clip. Text wider than 92% of the canvas is handled in one of two ways:
- **default:** the font is shrunk until the text fits;
- **`--fixed-font` / `fit=False`:** the canvas is widened to $w'$, and `text_duration` sets
  $d = 5\,w'/900$ s. Glyphs keep their full height of about 3.5 kHz, and the clip gets longer.

**Cost.** Each Griffin–Lim iteration is one STFT and one ISTFT, which is $M$ FFTs of length $N$.
Encoding therefore costs $O(I\,M\,N\log N)$, with $M \propto d$.
- With a fixed 5 s clip, the cost does not depend on the length of the text.
- With `--fixed-font`, the cost grows linearly with the text's rendered width.

Measured on the author's laptop: a 5 s clip encodes in 0.7–0.9 s and a 30.1 s clip in 6.3 s. Decoding
takes about 0.02 s.

**Scores.** Both compare the decoded image $\hat I$ with the target on the same grid. They measure
fidelity to the target, not whether a person can read it.
- **AUC** (legibility): how well decoded intensity separates stroke pixels ($T > 0.5$) from background.
- **SSIM** is reported beside it. SSIM understates legibility under noise on sparse images, and AUC
  saturates on clean text.

## SETUP

```sh
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m pytest -q
```

That one command creates the environment and runs the tests. Run `source .venv/bin/activate` before
the commands below.

The core and the demo need nothing else. Other parts need:
- **video:** cairo, pango, ffmpeg, plus `latex` and `dvisvgm` (TinyTeX; `video/render.sh` adds it to
  `PATH`);
- **paper:** [tectonic](https://tectonic-typesetting.github.io), which downloads its TeX bundle the
  first time it runs;
- **`analyze_recording.py`:** ffmpeg, for formats other than WAV.

## COMMANDS

### python -m sonocanvas

Encodes TEXT into a WAV file, then decodes the WAV and prints SSIM, PSNR and the final spectral
convergence.

| Option | Meaning |
|---|---|
| `TEXT` | Text to draw. Quote it if it contains spaces. |
| `-o, --out PATH` | Output WAV: 16-bit PCM, 22050 Hz, mono. Required. |
| `--png PATH` | Also save the target spectrogram above the decoded one, for checking by eye (the "answer key"). |
| `--mode edges\|raw` | Preprocessing; default `edges`. `edges` keeps Sobel outlines and survives best. `raw` keeps filled shapes, which come back speckled. |
| `--fixed-font` | Keep the 170 px font for long text and lengthen the clip, instead of shrinking the text. |

Viewing the result: open the WAV in a spectrogram app with a **linear** frequency axis reaching at least
8 kHz. Set the app's time window to at least the clip length.

### python experiments.py

Runs the six controlled experiments. Each one writes `data/NAME.npz`, then draws
`figures/NAME.{pdf,png}` from that file alone, and replaces its own rows in `results/table.csv`. Means
and standard deviations are taken over five seeds (`SEEDS`).

Test images:
- **text:** `render_text("DA321")`, using `edges`;
- **line:** the scikit-image horse silhouette, using `edges`;
- **photo:** scikit-image Chelsea, using `raw`.

| Name | Varies |
|---|---|
| `edges_vs_filled` | `raw` vs `edges` preprocessing; also diagnoses what SSIM measures |
| `sweep_window_len` | $N \in \{128,\dots,4096\}$, scored on one fixed image grid and also at half resolution |
| `sweep_window_type` | rect, Hann, Hamming, Blackman; window spectra and sidelobes |
| `gl_iterations` | iterations $\in \{0,1,2,5,10,20,60,100,200\}$; 0 is random phase only. **Takes hours.** |
| `mel_vs_linear` | encode on a linear or mel axis, view on either |
| `noise_robustness` | additive white noise at 30, 20, 10, 5 and 0 dB SNR, relative and fixed-floor |

| Option | Meaning |
|---|---|
| `NAME ...` | Run only these experiments. Default: all of them. |
| `--plot-only` | Redraw the figures from the existing `data/*.npz` without recomputing. |

### python analyze_recording.py

Scores a real recording of a demo WAV that was played through a speaker. The steps are:
1. Read the audio. Formats other than WAV go through ffmpeg.
2. Resample to 22050 Hz.
3. Align the recording to the encoded clip by band-limited cross-correlation:
   $d^\* = \arg\max_d |\sum_n r[n+d]\,x[n]|$, computed by FFT.
4. Decode the aligned 5 s.
5. Report AUC, SSIM, PSNR, the delay, the correlation peak ratio and the in-band SNR.

The procedure for recording takes is in `ACOUSTIC_TEST.md`.

| Option | Meaning |
|---|---|
| `RECORDING` | WAV, or anything ffmpeg reads (for example a phone's `.m4a`). |
| `--dist M` | Speaker-to-microphone distance in metres. Required. |
| `--word W` | The word that was played (default `DA321`). It is re-encoded exactly as in `demo/`. |
| `--device D` | Label for the recording device (default `phone`). |
| `--take N` | Take number (default 1). Re-running a take replaces its row. |
| `--video` | Also write `data/acoustic_video.npz`, which scene 7 of the video then uses. |

Writes a row to `results/acoustic.csv`, and the image to
`figures/acoustic/WORD_Dm_DEVICE_TAKE.png`.

It warns that the scores may be meaningless when the correlation peak ratio is below 40. In simulation,
correct takes gave 130–158 and the wrong word gave 19.

### python paper/make_numbers.py

Generates `paper/numbers.tex` (one macro per result), `paper/table1.tex`, and the Results table at the
end of this README. The sources are `results/table.csv` and `data/*.npz`, plus `results/acoustic.csv`
when it exists. No number in the paper is typed by hand. Rounding is half-up. Then compile with
`cd paper && tectonic main.tex`.

### streamlit run app.py

Live demo: type a word or upload an image, then hear it, see the decoded spectrogram, and read its AUC,
SSIM and PSNR. It runs without internet, and usage statistics are off (`.streamlit/config.toml`).

| Control | Meaning |
|---|---|
| DA321 / HELLO / IITG | Preset words. |
| Word | Free text, up to 24 characters. |
| Keep font size (longer audio) | Same as `--fixed-font`. |
| Or upload an image | PNG or JPEG, encoded as a 5 s clip. |
| Preprocessing | `edges` or `raw`. |
| Presentation mode | Toggle, or open the URL with `?present=1`. Large text, presets only, always 5 s. |

In presentation mode, a live encode that takes longer than `SONOCANVAS_FALLBACK_S` plays
`demo/WORD.wav` instead. That file is bit-identical to a fresh encode. The caption names which source
was used.

### video/render.sh [m|h]

Renders the eight Manim scenes and joins them into `video/sonocanvas_720p30.mp4` (`m`, the default) or
`video/sonocanvas_1080p60.mp4` (`h`). It prints the total duration. Each scene's length comes from
`video/narration.md` through `video/timing.py`.

| Scene | Class | Content |
|---|---|---|
| `scene_01_hook.py` | `Hook` | The screech, then the word appearing in its spectrogram |
| `scene_02_pixel.py` | `Pixel` | A spectrogram pixel as a windowed inner product |
| `scene_03_alphabet.py` | `Alphabet` | Painting with sound: a burst is a dot, a tone a horizontal line, a click a vertical line, a chirp a slope |
| `scene_04_backwards.py` | `Backwards` | Synthesis by overlap-add |
| `scene_05_phase.py` | `MissingPhase` | The missing phase; Griffin–Lim |
| `scene_06_uncertainty.py` | `Uncertainty` | The window-length trade-off, from `data/sweep_window_len.npz` |
| `scene_07_air.py` | `Air` | Griffin–Lim grain (outlines vs filled shapes), then speaker to microphone (uses `data/acoustic_video.npz` once a take is chosen) |
| `scene_08_outro.py` | `Outro` | The whole loop, phone footage (`video/assets/phone.mp4` when recorded), the invitation to the live demo |

Preview one scene: `manim -pqm video/scene_06_uncertainty.py Uncertainty`.

### python video/timing.py

Prints each scene's length and writes the start and end times into the `narration.md` headings.

- Each sentence lasts its word count at 120 wpm (`WPM`).
- `[3 s]` in the text is a silent beat.
- Once `media/voice/NN.wav` exists, that scene's spoken beats are stretched to match the recording.
- It flags a total over 5:00.

### python video/add_voice.py [--prepare]

- **`--prepare`** trims the silence at both ends of each `video/voice/01.*` … `08.*`, and writes them to
  `media/voice/NN.wav` (48 kHz mono).
- **No option:** after `render.sh h`, it mixes each voice file in at its scene's first frame. The scene
  audio stays underneath at 0.7 gain (`BED`). It then loudness-normalises in two passes to −16 LUFS,
  with a −1.5 dBTP ceiling, and writes `video/sonocanvas_final_1080p.mp4`.

### video/check_scene.sh CLIP.mp4 OUT.png

Writes a contact sheet of 6 evenly spaced frames and lists any still hold longer than 3 s.

### ./make_final.sh

Assembles the presentation kit in `final/`:
- the paper PDF;
- `DEMO_CHECKLIST.md`;
- six WAVs (DA321, HELLO, IITG, MMDP, GABOR, PIXEL), each with its answer-key PNG;
- the voiced video, or the silent draft if no voiced video exists yet.

It checks that the preset WAVs match `demo/` byte for byte, and prints the kit's total size.

## LIBRARY

All of these are importable as `sonocanvas.NAME`. Images are float arrays in $[0,1]$ with row 0 at
the top. Spectrogram arrays are bins × frames.

### Encoder and decoder (`sonocanvas/__init__.py`)

**`encode(img, *, n_fft=1024, hop=256, dur=5.0, f_lo=1000, f_hi=8000, sr=22050, scale="linear", window="hann", n_iter=60, init="random", seed=0) -> (x, info)`**
Turns an image into a waveform $x$ with $|\mathrm{STFT}(x)| \approx S(I)$, peak-normalised to
$x \leftarrow 0.9\,x/\max|x|$. The `info` dict holds:
- `S`: the full target magnitude, $(N/2+1)\times M$;
- `band`: a boolean bin mask;
- `target`: `S[band]` flipped and divided by its max, which is the grid `decode` returns;
- `errs`: the spectral convergence after each iteration;
- `sr`: the sample rate.

**`decode(x, *, n_fft=1024, hop=256, dur=5.0, f_lo=1000, f_hi=8000, sr=22050, scale="linear", window="hann") -> rec`**
Turns a waveform into the image $\hat I = |\mathrm{STFT}(x)|[\text{band}]/\max$, on the same grid as
`info["target"]`. It uses only these shared parameters, never encoder state. Pass the same `dur` that
was used to encode.

### Image preparation (`sonocanvas/image_prep.py`)

**`load(path) -> img`**
Reads an image as grayscale: $I = L/255$, where $L$ is Pillow's ITU-R 601 luma.

**`render_text(text, w=900, h=260, fit=True) -> img`**
Draws white DejaVu Sans Bold at 170 px, centred on a black $w \times h$ canvas.
- Text wider than $0.92\,w$ is shrunk to fit when `fit=True`.
- When `fit=False`, the canvas is widened to $w' = \lceil(b_2-b_0)/0.92\rceil$ instead, where
  $(b_0,b_1,b_2,b_3)$ is the text's bounding box.
- It raises if the font file is missing; there is no silent fallback.

**`text_duration(img, w=900, dur=5.0) -> float`**
Returns the clip length that keeps a widened canvas at the default time scale:
$d\cdot\max(1, w'/w)$ seconds. Use it only for text, not for uploaded images.

**`preprocess(img, mode="raw", gamma=1.0, dilate=1) -> img`**
Applies point processing, then optionally extracts edges.
1. Contrast stretch: $I' = (I-\min I)/(\max I-\min I)$.
2. Gamma: $I'' = I'^{\gamma}$.
3. With `mode="edges"`:
   - Sobel gradient magnitude $G = \sqrt{(S_x * I'')^2 + (S_y * I'')^2}$;
   - threshold $E = [G > 0.25\max G]$;
   - dilate `dilate` times with a 3×3 cross.

It raises `ValueError` for an unknown mode.

**`FONT`**
Path to `DejaVuSans-Bold.ttf`, which ships with matplotlib, so text renders identically on every
machine.

### Frequency map (`sonocanvas/freq_map.py`)

**`image_to_mag(img, n_fft, hop, dur, f_lo, f_hi, sr, scale="linear") -> (S, band)`**
Builds the target magnitude $S$, of shape $(N/2+1,\; 1+\lfloor d f_s\rfloor/H)$, and the band mask.
- The image's columns are resampled to frames.
- Bin $k$ sits at $f_k = k f_s/N$ and gets the normalised position $p_k = (f_k-f_\text{lo})/(f_\text{hi}-f_\text{lo})$ when `linear`.
- When `mel`, $p_k$ is the same ratio computed on mel values.
- $S[k,m]$ interpolates the flipped image column at $p_k$ when $p_k\in[0,1]$, and is 0 otherwise.

It raises `ValueError` for an unknown scale.

**`mel(f) -> m`**
$m(f) = 2595\log_{10}(1+f/700)$.

### STFT (`sonocanvas/stft.py`)

**`window(name, N) -> w`**
A periodic (DFT-even) window of length $N$:
- `hann`: $\tfrac12-\tfrac12\cos(2\pi n/N)$
- `hamming`: $0.54-0.46\cos(2\pi n/N)$
- `blackman`: $0.42-0.5\cos(2\pi n/N)+0.08\cos(4\pi n/N)$
- `rect`: $1$

It raises `ValueError` for an unknown name.

**`stft(x, n_fft, hop, win) -> X`**
The centre-padded STFT, of shape $(N/2+1, M)$:
$X[k,m] = \sum_{n=0}^{N-1} x[n+mH]\,w[n]\,e^{-j2\pi kn/N}$.
- $x$ is zero-padded by $N/2$ on each side.
- The phase is frame-relative. $|X|$ is the same as in the absolute-time convention.

**`istft(X, n_fft, hop, win) -> x`**
Weighted overlap-add, the least-squares inverse:
$\hat x[n] = \sum_m w[n-mH]\,y_m[n-mH] \,/\, \sum_m w^2[n-mH]$, where $y_m = \mathrm{IDFT}(X[:,m])$.
It returns $H(M-1)$ samples.

### Phase recovery (`sonocanvas/phase.py`)

**`griffin_lim(S, n_fft, hop, win, n_iter=60, init="random", seed=0) -> (x, errs)`**
Griffin–Lim alternating projections:
$X^{(i+1)} = S\cdot e^{j\angle \mathrm{STFT}(\mathrm{ISTFT}(X^{(i)}))}$.
- Starting point: `random` uses $X^{(0)} = S e^{j2\pi U}$ with $U\sim\mathcal U[0,1)$, seeded; `zero`
  uses $X^{(0)} = S$.
- `errs[i]` is the spectral convergence after iteration $i$.
- With `n_iter=0`, it returns $\mathrm{ISTFT}(X^{(0)})$, which is random phase only.

It raises `ValueError` for an unknown `init`.

### Metrics (`sonocanvas/metrics.py`)

**`auc(target, rec) -> float`**
The stroke/background ROC-AUC, computed as the Mann–Whitney statistic:
$A = P(\hat I_s > \hat I_b) + \tfrac12 P(\hat I_s = \hat I_b)$, for a stroke pixel $s$ ($T>0.5$) and a
background pixel $b$.
- It depends only on the order of the pixel values.
- $A = 1$ is perfect separation, and $A = 0.5$ is none.

**`ssim(a, b) -> float`**
The mean SSIM (Wang et al. 2004) from scikit-image, with a 7×7 window and `data_range=1`:
$\frac{(2\mu_a\mu_b+C_1)(2\sigma_{ab}+C_2)}{(\mu_a^2+\mu_b^2+C_1)(\sigma_a^2+\sigma_b^2+C_2)}$.

**`psnr(a, b) -> float`**
$10\log_{10}(1/\mathrm{MSE})$ dB.

**`spectral_convergence(Y, S) -> float`**
$\|\,|Y|-S\,\|_F / \|S\|_F$, for a complex STFT $Y$ and a target magnitude $S$.

**`recover(x, n_fft, hop, win, band, shape) -> rec`**
$|\mathrm{STFT}(x)|[\text{band}]/\max$, flipped so that row 0 is the highest frequency, then resampled
to `shape`. This is the decoder that `decode` calls.

### Script internals

These functions are not part of the package. They are documented so that the scripts can be read.

| Script | Function | Does |
|---|---|---|
| `sonocanvas/__main__.py` | `figure(target, rec, info, s, p, dur=5.0, f_lo, f_hi)` | Two-panel figure: target above, decoded below. Its width grows with `dur`. |
| `experiments.py` | `base_images()` | The text, line and photo test images |
| | `trial(img, seeds, refs, **p)` | Encodes and decodes once per seed, passing `**p` to `encode`; also scores on each grid in `refs` |
| | `flat_stats`, `diagnostics` | Split SSIM into its contributions from flat windows, from strokes, and from the gain |
| | `window_spectrum(name, N, pad)` | $\lvert W(f)\rvert$ in dB, and the highest sidelobe |
| | `view(rec, scale)`, `mel_inv(m)` | Redraw a decoded image on a linear or mel viewer axis |
| | `noisy_scores(x, target, sigma)` | Scores with added white noise of standard deviation $\sigma$ |
| | `update_table(new)` | Replaces one experiment's rows in `results/table.csv` |
| `analyze_recording.py` | `read_audio`, `to_rate` | Mono float audio at 22050 Hz |
| | `align(r, x)` | The aligned segment, the delay in samples, and the correlation peak ratio |
| | `snr_db(r, d, n)` | In-band SNR from the room noise around the clip: $10\log_{10}(P_\text{clip}/P_\text{noise}-1)$ |
| `app.py` | `encode(img, dur)`, `run` | A live encode with scores; `run` is the cached version |
| | `run_presenting(key, word, img)` | Background encode with a timeout fallback to `prerendered` |
| | `prerendered(word, img)` | Decodes the shipped `demo/WORD.wav` |
| `video/common.py` | `Timed` | Base scene class: `say()` fits one narration sentence, `finish()` fades out |
| | `heat`, `trace`, `label`, `tick`, `math` | Style helpers. Readable text is at least 34 pt. |
| | `acoustic()`, `placeholder(s)` | The chosen recording for scene 7, and the banner on stand-in material |
| `video/make_assets.py` | `main()` | Writes `data/video.npz` (real waveforms and decoded images for the scenes) |

## FILES

| Path | Contents |
|---|---|
| `sonocanvas/` | The library above |
| `tests/` | STFT round trip, COLA, tone bin, end-to-end AUC/SSIM, fixed-font text, metrics, app |
| `data/*.npz` | Experiment output. Figures, paper and video are drawn from these. |
| `figures/` | Experiment figures (PDF and PNG). `figures/acoustic/` holds recorded takes. |
| `results/table.csv` | Every result: experiment, image, param, AUC, AUC std, SSIM, SSIM std, PSNR |
| `results/acoustic.csv` | One row per recorded take (written by `analyze_recording.py`) |
| `demo/*.wav` | Pre-rendered presets, the app's fallback |
| `paper/` | `main.tex` (IEEEtran), `refs.bib`, generated `numbers.tex` and `table1.tex` |
| `video/` | Scenes, `narration.md` (sets the timing), `render.sh`, `voice/` (your recordings) |
| `media/` | Manim output, `list.txt`, trimmed voice files |
| `final/` | Presentation kit (generated, not committed) |
| `ACOUSTIC_TEST.md`, `DEMO_CHECKLIST.md`, `REVIEW.md` | Recording procedure, live-demo checklist, review notes |
| `prototype/` | The original single-file prototype, kept for reference |

## ENVIRONMENT

| Variable | Effect |
|---|---|
| `SONOCANVAS_FALLBACK_S` | Presentation-mode timeout in seconds before the app plays `demo/WORD.wav` (default 4). |
| `PATH` | Must reach `ffmpeg`, `ffprobe` and, for the video, TinyTeX. `render.sh` prepends TinyTeX itself. |

## EXIT STATUS

- **0** on success.
- **2** from any argparse CLI on a usage error.
- **`analyze_recording.py`** exits with a message if the recording is shorter than the clip, or if a
  format other than WAV is given and ffmpeg is missing.
- **`video/add_voice.py`** exits if a voice file is missing. It fails an assertion if the mix is longer
  than 5:00 or reaches 0 dBFS.
- **`make_final.sh`** exits with 1 if a preset WAV differs from `demo/`.
- **Shell scripts** stop at the first failing command (`set -e`).

## EXAMPLES

```sh
python -m sonocanvas "HELLO" -o hello.wav --png hello.png                 # 5 s, 170 px
python -m sonocanvas "DRAWING WITH SOUND" -o dws.wav --fixed-font         # 13.6 s, still 170 px
python experiments.py noise_robustness && python paper/make_numbers.py    # one experiment, refresh numbers
python analyze_recording.py recordings/DA321_1m_1.m4a --dist 1 --take 1   # score a recorded take
python video/add_voice.py --prepare && video/render.sh h && python video/add_voice.py
```

Rebuild everything from scratch:

```sh
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m pytest -q
source .venv/bin/activate
python experiments.py                                # gl_iterations takes hours
python paper/make_numbers.py && (cd paper && tectonic main.tex)
video/render.sh h                                    # after add_voice.py --prepare, if voice files exist
python video/add_voice.py                            # only with voice files
./make_final.sh
```

## RESULTS

These are the headline rows of [`results/table.csv`](results/table.csv), written here by
`paper/make_numbers.py`. They are means over five seeds. Text and line use edge preprocessing and the
photograph uses raw, except in the first two rows.

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

## LIMITATIONS

- AUC and SSIM measure fidelity to the target image, not readability. Small text can score high while
  being hard to read on a phone.
- Acoustic recovery through a real speaker and microphone is measured only once `results/acoustic.csv`
  exists (`ACOUSTIC_TEST.md`).
- The message is visible to anyone who opens a spectrogram, so this is not steganography.

## SEE ALSO

`ACOUSTIC_TEST.md`, `DEMO_CHECKLIST.md`, `REVIEW.md`, `paper/main.pdf`, `video/narration.md`.

D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform," *IEEE
Trans. ASSP*, 1984. Z. Wang et al., "Image quality assessment: from error visibility to structural
similarity," *IEEE Trans. Image Process.*, 2004.

Test images from scikit-image: the horse silhouette and Chelsea the cat, both CC0.
