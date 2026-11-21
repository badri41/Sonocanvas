# SONOCANVAS(1)

Drawing with Sound. Individual project for DA321 Multimodal Data Processing, IIT Guwahati.

## NAME

**sonocanvas**: encode an image or typed text into audio whose spectrogram shows the picture, and decode
it back. The core and all computation are C++23.

## SYNOPSIS

```
make                                   # build bin/*
make test                              # C++ unit tests, then the app tests
bin/sonocanvas encode TEXT -o OUT.wav [--image FILE] [--mode edges|raw] [--fixed-font] [--png F] [--npz F]
bin/sonocanvas decode IN.wav TEXT     [--image FILE] [--mode edges|raw] [--fixed-font] [--png F] [--npz F]
bin/experiments [NAME ...]
bin/analyze_recording RECORDING --dist METRES [--word WORD] [--device NAME] [--take N] [--video]
bin/make_numbers
bin/make_assets
make experiments | figures | numbers | paper | assets | video | demo | final | clean
streamlit run app.py                   # http://localhost:8501/?present=1 for presentation mode
```

```cpp
#include "sonocanvas.hpp"
sc::Encoded e = sc::encode(sc::preprocess(sc::render_text("HELLO"), "edges"));
sc::Image rec = sc::decode(e.x);
double a = sc::auc(e.target, rec), s = sc::ssim(e.target, rec);
```

## DESCRIPTION

An image $I$ is mapped onto a target STFT magnitude $S$. The time axis becomes STFT frames, and the
vertical axis becomes the bins between 1 and 8 kHz, with the top row at 8 kHz. The phase that makes
$S$ a valid spectrogram is unknown, so Griffin–Lim estimates it. The waveform is then synthesised by
weighted overlap-add. Decoding is the analysis STFT, cropped to the same band.

$$I \;\xrightarrow{\text{image\_to\_mag}}\; S \;\xrightarrow{\text{griffin\_lim}}\; x
\;\xrightarrow{\text{stft, crop}}\; \hat I$$

The STFT framing, the weighted overlap-add and Griffin–Lim are written by hand in `cpp/`. Libraries
do the generic work:
- **OpenCV:** the DFT (`cv::dft`), resizing, Sobel, dilation, the box filter for SSIM, and FreeType
  text;
- **libsndfile:** WAV files;
- **CLI11:** command-line options;
- **doctest:** tests;
- **cnpy:** `.npz` files for the Python plotting and video code.

**Defaults:** $f_s = 22050$ Hz, Hann window, $N = 1024$, hop $H = 256$, band 1000–8000 Hz, clip length
$d = 5$ s, 60 Griffin–Lim iterations.

With these defaults:
- the band is bins 47–371 (1012–7989 Hz), which is 325 bins;
- a 5 s clip has $M = 1 + \lfloor d f_s \rfloor / H = 431$ frames;
- the decoded image is 325 × 431;
- the WAV holds $H(M-1) = 110080$ samples (4.99 s), peak-normalised to 0.9.

**Text canvas.** `render_text` draws DejaVu Sans Bold at 170 px (`cpp/fonts/`), centred on a 900 × 260
canvas. 900 px of width become the 5 s clip. Text wider than 92% of the canvas is handled in one of two
ways:
- **default:** the font is shrunk until the text fits;
- **`--fixed-font` / `fit = false`:** the canvas is widened to $w'$, and `text_duration` sets
  $d = 5\,w'/900$ s. Glyphs keep their full height of about 3.5 kHz, and the clip gets longer.

**Cost.** Each Griffin–Lim iteration is one STFT and one ISTFT, which is $M$ FFTs of length $N$.
Encoding therefore costs $O(I\,M\,N\log N)$, with $M \propto d$.
- With a fixed 5 s clip, the cost does not depend on the length of the text.
- With `--fixed-font`, the cost grows linearly with the text's rendered width.

Measured on the author's M1 laptop (`-O2`, wall time including process start, median of 5 runs after
one warm-up, with a video render running in the background):

| Workload | C++ (`bin/`) | The earlier Python version |
|---|---|---|
| encode a 5 s clip | 0.38 s | 0.8 s |
| encode a 30.1 s clip (43 characters, `--fixed-font`) | 2.1 s | 6.3 s |
| decode and score a 5 s WAV | 0.04 s | not measured |
| all six experiments | about 2.5 min | hours (`gl_iterations` alone) |

**Scores.** Both compare the decoded image $\hat I$ with the target on the same grid. They measure
fidelity to the target, not whether a person can read it.
- **AUC** (legibility): how well decoded intensity separates stroke pixels ($T > 0.5$) from background.
- **SSIM** is reported beside it. SSIM understates legibility under noise on sparse images, and AUC
  saturates on clean text.

## SETUP

```sh
brew install opencv libsndfile cli11 doctest ffmpeg
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
make && make test
```

- **`make`** builds the C++ programs into `bin/`. It uses `clang++` (C++23) and expects Homebrew's
  prefix `/opt/homebrew`.
- **Python** is used only for the plots, the app, the video and the voice mix. Run
  `source .venv/bin/activate` before those.
- **Video:** also needs cairo, pango, plus `latex` and `dvisvgm` (TinyTeX; `video/render.sh` adds it to
  `PATH`).
- **Paper:** needs [tectonic](https://tectonic-typesetting.github.io), which downloads its TeX bundle
  the first time it runs.

Your editor's clangd finds the headers through `compile_flags.txt`.

## COMMANDS

### bin/sonocanvas encode | decode

**`encode`** draws TEXT (or loads `--image`), encodes it, and writes the WAV.

**`decode`** reads an existing WAV and scores it against the picture that TEXT (or `--image`) would
make. Both print AUC, SSIM and PSNR; `encode` also prints the final spectral convergence.

| Option | Meaning |
|---|---|
| `TEXT` | Text to draw. Quote it if it contains spaces. |
| `-o, --out PATH` | (`encode`) Output WAV: 16-bit PCM, 22050 Hz, mono. |
| `IN` | (`decode`) The WAV to score. It must be 22050 Hz. |
| `--image FILE` | A grayscale PNG or JPEG instead of TEXT. Always a 5 s clip. |
| `--mode edges\|raw` | Preprocessing; default `edges`. `edges` keeps Sobel outlines and survives best. `raw` keeps filled shapes, which come back speckled. |
| `--fixed-font` | Keep the 170 px font for long text and lengthen the clip, instead of shrinking the text. |
| `--png PATH` | Save the target above the decoded picture, for checking by eye (the "answer key"). |
| `--npz PATH` | Save `target`, `rec`, `auc`, `ssim`, `psnr`, `sc` and `dur` as NumPy arrays (the app reads this). |

Viewing the result: open the WAV in a spectrogram app with a **linear** frequency axis reaching at least
8 kHz. Set the app's time window to at least the clip length.

It exits with 1 if the picture is empty (try the other `--mode`), and with 2 on a usage error.

### bin/experiments [NAME ...]

Runs the six controlled experiments, or only the named ones. Each one writes `data/NAME.npz` and
replaces its own rows in `results/table.csv`. Means and population standard deviations are taken over
five seeds. `make experiments` runs them all and then redraws the figures.

Test images:
- **text:** `render_text("DA321")`, using `edges`;
- **line:** `data/images/horse.png` (CC0), using `edges`;
- **photo:** `data/images/chelsea.png` (CC0, 16-bit gray), using `raw`.

| Name | Varies |
|---|---|
| `edges_vs_filled` | `raw` vs `edges` preprocessing; also diagnoses what SSIM measures |
| `sweep_window_len` | $N \in \{128,\dots,4096\}$, scored on one fixed image grid and also at half resolution |
| `sweep_window_type` | rect, Hann, Hamming, Blackman; window spectra and sidelobes |
| `gl_iterations` | iterations $\in \{0,1,2,5,10,20,60,100,200\}$; 0 is random phase only |
| `mel_vs_linear` | encode on a linear or mel axis, view on either |
| `noise_robustness` | additive white noise at 30, 20, 10, 5 and 0 dB SNR, relative and fixed-floor |

### python plots.py [NAME ...]

Draws `figures/NAME.{pdf,png}` from `data/NAME.npz` with matplotlib. These are the paper's figures, and
this script is the only thing that draws them.

### bin/analyze_recording

Scores a real recording of a demo WAV that was played through a speaker. The steps are:
1. Convert the recording to 22050 Hz mono with ffmpeg (any format, for example a phone's `.m4a`).
2. Keep only 1–8 kHz, by zeroing all other DFT bins.
3. Find the delay by cross-correlation with the encoded clip (`cv::matchTemplate`):
   $d^\* = \arg\max_d |\sum_n r[n+d]\,x[n]|$.
4. Decode the aligned 5 s.
5. Report AUC, SSIM, PSNR, the delay, the correlation peak ratio and the in-band SNR.

The procedure for recording takes is in `ACOUSTIC_TEST.md`.

| Option | Meaning |
|---|---|
| `RECORDING` | The recording file. |
| `--dist M` | Speaker-to-microphone distance in metres. Required. |
| `--word W` | The word that was played (default `DA321`). It is re-encoded exactly as in `demo/`. |
| `--device D` | Label for the recording device (default `phone`). |
| `--take N` | Take number (default 1). Re-running a take replaces its row. |
| `--video` | Also write `data/acoustic_video.npz`, which scene 7 of the video then uses. |

Writes a row to `results/acoustic.csv` and the image to `figures/acoustic/WORD_Dm_DEVICE_TAKE.png`.

It warns that the scores may be meaningless when the correlation peak ratio is below 40. In a simulated
take, the correct word gave 194 and the wrong word 22.

### bin/make_numbers

Generates `paper/numbers.tex` (one macro per result), `paper/table1.tex`, `paper/acoustic.tex`, and
the Results table at the end of this README. The sources are `results/table.csv` and `data/*.npz`, plus
`results/acoustic.csv` when it exists. No number in the paper is typed by hand.

Rounding is half-up on each value's shortest decimal form, so 0.9135 prints as 0.914. `make paper` runs
this and then tectonic.

### bin/make_assets

Writes `data/video.npz`: the DA321 waveform, its target, the decoded images after 0, 1, 5, 20 and 60
Griffin–Lim iterations, and scene 3's five shapes (dot, horizontal, vertical, diagonal, circle) encoded
at 1.6 s.

### streamlit run app.py

Live demo: type a word or upload an image, then hear it, see the answer key, and read its AUC, SSIM and
PSNR. The page only runs `bin/sonocanvas` and shows what it writes. It runs without internet, and usage
statistics are off (`.streamlit/config.toml`).

| Control | Meaning |
|---|---|
| DA321 / HELLO / IITG | Preset words. |
| Word | Free text, up to 24 characters. |
| Keep font size (longer audio) | Same as `--fixed-font`. |
| Or upload an image | PNG or JPEG, encoded as a 5 s clip. |
| Preprocessing | `edges` or `raw`. |
| Presentation mode | Toggle, or open the URL with `?present=1`. Large text, presets only, always 5 s. |

In presentation mode, an encode that takes longer than `SONOCANVAS_FALLBACK_S` plays `demo/WORD.wav`
instead. That file is bit-identical to a fresh encode. The caption names which source was used.

### video/render.sh [m|h]

Renders the eight Manim scenes and joins them into `video/sonocanvas_720p30.mp4` (`m`, the default) or
`video/sonocanvas_1080p60.mp4` (`h`). It prints the total duration. Each scene's length comes from
`video/narration.md` through `video/timing.py`. The scenes read only `data/*.npz`.

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

- Each sentence lasts its word count at 120 wpm.
- `[3 s]` in the text is a silent beat.
- Once `media/voice/NN.wav` exists, that scene's spoken beats are stretched to match the recording.
- It flags a total over 5:00.

### python video/add_voice.py [--prepare]

- **`--prepare`** trims the silence at both ends of each `video/voice/01.*` … `08.*`, and writes them to
  `media/voice/NN.wav` (48 kHz mono).
- **No option:** after `render.sh h`, it mixes each voice file in at its scene's first frame. The scene
  audio stays underneath at 0.7 gain. It then loudness-normalises in two passes to −16 LUFS, with a
  −1.5 dBTP ceiling, and writes `video/sonocanvas_final_1080p.mp4`.

### video/check_scene.sh CLIP.mp4 OUT.png

Writes a contact sheet of 6 evenly spaced frames and lists any still hold longer than 3 s.

### make final

Assembles the presentation kit in `final/`:
- the paper PDF;
- `DEMO_CHECKLIST.md`;
- six WAVs (DA321, HELLO, IITG, MMDP, GABOR, PIXEL), each with its answer-key PNG;
- the voiced video, or the silent draft if no voiced video exists yet.

It checks that the preset WAVs match `demo/` byte for byte, and prints the kit's total size.

## LIBRARY (cpp/sonocanvas.hpp, namespace sc)

Types:
- **`Image`** = `cv::Mat_<double>`: rows × cols, values in $[0,1]$, row 0 at the top.
- **`Spectrum`** = `cv::Mat_<std::complex<double>>`: bins × frames, bin 0 at 0 Hz.
- **`Signal`** = `std::vector<double>`: audio samples.
- **`Params`** holds the shared settings, with the defaults above: `n_fft`, `hop`, `dur`, `f_lo`, `f_hi`,
  `sr`, `scale` (`"linear"` or `"mel"`) and `window`. `frames()` returns
  $M = 1 + \lfloor d f_s\rfloor/H$.

### Encoder and decoder (`sonocanvas.cpp`)

**`Encoded encode(const Image &img, const Params &p = {}, int n_iter = 60, const std::string &init = "random", unsigned seed = 0)`**
Turns an image into a waveform $x$ with $|\mathrm{STFT}(x)| \approx S(I)$, peak-normalised to
$x \leftarrow 0.9\,x/\max|x|$. `Encoded` holds:
- `x`: the waveform;
- `S`: the full target magnitude, $(N/2+1)\times M$;
- `target`: `S` on the band, flipped and divided by its max, which is the grid `decode` returns;
- `errs`: the spectral convergence after each iteration.

**`Image decode(const Signal &x, const Params &p = {})`**
Turns a waveform into the image $\hat I = |\mathrm{STFT}(x)|[\text{band}]/\max$, on the target grid. It
uses only `p`, never encoder state. Pass the same `dur` that was used to encode.

**`Image band_image(const Image &mag, const Params &p)`**
Takes a bins × frames magnitude, keeps the band, flips it so row 0 is $f_\text{hi}$, and divides by its
max.

### Image preparation (`image_prep.cpp`)

**`Image load(const std::string &path)`**
Reads an 8- or 16-bit grayscale image as $[0,1]$.

**`Image render_text(const std::string &text, int w = 900, int h = 260, bool fit = true)`**
Draws white DejaVu Sans Bold at 170 px, centred by its ink bounding box on a black $w \times h$ canvas.
- Text wider than $0.92\,w$ is shrunk to fit when `fit` is true.
- When `fit` is false, the canvas is widened to $w' = \lceil \text{width}/0.92\rceil$ instead.

**`double text_duration(const Image &img, int w = 900, double dur = 5.0)`**
Returns the clip length that keeps a widened canvas at the default time scale:
$d\cdot\max(1, w'/w)$. Use it only for text.

**`Image preprocess(const Image &img, const std::string &mode = "raw", double gamma = 1.0, int dilate = 1)`**
Applies point processing, then optionally extracts edges.
1. Contrast stretch: $I' = (I-\min I)/(\max I-\min I)$.
2. Gamma: $I'' = I'^{\gamma}$.
3. With `"edges"`:
   - Sobel magnitude $G = \sqrt{G_x^2 + G_y^2}$;
   - threshold $E = [G > 0.25\max G]$;
   - dilate `dilate` times with a 3×3 cross.

It throws `std::invalid_argument` for an unknown mode.

**`Image resize(const Image &img, int rows, int cols)`**
An anti-aliased resize, the same recipe as scikit-image's:
1. Blur with a Gaussian of $\sigma = \max(0, (s-1)/2)$ per axis, where $s$ is the shrink factor.
2. Interpolate bilinearly.

### Frequency map (`freq_map.cpp`)

**`Image image_to_mag(const Image &img, const Params &p)`**
Builds the target magnitude $S$, $(N/2+1) \times M$.
- The image's columns are resampled to the $M$ frames.
- Bin $k$ at $f_k = k f_s/N$ gets the position $p_k = (f_k-f_\text{lo})/(f_\text{hi}-f_\text{lo})$,
  computed on mel values for `"mel"`.
- $S[k,m]$ interpolates column $m$ linearly at $p_k$; bins outside the band are 0.

**`cv::Range band(const Params &p)`**
The bins with $f_\text{lo} \le f_k \le f_\text{hi}$, as a row range.

**`double mel(double f)`**
$m(f) = 2595\log_{10}(1+f/700)$.

### STFT (`stft.cpp`)

**`std::vector<double> window(const std::string &name, int N)`**
A periodic window of length $N$:
- `hann`: $\tfrac12-\tfrac12\cos(2\pi n/N)$
- `hamming`: $0.54-0.46\cos(2\pi n/N)$
- `blackman`: $0.42-0.5\cos(2\pi n/N)+0.08\cos(4\pi n/N)$
- `rect`: $1$

**`Spectrum stft(const Signal &x, int n_fft, int hop, const std::vector<double> &win)`**
The centre-padded STFT:
$X[k,m] = \sum_{n=0}^{N-1} x[n+mH]\,w[n]\,e^{-j2\pi kn/N}$, $k = 0..N/2$.
- $x$ is zero-padded by $N/2$ on each side.
- Each windowed frame is a row of a matrix, and one `cv::dft` call with `DFT_ROWS` transforms them all.

**`Signal istft(const Spectrum &X, int n_fft, int hop, const std::vector<double> &win)`**
Weighted overlap-add, the least-squares inverse:
$\hat x[n] = \sum_m w[n-mH]\,y_m[n-mH] \,/\, \sum_m w^2[n-mH]$.
- Each $y_m$ comes from one inverse `cv::dft` of the frame's packed half spectrum.
- It returns $H(M-1)$ samples.

**`Image magnitude(const Spectrum &X)`**
$|X|$ element-wise.

### Phase recovery (`phase.cpp`)

**`Phase_Result griffin_lim(const Image &S, int n_fft, int hop, const std::vector<double> &win, int n_iter = 60, const std::string &init = "random", unsigned seed = 0)`**
Each iteration computes $Y = \mathrm{STFT}(\mathrm{ISTFT}(X))$, then sets $X \leftarrow S \cdot Y/|Y|$,
which keeps $Y$'s phase and restores the target magnitudes.
- Starting point: `random` uses $X^{(0)} = S e^{j2\pi U}$, with $U$ uniform from a seeded
  `std::mt19937`; `zero` uses $X^{(0)} = S$.
- It returns `x` and `errs`, the spectral convergence per iteration.

### Metrics (`metrics.cpp`)

**`double auc(const Image &target, const Image &rec)`**
The stroke/background AUC, computed as the Mann–Whitney statistic:
$A = P(\hat I_s > \hat I_b) + \tfrac12 P(\hat I_s = \hat I_b)$.
- It ranks all the decoded values once, with ties sharing their average rank.
- $A = 1$ is perfect separation, and $A = 0.5$ is none.

**`Image ssim_map(const Image &a, const Image &b)`** and **`double ssim(const Image &a, const Image &b)`**
SSIM (Wang et al. 2004), with a 7×7 box window (`cv::blur`) and sample (co)variances:
$\frac{(2\mu_a\mu_b+C_1)(2\sigma_{ab}+C_2)}{(\mu_a^2+\mu_b^2+C_1)(\sigma_a^2+\sigma_b^2+C_2)}$.
`ssim` is the map's mean with the 3-pixel border excluded.

**`double psnr(const Image &a, const Image &b)`**
$10\log_{10}(1/\mathrm{MSE})$ dB.

**`double spectral_convergence(const Spectrum &Y, const Image &S)`**
$\|\,|Y|-S\,\|_F / \|S\|_F$.

### Files (`io.cpp`)

| Function | Does |
|---|---|
| `write_wav(path, x, sr)`, `read_wav(path, &sr)` | 16-bit WAV through libsndfile; reading mixes down to mono |
| `save_figure(path, target, rec, top, bottom)` | The answer-key PNG: two magma panels with titles |
| `split_csv(line)`, `read_csv(path)`, `write_csv(path, header, rows)` | Plain CSV, no quoting. `read_csv` skips the header, `write_csv` creates the folder |
| `Npz(path).add(name, value)` | Writes images, vectors, tables (rows x cols), integer vectors or scalars into a NumPy `.npz`; creates the folder |

## FILES

| Path | Contents |
|---|---|
| `Makefile` | Build and pipeline targets (`make help`) |
| `cpp/` | The library, the five programs, `test_core.cpp`, `fonts/DejaVuSans-Bold.ttf`, `third_party/cnpy` (MIT; patched to write 0-d arrays) |
| `plots.py`, `app.py` | The figures and the demo page (Python) |
| `tests/test_app.py` | The app's tests (pytest) |
| `data/*.npz` | Experiment and video data. Figures, paper and video are drawn from these. |
| `data/images/` | The two CC0 test images |
| `figures/` | Experiment figures (PDF and PNG). `figures/acoustic/` holds recorded takes. |
| `results/table.csv` | Every result: experiment, image, param, AUC, AUC std, SSIM, SSIM std, PSNR |
| `results/acoustic.csv` | One row per recorded take |
| `demo/*.wav` | Pre-rendered presets, the app's fallback |
| `paper/` | `main.tex` (IEEEtran), `refs.bib`, generated `numbers.tex`, `table1.tex`, `acoustic.tex` |
| `video/` | Manim scenes, `narration.md` (sets the timing), `render.sh`, `voice/` (your recordings) |
| `final/` | Presentation kit (generated, not committed) |

## ENVIRONMENT

| Variable | Effect |
|---|---|
| `SONOCANVAS_FALLBACK_S` | Presentation-mode timeout in seconds before the app plays `demo/WORD.wav` (default 4). |
| `PATH` | Must reach `clang++`, `ffmpeg`, `ffprobe` and, for the video, TinyTeX. |

## EXIT STATUS

- **0** on success.
- **1** from `bin/sonocanvas` when the picture is empty or a WAV is not 22050 Hz, and from
  `bin/analyze_recording` when the recording is shorter than the clip.
- **2** on a usage error.
- **`make final`** fails if a preset WAV differs from `demo/`.
- **`video/add_voice.py`** fails if the mix is longer than 5:00 or reaches 0 dBFS.

## EXAMPLES

```sh
bin/sonocanvas encode HELLO -o hello.wav --png hello.png                 # 5 s, 170 px
bin/sonocanvas encode "DRAWING WITH SOUND" -o dws.wav --fixed-font       # longer clip, still 170 px
bin/sonocanvas decode hello.wav HELLO                                    # score a WAV
bin/experiments noise_robustness && python plots.py noise_robustness && make paper
bin/analyze_recording recordings/DA321_1m_1.m4a --dist 1 --take 1       # score a recorded take
```

Rebuild everything from scratch:

```sh
make && make test
make experiments                      # about 2.5 min, then figures
make paper                            # numbers, then paper/main.pdf
make assets && video/render.sh h      # after add_voice.py --prepare, if voice files exist
python video/add_voice.py             # only with voice files
make demo && make final
```

## RESULTS

These are the headline rows of [`results/table.csv`](results/table.csv), written here by
`bin/make_numbers`. They are means over five seeds. Text and line use edge preprocessing and the
photograph uses raw, except in the first two rows.

<!-- results -->
| Configuration | Text AUC | Text SSIM | Line AUC | Line SSIM | Photo SSIM |
|---|---|---|---|---|---|
| Filled / raw | 0.9999 | 0.869 | 1.0000 | 0.698 | 0.377 |
| Sobel edges | 0.9998 | 0.952 | 0.9997 | 0.947 | 0.898 |
| N=128, image grid | 0.9556 | 0.805 | 0.9545 | 0.764 | 0.522 |
| N=1024, image grid | 0.9999 | 0.916 | 0.9995 | 0.936 | 0.509 |
| N=4096, image grid | 0.9889 | 0.758 | 0.9976 | 0.831 | 0.637 |
| Rectangular window | 0.9973 | 0.729 | 0.9998 | 0.744 | 0.512 |
| Random phase only | 0.9897 | 0.848 | 0.9917 | 0.854 | 0.113 |
| 200 iterations | 0.9999 | 0.955 | 0.9998 | 0.951 | 0.417 |
| Linear encode, mel view | 0.7838 | 0.707 | 0.7162 | 0.665 | 0.111 |
| 30 dB SNR | 0.9998 | 0.874 | 0.9997 | 0.845 | 0.375 |
| 10 dB SNR | 0.9998 | 0.213 | 0.9997 | 0.188 | 0.259 |
| 0 dB SNR | 0.9976 | 0.129 | 0.9954 | 0.115 | 0.100 |
<!-- /results -->

## LIMITATIONS

- AUC and SSIM measure fidelity to the target image, not readability. Small text can score high while
  being hard to read on a phone.
- Acoustic recovery through a real speaker and microphone is measured only once `results/acoustic.csv`
  exists (`ACOUSTIC_TEST.md`).
- The message is visible to anyone who opens a spectrogram, so this is not steganography.

## SEE ALSO

`ACOUSTIC_TEST.md`, `DEMO_CHECKLIST.md`, `paper/main.pdf`, `video/narration.md`.

D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform," *IEEE
Trans. ASSP*, 1984. Z. Wang et al., "Image quality assessment: from error visibility to structural
similarity," *IEEE Trans. Image Process.*, 2004.

Test images from scikit-image: the horse silhouette and Chelsea the cat, both CC0.
