# SonoCanvas: Drawing with Sound

Type a word and get a WAV. Open the WAV in a spectrogram viewer, and you can read the word.

Individual project for DA3021 Multimodal Data Processing, IIT Guwahati.

## How it works

1. Draw the text (or load an image) and keep its edges.
2. Put the picture onto the 1 to 8 kHz band. Time goes left to right, and the top row is 8 kHz.
3. The picture gives only the magnitude of the STFT. Griffin-Lim finds a phase that fits it.
4. An inverse STFT (overlap-add) gives the audio.
5. To decode, take the STFT of the audio and crop the same band.

The STFT, the overlap-add and Griffin-Lim are written by hand in `src/sonocanvas.cpp`. OpenCV does the
FFT and the image work. libsndfile reads and writes the WAV files.

## Build

macOS with Homebrew:

```sh
brew install opencv libsndfile cli11 doctest ffmpeg
make
make test
```

`make test` also runs the app tests. These need the Python environment:

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Use

```sh
bin/sonocanvas encode HELLO -o hello.wav --png hello.png    # make the audio
bin/sonocanvas decode hello.wav HELLO                       # score the audio against HELLO
streamlit run app.py                                        # demo page in the browser
```

`--png` saves the target picture above the decoded picture. Open the WAV in any spectrogram app with a
linear frequency axis up to 8 kHz.

Other options: `--image FILE` encodes a PNG or JPEG. `--mode raw` keeps filled shapes instead of edges.
`--fixed-font` keeps the font size and makes the audio longer. `bin/sonocanvas encode --help` lists them all.

## Experiments

```sh
bin/experiments          # all six experiments, about 2.5 min
python plots.py          # draw figures/ from data/
```

The experiments change one thing at a time: edges vs filled shapes, window length, window type,
Griffin-Lim iterations, mel vs linear axis, and noise. Each result is the mean over five seeds and goes
to `results/table.csv`.

`bin/analyze_recording` scores a real recording of a WAV played through a speaker. The phone takes
at 1 m are in `results/acoustic.csv` and `figures/acoustic/`.

## Results

Scores compare the decoded picture with the target. AUC says how well strokes stand out from the
background (1 is perfect, 0.5 is chance). SSIM compares the structure.

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

## Files

| Path | What |
|---|---|
| `src/` | The C++ library, the programs and the tests |
| `app.py` | The Streamlit demo page |
| `plots.py` | Draws the figures |
| `data/` | Experiment data (`.npz`) and the two test images |
| `figures/` | Experiment figures |
| `results/` | Result tables (CSV) |

## License

MIT. The DejaVu font and cnpy keep their own licenses (`src/fonts/`, `src/third_party/`). The test
images come from scikit-image and are CC0.
