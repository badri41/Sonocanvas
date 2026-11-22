/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

SonoCanvas: hide a picture in the spectrogram of a sound.

encode: image -> target magnitude S -> Griffin-Lim phase -> weighted overlap-add -> x
decode: x -> |STFT(x)| on the band -> image on the target grid

An Image is rows x cols with values in [0, 1], row 0 at the top. A Spectrum is bins x frames,
bin 0 is 0 Hz. Audio is just a vector of doubles.
*/

#pragma once

#include <complex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace sc {

using Signal = std::vector<double>;
using Image = cv::Mat_<double>;
using Spectrum = cv::Mat_<std::complex<double>>;

/*
   Everything the encoder and the decoder have to agree on. The defaults are the ones from the paper.
*/
struct Params {
    int n_fft = 1024;              // window length N
    int hop = 256;                 // hop H
    double dur = 5.0;              // clip length d in seconds
    double f_lo = 1000, f_hi = 8000;
    int sr = 22050;                // sampling rate f_s
    std::string scale = "linear";  // "linear" or "mel"
    std::string window = "hann";   // "hann", "hamming", "blackman" or "rect"
    int frames() const { return 1 + int(dur * sr) / hop; }  // M = 1 + floor(d*f_s)/H
};

/*
   Periodic window of length N, n = 0..N-1:
     hann     w[n] = 1/2 - 1/2*cos(2pi*n/N)
     hamming  w[n] = 0.54 - 0.46*cos(2pi*n/N)
     blackman w[n] = 0.42 - 0.5*cos(2pi*n/N) + 0.08*cos(4pi*n/N)
     rect     w[n] = 1
*/
std::vector<double> window(const std::string &name, int N);

/*
   STFT. I pad x with N/2 zeros on both sides so the first frame is centred on sample 0,
   then X[k,m] = sum_{n=0..N-1} x[n + mH]*w[n]*exp(-j*2pi*k*n/N). You get (N/2+1) x M back.
*/
Spectrum stft(const Signal &x, int n_fft, int hop, const std::vector<double> &win);

/*
   Inverse STFT by weighted overlap-add (Griffin & Lim 1984). Every frame goes back through the
   IDFT, gets windowed again and summed, then I divide by the summed squared window:
     xhat[n] = sum_m w[n-mH]*y_m[n-mH] / sum_m w[n-mH]^2
   That's the least-squares answer, so it does the right thing even when X isn't a real STFT,
   which inside Griffin-Lim it never is.
   The N/2 padding is cut off, so you get H*(M-1) samples.
*/
Signal istft(const Spectrum &X, int n_fft, int hop, const std::vector<double> &win);

/*
   |X| for every cell.
*/
Image magnitude(const Spectrum &X);

struct Phase_Result {
    Signal x;                  // the final waveform
    std::vector<double> errs;  // spectral convergence after every iteration
};

/*
   Griffin-Lim. All I have is the magnitudes S, no phase. So I guess a phase, go to the time
   domain and back, keep whatever phase came out and put my magnitudes back on it. n_iter times.
     X_{i+1} = S*Y/|Y|  where  Y = STFT(ISTFT(X_i))
   init "random" starts from a seeded random phase, init "zero" starts from phase 0.
*/
Phase_Result griffin_lim(const Image &S, int n_fft, int hop, const std::vector<double> &win, int n_iter = 60,
                         const std::string &init = "random", unsigned seed = 0);

/*
   Hz to mel: m(f) = 2595*log10(1 + f/700).
*/
double mel(double f);

/*
   The FFT bins that fall between f_lo and f_hi, as a row range of S.
*/
cv::Range band(const Params &p);

/*
   Turns the picture into the magnitude I want the sound to have. Columns become the M frames.
   Each bin in the band looks up where it sits between f_lo (bottom row) and f_hi (top row) and
   takes the picture's value there, interpolated between the two nearest rows.
   Everything outside the band stays 0.
*/
Image image_to_mag(const Image &img, const Params &p);

/*
   Loads a grayscale image (8 or 16 bit) into [0, 1].
*/
Image load(const std::string &path);

/*
   White DejaVu Sans Bold at 170 px, centred on a black w x h canvas. If the text is too wide
   (more than 92% of w) I either shrink the font (fit = true) or make the canvas wider (fit = false).
*/
Image render_text(const std::string &text, int w = 900, int h = 260, bool fit = true);

/*
   How long the clip has to be so a wider canvas still gets 900 px per 5 s: d*max(1, w'/w).
*/
double text_duration(const Image &img, int w = 900, double dur = 5.0);

/*
   Stretches the image to [0, 1] and applies gamma. "raw" stops there. "edges" keeps only the
   outlines: Sobel magnitude G, keep G > 0.25*max G, then thicken them with a 3x3 cross `dilate` times.
*/
Image preprocess(const Image &img, const std::string &mode = "raw", double gamma = 1.0, int dilate = 1);

/*
   Resize without aliasing: blur first (sigma = max(0, (s - 1)/2), s = how much I shrink),
   then bilinear. Same recipe as scikit-image's resize. Plain cv::resize aliases when it shrinks.
*/
Image resize(const Image &img, int rows, int cols);

/*
   SSIM (Wang et al. 2004) over 7x7 windows:
     (2*mu_a*mu_b + C1)*(2*sigma_ab + C2) / ((mu_a^2 + mu_b^2 + C1)*(sigma_a^2 + sigma_b^2 + C2))
   with C1 = 0.01^2, C2 = 0.03^2. ssim() averages the map, skipping a 3 pixel border.
*/
Image ssim_map(const Image &a, const Image &b);
double ssim(const Image &a, const Image &b);

/*
   PSNR = 10*log10(1/MSE) dB.
*/
double psnr(const Image &a, const Image &b);

/*
   How well the decoded image tells strokes from background. Pick a random stroke pixel s
   (target > 0.5) and a random background pixel b; AUC is the chance that rec[s] > rec[b],
   ties count half. 1 is perfect, 0.5 is a coin flip. SSIM gets confused by speckle on flat
   areas, this doesn't, so this is the number I trust for "can you read it".
*/
double auc(const Image &target, const Image &rec);

/*
   SC = norm(|Y| - S) / norm(S). How far the magnitudes I got are from the ones I wanted.
*/
double spectral_convergence(const Spectrum &Y, const Image &S);

struct Encoded {
    Signal x;                  // the waveform, peak at 0.9
    Image S;                   // full target magnitude, (N/2+1) x M
    Image target;              // S cut to the band, flipped so row 0 is f_hi, max 1
    std::vector<double> errs;  // Griffin-Lim error per iteration
};

/*
   Cuts a magnitude down to the band, flips it so high frequencies are on top and scales it to max 1.
*/
Image band_image(const Image &mag, const Params &p);

/*
   The whole encoder: picture -> target magnitude -> Griffin-Lim -> waveform.
*/
Encoded encode(const Image &img, const Params &p = {}, int n_iter = 60, const std::string &init = "random",
               unsigned seed = 0);

/*
   The decoder: STFT the sound, cut out the band, and put it on the same grid as Encoded::target.
   It only knows Params, nothing from the encoder, exactly like a phone app.
*/
Image decode(const Signal &x, const Params &p = {});

/*
   16-bit mono WAV out. read_wav mixes everything down to mono and tells you the sample rate.
*/
void write_wav(const std::string &path, const Signal &x, int sr);
Signal read_wav(const std::string &path, int *sr);

/*
   The "answer key" PNG: target on top, decoded below, both in magma with a title.
*/
void save_figure(const std::string &path, const Image &target, const Image &rec, const std::string &top,
                 const std::string &bottom);

/*
   Minimal CSV: no quoting, so no field may contain a comma (none of mine do).
   split_csv also eats a trailing \r (files saved on Windows) and keeps an empty last field.
   read_csv skips the header line, write_csv writes it and creates the folder if needed.
*/
std::vector<std::string> split_csv(std::string line);
std::vector<std::vector<std::string>> read_csv(const std::string &path);
void write_csv(const std::string &path, const std::string &header, const std::vector<std::vector<std::string>> &rows);

/*
   Collects named arrays into a NumPy .npz so plots.py and the video can read them.
   Images, vectors and tables (rows x cols) go in as float32, int vectors as int64, scalars as float64.
*/
class Npz {
public:
    explicit Npz(std::string path) : path_(std::move(path)) {}
    void add(const std::string &name, const Image &m);
    void add(const std::string &name, const std::vector<double> &v);
    void add(const std::string &name, const std::vector<std::vector<double>> &rows);
    void add(const std::string &name, const std::vector<int> &v);
    void add(const std::string &name, double v);

private:
    std::string path_;
    bool first_ = true;
    std::string mode();
};

}  // namespace sc
