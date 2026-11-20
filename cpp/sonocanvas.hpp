// SonoCanvas: embed an image in the STFT magnitude of an audio signal.
//
//   encode: image -> target magnitude S -> Griffin-Lim phase -> weighted overlap-add -> x
//   decode: x -> |STFT(x)| on the band -> image on the target grid
//
// Conventions: an Image is rows x cols with values in [0, 1] and row 0 at the top.
// A Spectrum or magnitude is bins x frames (bin 0 = 0 Hz). Audio is a vector of doubles.
#pragma once

#include <complex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace sc {

using Signal = std::vector<double>;
using Image = cv::Mat_<double>;
using Spectrum = cv::Mat_<std::complex<double>>;

// Parameters shared by the encoder and the decoder. The defaults are the paper's.
struct Params {
    int n_fft = 1024;              // window length N
    int hop = 256;                 // hop H
    double dur = 5.0;              // clip length d in seconds
    double f_lo = 1000, f_hi = 8000;
    int sr = 22050;                // sampling rate f_s
    std::string scale = "linear";  // frequency axis of the picture: "linear" or "mel"
    std::string window = "hann";   // "hann", "hamming", "blackman" or "rect"
    int frames() const { return 1 + int(dur * sr) / hop; }  // M = 1 + \lfloor d f_s \rfloor / H
};

// ---- stft.cpp: analysis and synthesis ----------------------------------------------------

// Periodic window of length N, n = 0..N-1:
//   hann     w[n] = 1/2 - 1/2 \cos(2\pi n/N)
//   hamming  w[n] = 0.54 - 0.46 \cos(2\pi n/N)
//   blackman w[n] = 0.42 - 0.5 \cos(2\pi n/N) + 0.08 \cos(4\pi n/N)
//   rect     w[n] = 1
std::vector<double> window(const std::string& name, int N);

// Centre-padded STFT, (N/2+1) x M. x is zero-padded by N/2 on both sides, then
//   X[k,m] = \sum_{n=0}^{N-1} x[n + mH] w[n] e^{-j 2\pi k n / N}.
Spectrum stft(const Signal& x, int n_fft, int hop, const std::vector<double>& win);

// Weighted overlap-add, the least-squares inverse (Griffin & Lim 1984):
//   y_m = IDFT(X[:,m]),  \hat x[n] = \sum_m w[n-mH] y_m[n-mH] / \sum_m w^2[n-mH].
// Returns H (M-1) samples (the N/2 centre padding removed).
Signal istft(const Spectrum& X, int n_fft, int hop, const std::vector<double>& win);

// |X| element-wise.
Image magnitude(const Spectrum& X);

// ---- phase.cpp: phase recovery -----------------------------------------------------------

struct PhaseResult {
    Signal x;                  // ISTFT(X^{(n_iter)})
    std::vector<double> errs;  // errs[i] = spectral convergence of STFT(ISTFT(X^{(i)})) against S
};

// Griffin-Lim: X^{(i+1)} = S \cdot Y/|Y| with Y = STFT(ISTFT(X^{(i)})).
// init "random": X^{(0)} = S e^{j 2\pi U}, U ~ Uniform[0,1) (seeded); init "zero": X^{(0)} = S.
PhaseResult griffin_lim(const Image& S, int n_fft, int hop, const std::vector<double>& win,
                        int n_iter = 60, const std::string& init = "random", unsigned seed = 0);

// ---- freq_map.cpp: image -> target magnitude ---------------------------------------------

// m(f) = 2595 \log_{10}(1 + f/700).
double mel(double f);

// Bins with f_lo <= f_k <= f_hi (f_k = k f_s / N) on the chosen axis, as a row range of S.
cv::Range band(const Params& p);

// Target magnitude S, (N/2+1) x M. Image columns are resampled to the M frames. Bin k gets the
// normalised position p_k = (f_k - f_lo)/(f_hi - f_lo) (linear; the same on mel values for "mel"),
// and S[k,m] linearly interpolates column m at p_k, bottom row = f_lo, top row = f_hi.
Image image_to_mag(const Image& img, const Params& p);

// ---- image_prep.cpp: images and text -----------------------------------------------------

// Grayscale image file as [0, 1] (8- or 16-bit).
Image load(const std::string& path);

// White DejaVu Sans Bold at 170 px, centred on a black w x h canvas. Text wider than 0.92 w is
// shrunk to fit (fit = true), or the canvas is widened to w' = \lceil width / 0.92 \rceil (fit = false).
Image render_text(const std::string& text, int w = 900, int h = 260, bool fit = true);

// Clip length that keeps a widened text canvas at 900 px per 5 s: d \cdot \max(1, w'/w).
double text_duration(const Image& img, int w = 900, double dur = 5.0);

// Contrast stretch I' = (I - \min I)/(\max I - \min I), gamma I'' = I'^\gamma. For mode "edges":
// Sobel magnitude G, threshold E = [G > 0.25 \max G], then dilate `dilate` times with a 3x3 cross.
Image preprocess(const Image& img, const std::string& mode = "raw", double gamma = 1.0, int dilate = 1);

// Anti-aliased resize: Gaussian blur with \sigma = \max(0, (s - 1)/2) per axis (s = shrink
// factor), then bilinear interpolation. The same recipe as scikit-image's resize.
Image resize(const Image& img, int rows, int cols);

// ---- metrics.cpp: scores -----------------------------------------------------------------

// SSIM map (Wang et al. 2004) with a 7x7 uniform window and sample (co)variances:
//   \frac{(2\mu_a\mu_b + C_1)(2\sigma_{ab} + C_2)}{(\mu_a^2+\mu_b^2+C_1)(\sigma_a^2+\sigma_b^2+C_2)},
// C_1 = 0.01^2, C_2 = 0.03^2. ssim() is its mean with the 3-pixel border excluded.
Image ssim_map(const Image& a, const Image& b);
double ssim(const Image& a, const Image& b);

// PSNR = 10 \log_{10}(1 / MSE) dB.
double psnr(const Image& a, const Image& b);

// Stroke/background AUC as the Mann-Whitney statistic:
//   A = P(\hat I_s > \hat I_b) + 1/2 P(\hat I_s = \hat I_b), s a stroke pixel (T > 0.5), b background.
double auc(const Image& target, const Image& rec);

// SC = \| |Y| - S \|_F / \|S\|_F.
double spectral_convergence(const Spectrum& Y, const Image& S);

// ---- sonocanvas.cpp: encoder and decoder -------------------------------------------------

struct Encoded {
    Signal x;                  // waveform, peak-normalised to 0.9
    Image S;                   // full target magnitude, (N/2+1) x M
    Image target;              // S on the band, flipped (row 0 = f_hi), divided by its max
    std::vector<double> errs;  // Griffin-Lim spectral convergence per iteration
};

// A bins x frames magnitude on the band, flipped so row 0 is f_hi, divided by its maximum.
Image band_image(const Image& mag, const Params& p);

Encoded encode(const Image& img, const Params& p = {}, int n_iter = 60,
               const std::string& init = "random", unsigned seed = 0);

// \hat I = |STFT(x)|[band] / max, flipped, on the same grid as Encoded::target.
// Uses only the shared parameters, never encoder state.
Image decode(const Signal& x, const Params& p = {});

// ---- io.cpp: files -----------------------------------------------------------------------

void write_wav(const std::string& path, const Signal& x, int sr);
Signal read_wav(const std::string& path, int* sr);  // mixed down to mono

// Two magma panels, target above decoded, with a title over each: the "answer key" PNG.
void save_figure(const std::string& path, const Image& target, const Image& rec,
                 const std::string& top, const std::string& bottom);

// Collects named arrays and writes them as a NumPy .npz (read by plots.py and the video).
class Npz {
public:
    explicit Npz(std::string path) : path_(std::move(path)) {}
    void add(const std::string& name, const Image& m);             // float32, rows x cols
    void add(const std::string& name, const std::vector<double>& v,
             std::vector<size_t> shape = {});                     // float32; shape defaults to 1-D
    void add(const std::string& name, const std::vector<int>& v);  // int64, 1-D
    void add(const std::string& name, double v);                   // float64 scalar
private:
    std::string path_;
    bool first_ = true;
    std::string mode();
};

}  // namespace sc
