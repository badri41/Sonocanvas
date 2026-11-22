/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

The whole SonoCanvas library, in the same order as sonocanvas.hpp: windows and STFT, Griffin-Lim,
picture -> magnitude, images and text, scores, encoder and decoder, files.

I wrote the STFT framing, the overlap-add and Griffin-Lim by hand, since that's the part of the method
I want visible. The rest leans on libraries: FFT, filters and text are OpenCV, WAV is libsndfile,
.npz is cnpy.
*/
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include <opencv2/freetype.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sndfile.h>

#include "sonocanvas.hpp"
#include "third_party/cnpy.h"

#ifndef SONOCANVAS_FONT
#define SONOCANVAS_FONT "src/fonts/DejaVuSans-Bold.ttf"
#endif

namespace sc {

std::vector<double> window(const std::string &name, int N)
{
    std::vector<double> w(N);
    for (int n = 0; n < N; ++n) {
        double t = 2 * std::numbers::pi * n / N;
        if (name == "hann") w[n] = 0.5 - 0.5 * std::cos(t);
        else if (name == "hamming") w[n] = 0.54 - 0.46 * std::cos(t);
        else if (name == "blackman") w[n] = 0.42 - 0.5 * std::cos(t) + 0.08 * std::cos(2 * t);
        else if (name == "rect") w[n] = 1;
        else throw std::invalid_argument("unknown window " + name);
    }
    return w;
}

/*
   Pad, cut into windowed frames (one per row), DFT every row, keep bins 0..N/2 and transpose
   so it comes out as bins x frames.
*/
Spectrum stft(const Signal &x, int N, int H, const std::vector<double> &w)
{
    Signal xp(x.size() + N, 0.0);
    std::copy(x.begin(), x.end(), xp.begin() + N / 2);
    int M = 1 + int(xp.size() - N) / H;

    cv::Mat frames(M, N, CV_64F);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) frames.at<double>(m, n) = xp[m * H + n] * w[n];

    cv::Mat F;
    cv::dft(frames, F, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    return Spectrum(F.colRange(0, N / 2 + 1).t());
}

/*
   Every frame goes back through the inverse DFT, gets multiplied by the window again and added
   at its place. At the end I divide by how much window landed on each sample. Skip that division
   and the output wobbles at the hop rate.
*/
Signal istft(const Spectrum &X, int N, int H, const std::vector<double> &w)
{
    int M = X.cols;
    // NOTE: cv::dft wants a real signal's half spectrum in this weird packed layout,
    // [Re_0, Re_1, Im_1, ..., Re_{N/2}]. Ugly, but then one inverse DFT per row hands back the
    // real frame, already divided by N.
    cv::Mat packed(M, N, CV_64F);
    for (int m = 0; m < M; ++m) {
        double *row = packed.ptr<double>(m);
        row[0] = X(0, m).real();
        for (int k = 1; k < N / 2; ++k) {
            row[2 * k - 1] = X(k, m).real();
            row[2 * k] = X(k, m).imag();
        }
        row[N - 1] = X(N / 2, m).real();
    }
    cv::Mat frames;
    cv::dft(packed, frames, cv::DFT_INVERSE | cv::DFT_ROWS | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    Signal y(N + H * (M - 1), 0.0), wsum(y.size(), 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            y[m * H + n] += frames.at<double>(m, n) * w[n];
            wsum[m * H + n] += w[n] * w[n];
        }
    for (size_t i = 0; i < y.size(); ++i) y[i] /= std::max(wsum[i], 1e-8);
    return Signal(y.begin() + N / 2, y.end() - N / 2);
}

Image magnitude(const Spectrum &X)
{
    Image mag(X.size());
    for (int k = 0; k < X.rows; ++k)
        for (int m = 0; m < X.cols; ++m) mag(k, m) = std::abs(X(k, m));
    return mag;
}

/*
   Start with S and some phase. Every iteration I do ISTFT then STFT, which snaps X to the closest
   thing that is an actual sound. Keep that phase, throw away its magnitude, put S back.
   Where |Y| is exactly 0 there's no phase to keep, so I just use phase 0 there.
*/
Phase_Result griffin_lim(const Image &S, int N, int H, const std::vector<double> &w, int n_iter,
                         const std::string &init, unsigned seed)
{
    if (init != "random" && init != "zero") throw std::invalid_argument("unknown init " + init);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    Spectrum X(S.size());
    for (int k = 0; k < S.rows; ++k)
        for (int m = 0; m < S.cols; ++m)
            X(k, m) = init == "random" ? std::polar(S(k, m), 2 * std::numbers::pi * U(rng))
                                       : std::complex<double>(S(k, m), 0.0);

    Phase_Result out;
    for (int i = 0; i < n_iter; ++i) {
        Spectrum Y = stft(istft(X, N, H, w), N, H, w).colRange(0, S.cols);
        out.errs.push_back(spectral_convergence(Y, S));
        for (int k = 0; k < S.rows; ++k)
            for (int m = 0; m < S.cols; ++m) {
                double a = std::abs(Y(k, m));
                X(k, m) = a > 0 ? S(k, m) * Y(k, m) / a : std::complex<double>(S(k, m), 0.0);
            }
    }
    out.x = istft(X, N, H, w);
    return out;
}

double mel(double f)
{
    return 2595 * std::log10(1 + f / 700);
}

/*
   Where frequency f sits between f_lo (0) and f_hi (1), on a linear or a mel axis.
*/
static double position(double f, const Params &p)
{
    if (p.scale == "mel") return (mel(f) - mel(p.f_lo)) / (mel(p.f_hi) - mel(p.f_lo));
    if (p.scale == "linear") return (f - p.f_lo) / (p.f_hi - p.f_lo);
    throw std::invalid_argument("unknown scale " + p.scale);
}

cv::Range band(const Params &p)
{
    int first = -1, last = -1;
    for (int k = 0; k <= p.n_fft / 2; ++k) {
        double q = position(k * double(p.sr) / p.n_fft, p);
        if (q >= 0 && q <= 1) {
            if (first < 0) first = k;
            last = k;
        }
    }
    return {first, last + 1};
}

/*
   I flip the picture first so row 0 is the lowest frequency, same as the bins. That way the bin
   index and the row index run in the same direction.
*/
Image image_to_mag(const Image &img, const Params &p)
{
    Image im = resize(img, img.rows, p.frames());
    cv::flip(im, im, 0);

    Image S = Image::zeros(p.n_fft / 2 + 1, p.frames());
    cv::Range b = band(p);
    for (int k = b.start; k < b.end; ++k) {
        double r = position(k * double(p.sr) / p.n_fft, p) * (im.rows - 1);
        int r0 = std::min(int(r), im.rows - 2);
        double t = r - r0;
        cv::Mat dst = S.row(k);
        cv::addWeighted(im.row(r0), 1 - t, im.row(r0 + 1), t, 0, dst);
    }
    return S;
}

Image load(const std::string &path)
{
    cv::Mat m = cv::imread(path, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
    if (m.empty()) throw std::runtime_error("cannot read image " + path);
    Image out;
    m.convertTo(out, CV_64F, 1.0 / (m.depth() == CV_16U ? 65535 : 255));
    return out;
}

/*
   Draws the text white at `px` pixels on a big black canvas and crops it to the ink, so I know
   exactly how wide the letters are. Advance widths include side bearings, so the pixels are more exact.
   The font is loaded once and kept; it throws if the file is missing.
*/
static cv::Mat ink(const std::string &text, int px)
{
    static cv::Ptr<cv::freetype::FreeType2> ft = [] {
        auto f = cv::freetype::createFreeType2();
        f->loadFontData(SONOCANVAS_FONT, 0);
        return f;
    }();
    cv::Mat big = cv::Mat::zeros(2 * px, px * int(text.size() + 2), CV_8U);
    ft->putText(big, text, {px / 2, 3 * px / 2}, px, cv::Scalar(255), -1, cv::LINE_AA, true);
    cv::Rect box = cv::boundingRect(big);
    return box.empty() ? cv::Mat() : big(box).clone();
}

Image render_text(const std::string &text, int w, int h, bool fit)
{
    cv::Mat t = ink(text, 170);
    if (!t.empty() && t.cols > 0.92 * w) {
        if (fit) t = ink(text, int(170 * 0.92 * w / t.cols));
        else w = int(std::ceil(t.cols / 0.92));
    }
    cv::Mat canvas = cv::Mat::zeros(h, w, CV_8U);
    if (!t.empty()) t.copyTo(canvas(cv::Rect((w - t.cols) / 2, (h - t.rows) / 2, t.cols, t.rows)));
    Image out;
    canvas.convertTo(out, CV_64F, 1.0 / 255);
    return out;
}

double text_duration(const Image &img, int w, double dur)
{
    return dur * std::max(1.0, double(img.cols) / w);
}

/*
   A flat image normalises to all zeros, which main() then rejects as "nothing to encode".
*/
Image preprocess(const Image &img, const std::string &mode, double gamma, int dilate)
{
    Image out;
    cv::normalize(img, out, 0, 1, cv::NORM_MINMAX);
    cv::pow(out, gamma, out);
    if (mode == "raw") return out;
    if (mode != "edges") throw std::invalid_argument("unknown mode " + mode);

    Image gx, gy, g;
    cv::Sobel(out, gx, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_REFLECT);
    cv::Sobel(out, gy, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_REFLECT);
    cv::magnitude(gx, gy, g);
    double gmax;
    cv::minMaxLoc(g, nullptr, &gmax);
    cv::Mat edges = g > 0.25 * gmax;
    if (dilate > 0) cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_CROSS, {3, 3}), {-1, -1}, dilate);
    edges.convertTo(out, CV_64F, 1.0 / 255);
    return out;
}

/*
   1-D Gaussian kernel, radius floor(4*sigma + 1/2). sigma = 0 means no blur at all.
*/
static cv::Mat gaussian(double sigma)
{
    if (sigma <= 0) return cv::Mat::ones(1, 1, CV_64F);
    return cv::getGaussianKernel(2 * int(4 * sigma + 0.5) + 1, sigma, CV_64F);
}

Image resize(const Image &img, int rows, int cols)
{
    double sy = std::max(0.0, (double(img.rows) / rows - 1) / 2);
    double sx = std::max(0.0, (double(img.cols) / cols - 1) / 2);
    Image blurred, out;
    cv::sepFilter2D(img, blurred, CV_64F, gaussian(sx), gaussian(sy), {-1, -1}, 0, cv::BORDER_REFLECT_101);
    cv::resize(blurred, out, {cols, rows}, 0, 0, cv::INTER_LINEAR);
    return out;
}

/*
   Means, variances and covariance from 7x7 box blurs. The 49/48 turns them into sample
   (co)variances, same as scikit-image, so my numbers line up with theirs.
*/
Image ssim_map(const Image &a, const Image &b)
{
    const double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03, n = 49.0 / 48.0;
    auto mean = [](const Image &m) {
        Image out;
        cv::blur(m, out, {7, 7}, {-1, -1}, cv::BORDER_REFLECT);
        return out;
    };
    Image ua = mean(a), ub = mean(b);
    Image vaa = n * (mean(a.mul(a)) - ua.mul(ua));
    Image vbb = n * (mean(b.mul(b)) - ub.mul(ub));
    Image vab = n * (mean(a.mul(b)) - ua.mul(ub));
    Image num = (2 * ua.mul(ub) + C1).mul(2 * vab + C2);
    Image den = (ua.mul(ua) + ub.mul(ub) + C1).mul(vaa + vbb + C2);
    return num / den;
}

double ssim(const Image &a, const Image &b)
{
    Image m = ssim_map(a, b);
    return cv::mean(m(cv::Rect(3, 3, m.cols - 6, m.rows - 6)))[0];
}

double psnr(const Image &a, const Image &b)
{
    double mse = cv::norm(a, b, cv::NORM_L2SQR) / double(a.total());
    return 10 * std::log10(1.0 / mse);
}

/*
   Mann-Whitney: rank all pixels of rec (ties get their average rank), add up the ranks of the
   stroke pixels, and AUC = (rank sum - n1*(n1 + 1)/2) / (n1*n0). Same as comparing every
   stroke/background pair, but O(n log n) instead of O(n^2). With 140k pixels the pairwise
   version would be billions of comparisons.
*/
double auc(const Image &target, const Image &rec)
{
    std::vector<double> t(target.begin(), target.end()), r(rec.begin(), rec.end());
    size_t n = r.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) { return r[i] < r[j]; });

    std::vector<double> rank(n);
    for (size_t i = 0; i < n;) {
        size_t j = i;
        while (j < n && r[order[j]] == r[order[i]]) ++j;
        for (size_t k = i; k < j; ++k) rank[order[k]] = (i + 1 + j) / 2.0;
        i = j;
    }
    double n1 = 0, sum = 0;
    for (size_t i = 0; i < n; ++i)
        if (t[i] > 0.5) {
            ++n1;
            sum += rank[i];
        }
    double n0 = n - n1;
    return (sum - n1 * (n1 + 1) / 2) / (n1 * n0);
}

double spectral_convergence(const Spectrum &Y, const Image &S)
{
    return cv::norm(magnitude(Y), S) / cv::norm(S);
}

Image band_image(const Image &mag, const Params &p)
{
    Image out;
    cv::flip(mag.rowRange(band(p)), out, 0);
    double mx;
    cv::minMaxLoc(out, nullptr, &mx);
    return out / mx;
}

/*
   I scale the waveform so its peak is 0.9: it never clips, and there's headroom for playback processing.
*/
Encoded encode(const Image &img, const Params &p, int n_iter, const std::string &init, unsigned seed)
{
    std::vector<double> w = window(p.window, p.n_fft);
    Encoded e;
    e.S = image_to_mag(img, p);
    Phase_Result gl = griffin_lim(e.S, p.n_fft, p.hop, w, n_iter, init, seed);
    double peak = 0;
    for (double v : gl.x) peak = std::max(peak, std::abs(v));
    for (double &v : gl.x) v *= 0.9 / peak;
    e.x = std::move(gl.x);
    e.errs = std::move(gl.errs);
    e.target = band_image(e.S, p);
    return e;
}

/*
   A WAV that is not exactly the encoder's length (a recording, say) gives a different number of
   frames, so I resize onto the target grid: band bins x M frames. For my own WAVs it's a no-op.
*/
Image decode(const Signal &x, const Params &p)
{
    Image rec = band_image(magnitude(stft(x, p.n_fft, p.hop, window(p.window, p.n_fft))), p);
    return resize(rec, rec.rows, p.frames());
}

void write_wav(const std::string &path, const Signal &x, int sr)
{
    SF_INFO info{};
    info.samplerate = sr;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE *f = sf_open(path.c_str(), SFM_WRITE, &info);
    if (!f) throw std::runtime_error("cannot write " + path + ": " + sf_strerror(nullptr));
    sf_write_double(f, x.data(), sf_count_t(x.size()));
    sf_close(f);
}

Signal read_wav(const std::string &path, int *sr)
{
    SF_INFO info{};
    SNDFILE *f = sf_open(path.c_str(), SFM_READ, &info);
    if (!f) throw std::runtime_error("cannot read " + path + ": " + sf_strerror(nullptr));
    std::vector<double> frames(size_t(info.frames) * info.channels);
    sf_readf_double(f, frames.data(), info.frames);
    sf_close(f);
    Signal x(info.frames, 0.0);
    for (sf_count_t i = 0; i < info.frames; ++i)
        for (int c = 0; c < info.channels; ++c) x[i] += frames[i * info.channels + c] / info.channels;
    *sr = info.samplerate;
    return x;
}

/*
   One panel of the answer key: the image in magma with a title above it.
   900 px wide for 5 s of audio, wider for longer clips.
*/
static cv::Mat panel(const Image &img, const std::string &title)
{
    int w = std::max(900, int(900.0 * img.cols / 431)), h = 300;
    cv::Mat u8, color;
    img.convertTo(u8, CV_8U, 255);
    cv::resize(u8, u8, {w, h}, 0, 0, cv::INTER_LINEAR);
    cv::applyColorMap(u8, color, cv::COLORMAP_MAGMA);
    cv::Mat out(h + 40, w, CV_8UC3, cv::Scalar(255, 255, 255));
    color.copyTo(out(cv::Rect(0, 40, w, h)));
    cv::putText(out, title, {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return out;
}

void save_figure(const std::string &path, const Image &target, const Image &rec, const std::string &top,
                 const std::string &bottom)
{
    cv::Mat fig;
    cv::vconcat(panel(target, top), panel(rec, bottom), fig);
    if (!cv::imwrite(path, fig)) throw std::runtime_error("cannot write " + path);
}

/*
   Creates the folder a file is about to go into, if it isn't there yet.
*/
static void make_parent(const std::string &path)
{
    auto dir = std::filesystem::path(path).parent_path();
    if (!dir.empty()) std::filesystem::create_directories(dir);
}

std::vector<std::string> split_csv(std::string line)
{
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> f;
    std::stringstream ss(line);
    for (std::string x; std::getline(ss, x, ',');) f.push_back(x);
    if (!line.empty() && line.back() == ',') f.push_back("");
    return f;
}

std::vector<std::vector<std::string>> read_csv(const std::string &path)
{
    std::vector<std::vector<std::string>> rows;
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) rows.push_back(split_csv(line));
    return rows;
}

void write_csv(const std::string &path, const std::string &header, const std::vector<std::vector<std::string>> &rows)
{
    make_parent(path);
    std::ofstream f(path);
    f << header << "\n";
    for (const auto &r : rows) {
        for (size_t i = 0; i < r.size(); ++i) f << (i ? "," : "") << r[i];
        f << "\n";
    }
}

/*
   The first array creates the file ("w"), every one after that appends ("a"). cnpy segfaults if it
   can't open the file, so I make sure the folder exists before the first write.
*/
std::string Npz::mode()
{
    if (!first_) return "a";
    make_parent(path_);
    first_ = false;
    return "w";
}

void Npz::add(const std::string &name, const Image &m)
{
    cv::Mat f;
    m.convertTo(f, CV_32F);
    f = f.clone();
    cnpy::npz_save(path_, name, f.ptr<float>(), {size_t(f.rows), size_t(f.cols)}, mode());
}

void Npz::add(const std::string &name, const std::vector<double> &v)
{
    std::vector<float> f(v.begin(), v.end());
    cnpy::npz_save(path_, name, f.data(), {f.size()}, mode());
}

/*
   Rows of equal length, stored row after row as one rows x cols array.
*/
void Npz::add(const std::string &name, const std::vector<std::vector<double>> &rows)
{
    std::vector<float> f;
    for (const auto &r : rows) f.insert(f.end(), r.begin(), r.end());
    cnpy::npz_save(path_, name, f.data(), {rows.size(), rows.empty() ? 0 : rows[0].size()}, mode());
}

void Npz::add(const std::string &name, const std::vector<int> &v)
{
    std::vector<long long> l(v.begin(), v.end());
    cnpy::npz_save(path_, name, l.data(), {l.size()}, mode());
}

void Npz::add(const std::string &name, double v)
{
    cnpy::npz_save(path_, name, &v, std::vector<size_t>{}, mode());
}

}  // namespace sc
