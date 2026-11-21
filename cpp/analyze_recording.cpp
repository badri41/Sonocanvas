/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Score a real recording of a demo WAV: align it, decode it, and report AUC and SSIM.

  bin/analyze_recording take.m4a --word DA321 --dist 2 --device phone [--take 1] [--video]

Any format ffmpeg reads (a phone's .m4a, a .wav) is converted to 22050 Hz mono first. Writes a row
to results/acoustic.csv (re-running a take replaces its row), the image to figures/acoustic/, and
with --video also data/acoustic_video.npz for scene 7.

Alignment: the delay is the peak of the cross-correlation of the band-limited recording with the
encoded signal, d* = argmax_d |sum_n r[n+d]*x[n]|. decode() then sees exactly the clip.
*/
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <sstream>

#include <CLI/CLI.hpp>
#include <opencv2/imgproc.hpp>

#include "sonocanvas.hpp"

using namespace sc;
constexpr int SR = 22050;
constexpr double F_LO = 1000, F_HI = 8000;
constexpr double MIN_PEAK_RATIO = 40;  // correct takes gave 130-158 in simulation, the wrong word 19

// The recording as 22050 Hz mono, peak-normalised to 1, via ffmpeg.
Signal read_audio(const std::string &path)
{
    std::string tmp = (std::filesystem::temp_directory_path() / "sonocanvas_take.wav").string();
    std::string cmd = std::format("ffmpeg -loglevel error -y -i '{}' -ac 1 -ar {} '{}'", path, SR, tmp);
    if (std::system(cmd.c_str()) != 0) throw std::runtime_error("ffmpeg could not read " + path);
    int sr;
    Signal r = read_wav(tmp, &sr);
    std::filesystem::remove(tmp);
    double peak = 1e-12;
    for (double v : r) peak = std::max(peak, std::abs(v));
    for (double &v : r) v /= peak;
    return r;
}

// Keep only F_LO..F_HI: zero every other DFT bin of the whole signal and transform back.
Signal bandpass(const Signal &r)
{
    cv::Mat x(1, int(r.size()), CV_64F, const_cast<double *>(r.data())), X, y;
    cv::dft(x, X, cv::DFT_COMPLEX_OUTPUT);
    int L = int(r.size());
    for (int k = 0; k < L; ++k) {
        double f = std::min(k, L - k) * double(SR) / L;  // bins above L/2 are negative frequencies
        if (f < F_LO || f > F_HI) X.at<cv::Vec2d>(0, k) = {0, 0};
    }
    cv::dft(X, y, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return Signal(y.ptr<double>(), y.ptr<double>() + L);
}

// |cross-correlation| of r with x at every delay where x fits inside r.
std::vector<float> correlate(const Signal &r, const Signal &x)
{
    cv::Mat rf, xf, c;
    cv::Mat(1, int(r.size()), CV_64F, const_cast<double *>(r.data())).convertTo(rf, CV_32F);
    cv::Mat(1, int(x.size()), CV_64F, const_cast<double *>(x.data())).convertTo(xf, CV_32F);
    cv::matchTemplate(rf, xf, c, cv::TM_CCORR);
    c = cv::abs(c);
    return std::vector<float>(c.ptr<float>(), c.ptr<float>() + c.cols);
}

// In-band SNR from the room noise before and after the clip: 10*log10(P_clip/P_noise - 1).
double snr_db(const Signal &b, size_t d, size_t n)
{
    size_t guard = SR / 10;  // skip the clip's onset and any tail
    double clip = 0, noise = 0;
    size_t quiet = 0;
    for (size_t i = d; i < d + n; ++i) clip += b[i] * b[i];
    for (size_t i = 0; i + guard < d; ++i, ++quiet) noise += b[i] * b[i];
    for (size_t i = d + n + guard; i < b.size(); ++i, ++quiet) noise += b[i] * b[i];
    if (quiet < SR / 2) return NAN;
    return 10 * std::log10(std::max((clip / n) / (noise / quiet) - 1, 1e-6));
}

// Add or replace this take's row in results/acoustic.csv, sorted by word, distance, device, take.
void save_row(const std::vector<std::string> &row)
{
    const std::string path = "results/acoustic.csv";
    std::vector<std::vector<std::string>> rows;
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        std::stringstream ss(line);
        for (std::string x; std::getline(ss, x, ',');) f.push_back(x);
        if (!std::equal(f.begin(), f.begin() + 4, row.begin())) rows.push_back(f);
    }
    rows.push_back(row);
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
        return std::tuple(a[0], std::stod(a[1]), a[2], a[3]) < std::tuple(b[0], std::stod(b[1]), b[2], b[3]);
    });
    std::filesystem::create_directories("results");
    std::ofstream f(path);
    f << "word,dist_m,device,take,auc,ssim,psnr_db,delay_s,peak_ratio,snr_db\n";
    for (const auto &r : rows) {
        for (size_t i = 0; i < r.size(); ++i) f << (i ? "," : "") << r[i];
        f << "\n";
    }
}

int main(int argc, char **argv)
{
    CLI::App app{"Score a recording of a demo WAV played through a speaker."};
    std::string recording, word = "DA321", device = "phone";
    double dist;
    int take = 1;
    bool video = false;
    app.add_option("recording", recording, "the recording (any format ffmpeg reads)")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--word", word, "the word in the WAV that was played");
    app.add_option("--dist", dist, "speaker-to-microphone distance in metres")->required();
    app.add_option("--device", device, "recording device, e.g. phone or mac");
    app.add_option("--take", take, "take number");
    app.add_flag("--video", video, "use this take in scene 7 of the video");
    CLI11_PARSE(app, argc, argv);

    Encoded e = encode(preprocess(render_text(word), "edges"));  // exactly the WAVs in demo/
    Signal r = read_audio(recording);
    if (r.size() < e.x.size()) {
        std::println(stderr, "recording is {:.1f} s but the clip is {:.1f} s: it cannot contain it",
                     r.size() / double(SR), e.x.size() / double(SR));
        return 1;
    }
    Signal b = bandpass(r);
    std::vector<float> c = correlate(b, e.x);
    size_t d = std::max_element(c.begin(), c.end()) - c.begin();
    std::vector<float> sorted = c;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    double ratio = c[d] / sorted[sorted.size() / 2];  // peak over median
    if (ratio < MIN_PEAK_RATIO)
        std::println("WARNING: weak correlation peak (ratio {:.1f} < {}). Check that the word and file match; "
                     "these scores may be meaningless.",
                     ratio, MIN_PEAK_RATIO);

    Signal y(r.begin() + d, r.begin() + d + e.x.size());
    Image rec = decode(y);
    double u = auc(e.target, rec), s = ssim(e.target, rec), q = psnr(e.target, rec), snr = snr_db(b, d, e.x.size());
    save_row({word, std::format("{}", dist), device, std::format("{}", take), std::format("{:.4f}", u),
              std::format("{:.4f}", s), std::format("{:.2f}", q), std::format("{:.3f}", d / double(SR)),
              std::format("{:.1f}", ratio), std::format("{:.1f}", snr)});

    std::string name = std::format("{}_{}m_{}_{}", word, dist, device, take);
    std::filesystem::create_directories("figures/acoustic");
    save_figure("figures/acoustic/" + name + ".png", e.target, rec, "target",
                std::format("recorded at {} m, {}: AUC {:.4f}, SSIM {:.3f}", dist, device, u, s));
    std::println(
        "{}: delay {:.3f} s, peak ratio {:.1f}, in-band SNR {:.1f} dB, AUC {:.4f}, SSIM {:.3f}, PSNR {:.1f} dB", name,
        d / double(SR), ratio, snr, u, s, q);
    std::println("wrote results/acoustic.csv and figures/acoustic/{}.png", name);
    if (video) {
        Npz out("data/acoustic_video.npz");
        out.add("x", y);
        out.add("rec", rec);
        out.add("target", e.target);
        out.add("dist", dist);
        std::println("wrote data/acoustic_video.npz: scene 7 will use this take");
    }
}
