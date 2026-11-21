/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

The six experiments from the paper.

  bin/experiments                     # all six
  bin/experiments noise_robustness    # one (or several) by name

Every experiment writes data/<name>.npz (plots.py and make_numbers read those) and replaces its own
rows in results/table.csv: experiment, image, param, AUC, AUC std, SSIM, SSIM std, PSNR.
Every number is a mean (and std) over five seeds, so a single lucky seed can't decide a result.

Three test pictures: text is DA321, line is the horse silhouette (its edges make a line drawing),
photo is Chelsea the cat. Both images are CC0 and live in data/images/. Text and line go in as
"edges", the photo as "raw", unless the experiment is about exactly that.
*/
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <numbers>
#include <print>
#include <random>

#include <CLI/CLI.hpp>
#include <opencv2/imgproc.hpp>

#include "sonocanvas.hpp"

using namespace sc;
using Row = std::vector<std::string>;

const std::vector<unsigned> SEEDS = {0, 1, 2, 3, 4};
const std::vector<std::string> NAMES = {"text", "line", "photo"};
const std::map<std::string, std::string> MODE = {{"text", "edges"}, {"line", "edges"}, {"photo", "raw"}};
constexpr int SR = 22050;
constexpr double F_LO = 1000, F_HI = 8000, C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;

double mean(const std::vector<double> &v)
{
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

/*
   Population std, so I divide by n and not n - 1.
*/
double stdev(const std::vector<double> &v)
{
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
}

/*
   Mean of m over the pixels where mask is on. NaN if the mask is empty.
*/
double masked_mean(const Image &m, const cv::Mat &mask)
{
    return cv::countNonZero(mask) ? cv::mean(m, mask)[0] : NAN;
}

/*
   Clamps every pixel to [0, 1].
*/
Image clip01(const Image &m)
{
    return cv::max(cv::min(m, 1.0), 0.0);
}

/*
   One line of results/table.csv: means and stds over the seeds, AUC and SSIM to 4 places.
*/
Row row(const std::string &exp, const std::string &image, const std::string &param, const std::vector<double> &ssims,
        const std::vector<double> &psnrs, const std::vector<double> &aucs)
{
    return {exp,
            image,
            param,
            std::format("{:.4f}", mean(aucs)),
            std::format("{:.4f}", stdev(aucs)),
            std::format("{:.4f}", mean(ssims)),
            std::format("{:.4f}", stdev(ssims)),
            std::format("{:.2f}", mean(psnrs))};
}

/*
   The three test pictures, before any preprocessing.
*/
std::map<std::string, Image> base_images()
{
    return {{"text", render_text("DA321")},
            {"line", load("data/images/horse.png")},
            {"photo", load("data/images/chelsea.png")}};
}

/*
   SSIM, PSNR and AUC of every seed. add() scores one decoded picture against its target.
*/
struct Scores {
    std::vector<double> ssim, psnr, auc;
    void add(const Image &target, const Image &rec)
    {
        ssim.push_back(sc::ssim(target, rec));
        psnr.push_back(sc::psnr(target, rec));
        auc.push_back(sc::auc(target, rec));
    }
};
struct Trial {
    Scores own;
    std::map<std::string, Scores> ref;      // per reference grid
    std::map<std::string, Image> ref_rec;   // seed 0, resized onto each reference grid
    Encoded first;                          // seed 0
    Image rec;                              // seed 0
};

/*
   Encode and decode once per seed and score it. If you pass reference images, the decoded picture
   also gets resized onto each of their grids and scored there. I keep seed 0's stuff for the plots.
*/
Trial trial(const Image &img, const Params &p = {}, int n_iter = 60, const std::string &init = "random",
            const std::vector<unsigned> &seeds = SEEDS, const std::map<std::string, Image> &refs = {})
{
    Trial t;
    for (unsigned s : seeds) {
        Encoded e = encode(img, p, n_iter, init, s);
        Image rec = decode(e.x, p);
        t.own.add(e.target, rec);
        for (const auto &[k, ref] : refs) {
            Image r2 = clip01(resize(rec, ref.rows, ref.cols));
            t.ref[k].add(ref, r2);
            if (s == seeds[0]) t.ref_rec[k] = r2;
        }
        if (s == seeds[0]) {
            t.first = std::move(e);
            t.rec = rec;
        }
    }
    return t;
}

/*
   What SSIM does on the flat parts of the target (7x7 windows with sigma_T^2 < C2, border skipped).
   There the target has no texture, so any speckle in rec gets punished hard.
     flat_frac   how many windows are flat
     flat_var    mean sigma_R^2 of rec on them
     flat_lum    mean luminance term l on them
     flat_ssim   mean SSIM on them, rest_ssim on everything else
     flat_pred   mean of l*C2/(sigma_R^2 + C2), which is what SSIM should be on a flat target
*/
std::map<std::string, double> flat_stats(const Image &target, const Image &rec)
{
    Image smap = ssim_map(target, rec);
    auto avg = [](const Image &m) {
        Image o;
        cv::blur(m, o, {7, 7}, {-1, -1}, cv::BORDER_REFLECT);
        return o;
    };
    auto var = [&](const Image &m) { return Image((avg(m.mul(m)) - avg(m).mul(avg(m))) * 49.0 / 48.0); };
    Image mt = avg(target), mr = avg(rec), vr = var(rec);
    cv::Mat body = cv::Mat::zeros(target.size(), CV_8U);
    body(cv::Rect(3, 3, target.cols - 6, target.rows - 6)) = 255;
    cv::Mat flat = (var(target) < C2) & body;
    Image lum = (2 * mt.mul(mr) + C1) / (mt.mul(mt) + mr.mul(mr) + C1);
    Image pred = lum.mul(C2 / (vr + C2));
    return {{"flat_frac", double(cv::countNonZero(flat)) / cv::countNonZero(body)},
            {"flat_var", masked_mean(vr, flat)},
            {"flat_lum", masked_mean(lum, flat)},
            {"flat_ssim", masked_mean(smap, flat)},
            {"rest_ssim", masked_mean(smap, body & ~flat)},
            {"flat_pred", masked_mean(pred, flat)}};
}

/*
   Pulls apart what SSIM is actually measuring on a decoded picture.
     support        mean SSIM on the lit pixels grown by 3, so the black background doesn't count
     ls_gain        SSIM after the best gain g = <T,R>/<R,R>, undoing the scale from dividing by max
     cv             sigma/mu of rec where the target is lit, i.e. how strong the speckle is
     interior       mean SSIM where the whole 7x7 window is lit (sigma_T ~ 0)
     interior_pred  l*C2/(sigma_R^2 + C2) from the measured mu_R, sigma_R, the prediction for interior
   Thin edges have no fully lit window, so interior is NaN for them.
*/
std::map<std::string, double> diagnostics(const Image &target, const Image &rec)
{
    Image smap = ssim_map(target, rec);
    cv::Mat sup, core, lit = target > 0.5;
    cv::dilate(target > 0.01, sup, cv::getStructuringElement(cv::MORPH_CROSS, {3, 3}), {-1, -1}, 3);
    cv::erode(target > 0.99, core, cv::Mat::ones(7, 7, CV_8U), {-1, -1}, 1, cv::BORDER_CONSTANT, 0);
    double g = target.dot(rec) / rec.dot(rec);
    double interior = NAN, pred = NAN;
    if (cv::countNonZero(core)) {
        interior = masked_mean(smap, core);
        cv::Scalar mu, sd;
        cv::meanStdDev(rec, mu, sd, core);
        pred = (2 * mu[0] + C1) / (1 + mu[0] * mu[0] + C1) * C2 / (sd[0] * sd[0] + C2);
    }
    cv::Scalar lit_mu, lit_sd;
    cv::meanStdDev(rec, lit_mu, lit_sd, lit);
    double n = double(target.total());
    return {{"support", masked_mean(smap, sup)},
            {"interior", interior},
            {"interior_pred", pred},
            {"gain", g},
            {"ls_gain", ssim(target, clip01(g * rec))},
            {"cv", lit_sd[0] / lit_mu[0]},
            {"lit_frac", cv::countNonZero(lit) / n},
            {"support_frac", cv::countNonZero(sup) / n}};
}

/*
   Experiment 1: the filled picture vs only its edges.
*/
std::vector<Row> edges_vs_filled(std::map<std::string, Image> &imgs)
{
    std::vector<Row> rows;
    Npz out("data/edges_vs_filled.npz");
    for (const auto &n : NAMES)
        for (std::string mode : {"raw", "edges"}) {
            Trial t = trial(preprocess(imgs[n], mode));
            rows.push_back(row("edges_vs_filled", n, mode, t.own.ssim, t.own.psnr, t.own.auc));
            auto d = diagnostics(t.first.target, t.rec);
            for (std::string k : {"support", "ls_gain", "interior", "interior_pred"})
                if (std::isfinite(d[k]))
                    rows.push_back({"edges_vs_filled", n, mode + "/" + k, "", "", std::format("{:.4f}", d[k]), "", ""});
            std::string key = n + "_" + mode + "_";
            double energy = 0;
            for (double v : t.first.x) energy += v * v;
            out.add(key + "target", t.first.target);
            out.add(key + "rec", t.rec);
            out.add(key + "ssim", mean(t.own.ssim));
            out.add(key + "auc", mean(t.own.auc));
            out.add(key + "energy", energy);
            for (const auto &[k, v] : d) out.add(key + k, v);
        }
    return rows;
}

const std::vector<int> LENGTHS = {128, 256, 512, 1024, 2048, 4096};

/*
   Experiment 2: window length N. Careful with the scoring: every N makes a different target grid,
   so "own" scores can't be compared across N. "image" scores every result against the original picture on
   one fixed grid, and "half" does the same at half size to show the ranking doesn't depend on the grid.
*/
std::vector<Row> sweep_window_len(std::map<std::string, Image> &imgs)
{
    std::vector<Row> rows;
    Npz out("data/sweep_window_len.npz");
    out.add("lengths", LENGTHS);
    for (const auto &n : NAMES) {
        Image img = preprocess(imgs[n], MODE.at(n));
        Image full = resize(img, 260, 900);
        std::map<std::string, Image> refs = {{"image", full}, {"half", resize(img, 130, 450)}};
        std::map<std::string, std::vector<std::vector<double>>> ssims, aucs;
        for (int N : LENGTHS) {
            Params p;
            p.n_fft = N;
            p.hop = N / 4;
            Trial t = trial(img, p, 60, "random", SEEDS, refs);
            for (std::string k : {"own", "image", "half"}) {
                const Scores &s = k == "own" ? t.own : t.ref[k];
                rows.push_back(row("sweep_window_len", n, std::format("N={}/{}", N, k), s.ssim, s.psnr, s.auc));
                ssims[k].push_back(s.ssim);
                aucs[k].push_back(s.auc);
            }
            for (const auto &[k, v] : flat_stats(full, t.ref_rec["image"]))
                out.add(std::format("{}_{}_{}", n, k, N), v);
            for (const auto &[k, v] : flat_stats(t.first.target, t.rec))
                out.add(std::format("{}_own_{}_{}", n, k, N), v);
            if (n == "text") out.add(std::format("text_rec_{}", N), t.ref_rec["image"]);
        }
        for (std::string k : {"own", "image", "half"}) {
            out.add(n + "_" + k, ssims[k]);
            out.add(n + "_" + k + "_auc", aucs[k]);
        }
    }
    return rows;
}

const std::vector<std::string> WINDOWS = {"rect", "hann", "hamming", "blackman"};

struct Window_Spectrum {
    std::vector<double> bins, db;
    double sidelobe;
};
/*
   What the window looks like in frequency: |W(f)| in dB, zero padded 32x so I can see the lobes,
   for bins 0..12. Also the highest sidelobe, i.e. how much a pixel leaks into its neighbours.
*/
Window_Spectrum window_spectrum(const std::string &name, int N = 1024, int pad = 32)
{
    cv::Mat w = cv::Mat::zeros(1, N * pad, CV_64F), F;
    std::vector<double> win = window(name, N);
    for (int n = 0; n < N; ++n) w.at<double>(0, n) = win[n];
    cv::dft(w, F, cv::DFT_COMPLEX_OUTPUT);
    std::vector<double> mag(N * pad / 2 + 1), db(mag.size());
    for (size_t k = 0; k < mag.size(); ++k)
        mag[k] = std::abs(std::complex<double>(F.at<cv::Vec2d>(0, k)[0], F.at<cv::Vec2d>(0, k)[1]));
    double peak = *std::max_element(mag.begin(), mag.end());
    for (size_t k = 0; k < mag.size(); ++k) db[k] = 20 * std::log10(std::max(mag[k] / peak, 1e-8));
    // NOTE: the main lobe ends where the response starts going up again.
    size_t first_null = 0;
    while (first_null + 1 < db.size() && !(db[first_null + 1] > db[first_null])) ++first_null;
    if (first_null + 1 == db.size()) first_null = 0;
    Window_Spectrum ws{{}, {}, *std::max_element(db.begin() + first_null, db.end())};
    for (int i = 0; i < 12 * pad; ++i) {
        ws.bins.push_back(double(i) / pad);
        ws.db.push_back(db[i]);
    }
    return ws;
}

/*
   Experiment 3: window shape, rect vs hann vs hamming vs blackman.
*/
std::vector<Row> sweep_window_type(std::map<std::string, Image> &imgs)
{
    std::vector<Row> rows;
    Npz out("data/sweep_window_type.npz");
    for (const auto &w : WINDOWS) {
        Window_Spectrum ws = window_spectrum(w);
        out.add(w + "_bins", ws.bins);
        out.add(w + "_db", ws.db);
        out.add(w + "_sidelobe_db", ws.sidelobe);
        for (const auto &n : NAMES) {
            Params p;
            p.window = w;
            Trial t = trial(preprocess(imgs[n], MODE.at(n)), p);
            rows.push_back(row("sweep_window_type", n, w, t.own.ssim, t.own.psnr, t.own.auc));
            out.add(n + "_" + w + "_ssim", t.own.ssim);
            out.add(n + "_" + w + "_auc", t.own.auc);
            for (const auto &[k, v] : flat_stats(t.first.target, t.rec)) out.add(n + "_" + w + "_" + k, v);
            if (n == "text") out.add("text_" + w + "_rec", t.rec);
        }
    }
    return rows;
}

const std::vector<int> ITERS = {0, 1, 2, 5, 10, 20, 60, 100, 200};
constexpr int CURVE_LEN = 201;  // errs[i] is the error of X_i, so i = 0..200 needs 201 passes

/*
   Experiment 4: how many Griffin-Lim iterations do I actually need? n_iter = 0 is just random phase,
   x = ISTFT(S*exp(j*phi)) with no fixing at all.
*/
std::vector<Row> gl_iterations(std::map<std::string, Image> &imgs)
{
    std::vector<Row> rows;
    Npz out("data/gl_iterations.npz");
    out.add("iters", ITERS);
    for (const auto &n : NAMES) {
        Image img = preprocess(imgs[n], MODE.at(n));
        std::vector<std::vector<double>> curves;
        for (unsigned s : SEEDS) curves.push_back(trial(img, {}, CURVE_LEN, "random", {s}).first.errs);
        out.add(n + "_errs_random", curves);
        out.add(n + "_errs_zero", trial(img, {}, CURVE_LEN, "zero", {0}).first.errs);
        std::vector<std::vector<double>> ssim_r, auc_r;
        std::vector<double> ssim_z;
        for (int k : ITERS) {
            Trial r = trial(img, {}, k), z = trial(img, {}, k, "zero", {0});
            rows.push_back(
                row("gl_iterations", n, std::format("random/iter={}", k), r.own.ssim, r.own.psnr, r.own.auc));
            rows.push_back(row("gl_iterations", n, std::format("zero/iter={}", k), z.own.ssim, z.own.psnr, z.own.auc));
            ssim_r.push_back(r.own.ssim);
            auc_r.push_back(r.own.auc);
            ssim_z.push_back(z.own.ssim[0]);
        }
        out.add(n + "_ssim_random", ssim_r);
        out.add(n + "_ssim_zero", ssim_z);
        out.add(n + "_auc_random", auc_r);
    }
    return rows;
}

/*
   mel back to Hz: f(m) = 700*(10^(m/2595) - 1).
*/
double mel_inv(double m)
{
    return 700 * (std::pow(10.0, m / 2595) - 1);
}

/*
   Redraws a decoded picture the way a spectrogram app with a linear or mel axis would show it.
   Row r (0 at the bottom, 1 at the top) shows frequency f(r), interpolated between the band's bins.
*/
Image view(const Image &rec, const std::string &scale, int n_fft = 1024)
{
    double df = double(SR) / n_fft, f0 = std::ceil(F_LO / df) * df;
    Image out(rec.size());
    int R = rec.rows;
    for (int i = 0; i < R; ++i) {
        double r = double(i) / (R - 1);
        double f = scale == "mel" ? mel_inv(mel(F_LO) + r * (mel(F_HI) - mel(F_LO))) : F_LO + r * (F_HI - F_LO);
        double q = std::clamp((f - f0) / df, 0.0, double(R - 1));
        int q0 = std::min(int(q), R - 2);
        double t = q - q0;
        // NOTE: rec row 0 is the top (highest bin), so bin b is rec row R-1-b.
        cv::Mat dst = out.row(R - 1 - i);
        cv::addWeighted(rec.row(R - 1 - q0), 1 - t, rec.row(R - 2 - q0), t, 0, dst);
    }
    return clip01(out);
}

/*
   Experiment 5: encode on a linear or mel axis, then look at it on a linear or mel axis.
   Shows what goes wrong when the encoder and the app don't agree.
*/
std::vector<Row> mel_vs_linear(std::map<std::string, Image> &imgs)
{
    std::vector<Row> rows;
    Npz out("data/mel_vs_linear.npz");
    for (const auto &n : NAMES) {
        Image img = preprocess(imgs[n], MODE.at(n));
        for (std::string enc : {"linear", "mel"}) {
            Params p;
            p.scale = enc;
            Trial t = trial(img, p);
            Image ref = resize(img, t.first.target.rows, t.first.target.cols);
            rows.push_back(row("mel_vs_linear", n, "encode=" + enc + "/bin_grid", t.own.ssim, t.own.psnr, t.own.auc));
            for (std::string vw : {"linear", "mel"}) {
                Image v = view(t.rec, vw);
                double s = ssim(ref, v), q = psnr(ref, v), u = auc(ref, v);
                rows.push_back({"mel_vs_linear", n, "encode=" + enc + "/view=" + vw, std::format("{:.4f}", u), "",
                                std::format("{:.4f}", s), "", std::format("{:.2f}", q)});
                std::string key = n + "_" + enc + "_" + vw + "_";
                out.add(key + "img", v);
                out.add(key + "ssim", s);
                out.add(key + "auc", u);
            }
        }
    }
    return rows;
}

const std::vector<int> SNRS = {30, 20, 10, 5, 0};

/*
   Noise std that gives this SNR for x. SNR = 10*log10(sum x^2 / sum n^2).
*/
double sigma_for(const Signal &x, double snr)
{
    double p = 0;
    for (double v : x) p += v * v;
    return std::sqrt(p / x.size() / std::pow(10.0, snr / 10));
}

/*
   x plus white Gaussian noise with std sigma. Seeded, so every run gets the same noise.
*/
Signal noisy(const Signal &x, double sigma, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> g;
    Signal y = x;
    for (double &v : y) v += sigma * g(rng);
    return y;
}

/*
   Adds noise five times (seeds 100..104), decodes each one and scores it.
*/
Scores noisy_scores(const Signal &x, const Image &target, double sigma)
{
    Scores s;
    for (unsigned seed : SEEDS) {
        s.add(target, decode(noisy(x, sigma, 100 + seed)));
    }
    return s;
}

/*
   Experiment 6: add white noise and decode. "snr" keeps the SNR fixed for each signal. But a real
   speaker fixes the peak and the room fixes the noise, so "floor" gives text raw and edges the
   same noise sigma (set so raw sits at the given SNR). That's the fair comparison.
*/
std::vector<Row> noise_robustness(std::map<std::string, Image> &imgs)
{
    std::vector<Row> rows;
    Npz out("data/noise_robustness.npz");
    out.add("snrs", SNRS);
    std::map<std::string, Encoded> enc;
    for (const auto &n : NAMES)
        for (std::string mode : {"raw", "edges"}) enc[n + "/" + mode] = encode(preprocess(imgs[n], mode));
    auto series = [&](const std::string &n, const std::string &mode, const Signal &level_of, const std::string &tag,
                      const std::string &key) {
        const Encoded &e = enc[n + "/" + mode];
        std::vector<std::vector<double>> ss, us;
        for (int snr : SNRS) {
            Scores s = noisy_scores(e.x, e.target, sigma_for(level_of, snr));
            rows.push_back(row("noise_robustness", n, std::format("{}/{}={}", mode, tag, snr), s.ssim, s.psnr, s.auc));
            ss.push_back(s.ssim);
            us.push_back(s.auc);
        }
        out.add(key + "ssim", ss);
        out.add(key + "auc", us);
    };
    for (const auto &n : NAMES)
        for (std::string mode : {"raw", "edges"}) series(n, mode, enc[n + "/" + mode].x, "snr", n + "_" + mode + "_");
    for (int snr : SNRS) {
        const Signal &x = enc["text/edges"].x;
        out.add(std::format("text_edges_rec_{}", snr), decode(noisy(x, sigma_for(x, snr), 100)));
    }
    for (std::string mode : {"raw", "edges"})
        series("text", mode, enc["text/raw"].x, "floor", "text_" + mode + "_floor_");
    return rows;
}

using Experiment = std::function<std::vector<Row>(std::map<std::string, Image> &)>;
const std::vector<std::pair<std::string, Experiment>> EXPERIMENTS = {
    {"edges_vs_filled", edges_vs_filled},     {"sweep_window_len", sweep_window_len},
    {"sweep_window_type", sweep_window_type}, {"gl_iterations", gl_iterations},
    {"mel_vs_linear", mel_vs_linear},         {"noise_robustness", noise_robustness}};

/*
   Swaps in the rows of the experiments I just ran and keeps the rest, in experiment order.
*/
void update_table(const std::map<std::string, std::vector<Row>> &fresh)
{
    const std::string path = "results/table.csv";
    std::vector<Row> rows;
    for (Row &r : read_csv(path))
        if (!r.empty() && !fresh.contains(r[0])) rows.push_back(r);
    for (const auto &[name, rs] : fresh) rows.insert(rows.end(), rs.begin(), rs.end());
    auto order = [](const std::string &e) {
        for (size_t i = 0; i < EXPERIMENTS.size(); ++i)
            if (EXPERIMENTS[i].first == e) return i;
        return EXPERIMENTS.size();
    };
    std::stable_sort(rows.begin(), rows.end(), [&](const Row &a, const Row &b) { return order(a[0]) < order(b[0]); });
    write_csv(path, "experiment,image,param,auc,auc_std,ssim,ssim_std,psnr_db", rows);
}

int main(int argc, char **argv)
{
    CLI::App app{"Run the SonoCanvas experiments."};
    std::vector<std::string> names;
    std::vector<std::string> all;
    for (const auto &[n, f] : EXPERIMENTS) all.push_back(n);
    app.add_option("names", names, "experiments to run (default: all)")->check(CLI::IsMember(all));
    CLI11_PARSE(app, argc, argv);
    if (names.empty()) names = all;
    for (auto d : {"data", "results", "figures"}) std::filesystem::create_directories(d);

    auto imgs = base_images();
    std::map<std::string, std::vector<Row>> fresh;
    for (const auto &[name, run] : EXPERIMENTS) {
        if (std::find(names.begin(), names.end(), name) == names.end()) continue;
        auto t0 = std::chrono::steady_clock::now();
        std::filesystem::remove("data/" + name + ".npz");
        fresh[name] = run(imgs);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::println("{}: {} rows in {:.0f} s", name, fresh[name].size(), secs);
        std::fflush(stdout);
    }
    update_table(fresh);
}
