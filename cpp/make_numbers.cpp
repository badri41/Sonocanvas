/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Writes paper/numbers.tex, paper/table1.tex, paper/acoustic.tex and the README results table, all
from results/table.csv and the .npz files in data/. Every number in main.tex is a macro made here,
so no result is typed by hand and the paper can't drift from the data.

Rounding is half-up on the shortest decimal form (0.9135 -> 0.914). Rounding the double itself
would sometimes go the wrong way, because 0.9135 is really 0.91349999... in binary.
*/
#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <set>

#include "sonocanvas.hpp"
#include "third_party/cnpy.h"

using Fields = std::map<std::string, std::string>;

/*
   Reads a CSV with a header into one map per row: column name -> value.
*/
std::vector<Fields> read_records(const std::string &path)
{
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    std::vector<std::string> header = sc::split_csv(line);
    std::vector<Fields> rows;
    for (const auto &v : sc::read_csv(path)) {
        Fields r;
        for (size_t i = 0; i < header.size(); ++i) r[header[i]] = i < v.size() ? v[i] : "";
        rows.push_back(r);
    }
    return rows;
}

/*
   Reading side of an .npz. at() throws with the name of the array if it's missing.
*/
struct Npz {
    cnpy::npz_t arrays;
    explicit Npz(const std::string &path) : arrays(cnpy::npz_load(path)) {}
    const cnpy::NpyArray &at(const std::string &key) const
    {
        auto it = arrays.find(key);
        if (it == arrays.end()) throw std::runtime_error("no array " + key);
        return it->second;
    }
    std::vector<double> vec(const std::string &key) const
    {
        const cnpy::NpyArray &a = at(key);
        if (a.word_size == 4) return {a.data<float>(), a.data<float>() + a.num_vals};
        return {a.data<double>(), a.data<double>() + a.num_vals};
    }
    double operator[](const std::string &key) const { return vec(key).at(0); }
    /*
       Column mean of a 2-D array, like numpy's .mean(0).
    */
    std::vector<double> mean0(const std::string &key) const
    {
        const cnpy::NpyArray &a = at(key);
        std::vector<double> v = vec(key), m(a.shape[1], 0.0);
        for (size_t i = 0; i < a.shape[0]; ++i)
            for (size_t j = 0; j < a.shape[1]; ++j) m[j] += v[i * a.shape[1] + j] / a.shape[0];
        return m;
    }
};

/*
   The shortest decimal string that reads back as exactly x. Same thing Python's str() gives you.
*/
std::string shortest(double x)
{
    char buf[64];
    auto [end, ec] = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::fixed);
    return std::string(buf, end);
}

/*
   Rounds a decimal string to `places` digits, halves go away from zero. Works on the digits
   directly, carrying 9s by hand, so no floating point is involved.
*/
std::string rnd(const std::string &s, int places = 3)
{
    bool neg = !s.empty() && s[0] == '-';
    std::string t = neg ? s.substr(1) : s;
    size_t dot = t.find('.');
    std::string ip = t.substr(0, dot), fp = dot == std::string::npos ? "" : t.substr(dot + 1);
    fp.resize(places + 1, '0');
    std::string digits = ip + fp.substr(0, places);
    if (fp[places] >= '5') {
        int i = int(digits.size()) - 1;
        while (i >= 0 && digits[i] == '9') digits[i--] = '0';
        if (i < 0) digits.insert(digits.begin(), '1');
        else ++digits[i];
    }
    std::string out =
        places ? digits.substr(0, digits.size() - places) + "." + digits.substr(digits.size() - places) : digits;
    return (neg ? "-" : "") + out;
}
std::string rnd(double x, int places = 3)
{
    return rnd(shortest(x), places);
}

/*
   a - b for two decimal strings with four places, done exactly in units of 10^-4.
*/
std::string sub4(const std::string &a, const std::string &b)
{
    auto units = [](std::string s) {
        s.erase(std::remove(s.begin(), s.end(), '.'), s.end());
        return std::stoll(s);
    };
    long long d = units(a) - units(b);
    std::string sign = d < 0 ? "-" : "";
    d = std::llabs(d);
    return std::format("{}{}.{:04}", sign, d / 10000, d % 10000);
}

std::map<std::string, Fields> R;  // key "experiment|image|param"
const std::map<char, std::string> EXP = {{'e', "edges_vs_filled"},   {'n', "sweep_window_len"},
                                         {'w', "sweep_window_type"}, {'g', "gl_iterations"},
                                         {'m', "mel_vs_linear"},     {'s', "noise_robustness"}};

/*
   One cell of the results table. e is a one letter code for the experiment (see EXP).
   ss() rounds an SSIM to 3 places. au() gives AUC as is, with 4 places, because clean scores
   only differ in the 4th digit.
*/
std::string get(char e, const std::string &image, const std::string &param, const std::string &col = "ssim")
{
    auto it = R.find(EXP.at(e) + "|" + image + "|" + param);
    if (it == R.end()) throw std::runtime_error("no row " + EXP.at(e) + " " + image + " " + param);
    return it->second.at(col);
}
std::string ss(char e, const std::string &image, const std::string &param, int places = 3)
{
    return rnd(get(e, image, param), places);
}
std::string au(char e, const std::string &image, const std::string &param)
{
    return get(e, image, param, "auc");
}

/*
   How many percent of an SSIM change (a to b) came from the flat windows.
*/
std::string flat_share(double frac, double flat_a, double flat_b, double rest_a, double rest_b)
{
    double f = frac * (flat_b - flat_a);
    return rnd(100 * f / (f + (1 - frac) * (rest_b - rest_a)), 0);
}

/*
   The difference between two SSIMs in the table, done exactly and then rounded.
*/
std::string gap(char e, const std::string &im_a, const std::string &p_a, const std::string &im_b,
                const std::string &p_b)
{
    return rnd(sub4(get(e, im_a, p_a), get(e, im_b, p_b)));
}

int main()
{
    for (const Fields &r : read_records("results/table.csv"))
        R[r.at("experiment") + "|" + r.at("image") + "|" + r.at("param")] = r;
    Npz E("data/edges_vs_filled.npz"), W("data/sweep_window_type.npz"), G("data/gl_iterations.npz"),
        L("data/sweep_window_len.npz");

    double max_std = 0;
    for (const auto &[k, r] : R)
        if (!r.at("ssim_std").empty()) max_std = std::max(max_std, std::stod(r.at("ssim_std")));
    auto energy = [&](const std::string &n) {
        return (E[n + "_edges_energy"] / E[n + "_edges_lit_frac"]) / (E[n + "_raw_energy"] / E[n + "_raw_lit_frac"]);
    };
    std::vector<double> sc_text = G.mean0("text_errs_random");
    double auc_clean_min = 1;
    for (std::string n : {"text", "line"})
        for (int N : {512, 1024})
            auc_clean_min = std::min(auc_clean_min, std::stod(au('n', n, std::format("N={}/image", N))));
    std::vector<double> own_var;
    // NOTE: these are the lengths in data/sweep_window_len.npz. They're hardcoded because that
    // array is int64 and cnpy can't tell int64 from float64.
    for (int N : {128, 256, 512, 1024, 2048, 4096}) own_var.push_back(L[std::format("photo_own_flat_var_{}", N)]);

    std::vector<std::pair<std::string, std::string>> M = {
        {"MaxStd", rnd(max_std, 3)},
        // Experiment 1
        {"TextRawTwo", ss('e', "text", "raw", 2)},
        {"TextEdgesTwo", ss('e', "text", "edges", 2)},
        {"TextRaw", ss('e', "text", "raw")},
        {"TextEdges", ss('e', "text", "edges")},
        {"LineRaw", ss('e', "line", "raw")},
        {"LineEdges", ss('e', "line", "edges")},
        {"PhotoRaw", ss('e', "photo", "raw")},
        {"PhotoEdges", ss('e', "photo", "edges")},
        {"TextRawPsnr", rnd(get('e', "text", "raw", "psnr_db"), 1)},
        {"TextEdgesPsnr", rnd(get('e', "text", "edges", "psnr_db"), 1)},
        {"TextRawSupport", ss('e', "text", "raw/support")},
        {"TextEdgesSupport", ss('e', "text", "edges/support")},
        {"TextRawLs", ss('e', "text", "raw/ls_gain")},
        {"TextEdgesLs", ss('e', "text", "edges/ls_gain")},
        {"TextRawInterior", ss('e', "text", "raw/interior")},
        {"TextRawPred", ss('e', "text", "raw/interior_pred")},
        {"LineRawInterior", ss('e', "line", "raw/interior")},
        {"LineRawPred", ss('e', "line", "raw/interior_pred")},
        {"CvRaw", rnd(E["text_raw_cv"], 2)},
        {"CvEdges", rnd(E["text_edges_cv"], 2)},
        {"LitRaw", rnd(E["text_raw_lit_frac"], 3)},
        {"LitEdges", rnd(E["text_edges_lit_frac"], 3)},
        {"GainRaw", rnd(E["text_raw_gain"], 2)},
        {"GainEdges", rnd(E["text_edges_gain"], 2)},
        // Experiment 2
        {"TextNOneTwoEight", ss('n', "text", "N=128/image")},
        {"TextNFiveTwelve", ss('n', "text", "N=512/image")},
        {"TextNOneK", ss('n', "text", "N=1024/image")},
        {"TextNFourK", ss('n', "text", "N=4096/image")},
        {"TextHalfFiveTwelve", ss('n', "text", "N=512/half")},
        {"TextHalfOneK", ss('n', "text", "N=1024/half")},
        {"TextOwnFourK", ss('n', "text", "N=4096/own")},
        {"LineNFiveTwelve", ss('n', "line", "N=512/image")},
        {"LineNOneK", ss('n', "line", "N=1024/image")},
        {"PhotoNOneK", ss('n', "photo", "N=1024/image")},
        {"PhotoNTwoK", ss('n', "photo", "N=2048/image")},
        {"PhotoNFourK", ss('n', "photo", "N=4096/image")},
        // Experiment 3
        {"TextRect", ss('w', "text", "rect")},
        {"TextHann", ss('w', "text", "hann")},
        {"TextHamming", ss('w', "text", "hamming")},
        {"TextBlackman", ss('w', "text", "blackman")},
        {"PhotoRect", ss('w', "photo", "rect")},
        {"PhotoHann", ss('w', "photo", "hann")},
        {"SideRect", rnd(W["rect_sidelobe_db"], 1)},
        {"SideHann", rnd(W["hann_sidelobe_db"], 1)},
        {"SideHamming", rnd(W["hamming_sidelobe_db"], 1)},
        {"SideBlackman", rnd(W["blackman_sidelobe_db"], 1)},
        // Experiment 4
        {"IterZero", ss('g', "text", "random/iter=0")},
        {"IterOne", ss('g', "text", "random/iter=1")},
        {"IterSixty", ss('g', "text", "random/iter=60")},
        {"IterTwoHundred", ss('g', "text", "random/iter=200")},
        {"ZeroInitTwoHundred", ss('g', "text", "zero/iter=200")},
        {"PhotoIterSixty", ss('g', "photo", "random/iter=60")},
        {"PhotoIterTwoHundred", ss('g', "photo", "random/iter=200")},
        {"ScZero", rnd(sc_text[0], 3)},
        {"ScSixty", rnd(sc_text[60], 3)},
        {"ScTwoHundred", rnd(sc_text[200], 3)},
        // Experiment 5
        {"LinLin", ss('m', "text", "encode=linear/view=linear")},
        {"LinMel", ss('m', "text", "encode=linear/view=mel")},
        {"MelLin", ss('m', "text", "encode=mel/view=linear")},
        {"MelMel", ss('m', "text", "encode=mel/view=mel")},
        // Experiment 6
        {"RawThirty", ss('s', "text", "raw/snr=30")},
        {"EdgesThirty", ss('s', "text", "edges/snr=30")},
        {"EdgesFloorThirty", ss('s', "text", "edges/floor=30")},
        {"EdgesTen", ss('s', "text", "edges/snr=10")},
        {"EdgesZero", ss('s', "text", "edges/snr=0")},
        {"PhotoThirty", ss('s', "photo", "raw/snr=30")},
        {"GapClean", gap('e', "text", "edges", "text", "raw")},
        {"GapSnr", gap('s', "text", "edges/snr=30", "text", "raw/snr=30")},
        {"GapFloor", gap('s', "text", "edges/floor=30", "text", "raw/floor=30")},
        {"EnergyPerLitText", rnd(energy("text"), 2)},
        {"EnergyPerLitLine", rnd(energy("line"), 1)},
        {"EnergyPerLitPhoto", rnd(energy("photo"), 1)},
        // AUC
        {"AucTextRaw", au('e', "text", "raw")},
        {"AucTextEdges", au('e', "text", "edges")},
        {"AucLineRaw", au('e', "line", "raw")},
        {"AucLineEdges", au('e', "line", "edges")},
        {"AucPhotoRaw", au('e', "photo", "raw")},
        {"AucPhotoEdges", au('e', "photo", "edges")},
        {"AucNOneTwoEight", au('n', "text", "N=128/image")},
        {"AucNFiveTwelve", au('n', "text", "N=512/image")},
        {"AucNOneK", au('n', "text", "N=1024/image")},
        {"AucNTwoK", au('n', "text", "N=2048/image")},
        {"AucNFourK", au('n', "text", "N=4096/image")},
        {"AucRect", au('w', "text", "rect")},
        {"AucHann", au('w', "text", "hann")},
        {"AucIterZero", au('g', "text", "random/iter=0")},
        {"AucIterSixty", au('g', "text", "random/iter=60")},
        {"AucPhotoIterZero", au('g', "photo", "random/iter=0")},
        {"AucPhotoIterSixty", au('g', "photo", "random/iter=60")},
        {"AucLinMel", au('m', "text", "encode=linear/view=mel")},
        {"AucMelLin", au('m', "text", "encode=mel/view=linear")},
        {"AucLinLin", au('m', "text", "encode=linear/view=linear")},
        {"AucEdgesThirty", au('s', "text", "edges/snr=30")},
        {"AucEdgesZero", au('s', "text", "edges/snr=0")},
        {"AucRawZero", au('s', "text", "raw/snr=0")},
        {"AucPhotoZero", au('s', "photo", "raw/snr=0")},
        {"AucEdgesFloorZero", au('s', "text", "edges/floor=0")},
        {"AucRawFloorZero", au('s', "text", "raw/floor=0")},
        {"AucCleanMin", rnd(auc_clean_min, 4)},
        // Flat regions of the photograph
        {"FlatFrac", rnd(100 * W["photo_hann_flat_frac"], 0)},
        {"FlatVarHann", rnd(W["photo_hann_flat_var"], 4)},
        {"FlatVarRect", rnd(W["photo_rect_flat_var"], 4)},
        {"FlatSsimHann", rnd(W["photo_hann_flat_ssim"], 2)},
        {"FlatSsimRect", rnd(W["photo_rect_flat_ssim"], 2)},
        {"FlatShareRect", flat_share(W["photo_hann_flat_frac"], W["photo_hann_flat_ssim"], W["photo_rect_flat_ssim"],
                                     W["photo_hann_rest_ssim"], W["photo_rect_rest_ssim"])},
        {"OwnVarLo", rnd(*std::min_element(own_var.begin(), own_var.end()), 4)},
        {"OwnVarHi", rnd(*std::max_element(own_var.begin(), own_var.end()), 4)},
        {"GridVarOneK", rnd(L["photo_flat_var_1024"], 4)},
        {"GridVarFourK", rnd(L["photo_flat_var_4096"], 4)},
        {"FlatShareLong", flat_share(L["photo_flat_frac_1024"], L["photo_flat_ssim_1024"], L["photo_flat_ssim_4096"],
                                     L["photo_rest_ssim_1024"], L["photo_rest_ssim_4096"])},
    };
    std::ofstream nums("paper/numbers.tex");
    nums << "% Generated by bin/make_numbers. Do not edit.\n";
    for (const auto &[k, v] : M) nums << std::format("\\newcommand{{\\num{}}}{{{}}}\n", k, v);

    const std::map<std::string, std::string> MODE = {{"text", "edges"}, {"line", "edges"}, {"photo", "raw"}};
    struct Headline {
        std::string label;
        char e;
        std::string param;
    };
    std::vector<Headline> rows = {{"Filled / raw", 'e', "raw"}, {"Sobel edges", 'e', "edges"}};
    for (int N : {128, 1024, 4096})
        rows.push_back({std::format("$N={}$, image grid", N), 'n', std::format("N={}/image", N)});
    rows.push_back({"Rectangular window", 'w', "rect"});
    rows.push_back({"Random phase only", 'g', "random/iter=0"});
    rows.push_back({"200 iterations", 'g', "random/iter=200"});
    rows.push_back({"Linear encode, mel view", 'm', "encode=linear/view=mel"});
    for (int s : {30, 10, 0}) rows.push_back({std::format("{} dB SNR", s), 's', std::format("snr={}", s)});
    // NOTE: the noise rows are keyed by mode too, and the mode depends on the picture.
    auto param = [&](const Headline &h, const std::string &n) {
        return h.e == 's' ? MODE.at(n) + "/" + h.param : h.param;
    };

    std::ofstream t1("paper/table1.tex");
    t1 << "% Generated by bin/make_numbers. Do not edit.\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
          " & \\multicolumn{2}{c}{Text} & \\multicolumn{2}{c}{Line} & Photo \\\\\n"
          "Configuration & AUC & SSIM & AUC & SSIM & SSIM \\\\\n\\midrule\n";
    std::string md =
        "| Configuration | Text AUC | Text SSIM | Line AUC | Line SSIM | Photo SSIM |\n|---|---|---|---|---|---|\n";
    for (const Headline &h : rows) {
        std::string cells = std::format("{} & {} & {} & {} & {}", au(h.e, "text", param(h, "text")),
                                        ss(h.e, "text", param(h, "text")), au(h.e, "line", param(h, "line")),
                                        ss(h.e, "line", param(h, "line")), ss(h.e, "photo", param(h, "photo")));
        t1 << h.label << " & " << cells << " \\\\\n";
        std::string label = h.label, mdcells = cells;
        std::erase(label, '$');
        for (size_t i; (i = mdcells.find(" & ")) != std::string::npos;) mdcells.replace(i, 3, " | ");
        md += "| " + label + " | " + mdcells + " |\n";
    }
    t1 << "\\bottomrule\n\\end{tabular}\n";

    std::ifstream rin("README.md");
    std::string readme((std::istreambuf_iterator<char>(rin)), {});
    const std::string open = "<!-- results -->\n", close = "<!-- /results -->";
    size_t a = readme.find(open), b = readme.find(close);
    if (a != std::string::npos && b != std::string::npos) {
        readme = readme.substr(0, a + open.size()) + md + readme.substr(b);
        std::ofstream("README.md") << readme;
    }

    // NOTE: until bin/analyze_recording has scored real recordings, the paper says plainly that
    // nothing was measured through a speaker. No acoustic number is ever made up.
    std::map<std::string, std::string> A = {
        {"Abstract", "Acoustic recovery through a physical speaker and microphone is not measured here."},
        {"Contribution", ""},
        {"Conclusion", ""},
        {"Limit", "Results are digital simulations with white noise: no measurement through a physical "
                  "speaker, room and microphone is reported here."},
        {"Future", "Future work should measure acoustic recovery directly."}};
    std::string sec = "% No acoustic measurements yet: run bin/analyze_recording on real recordings.\n";
    bool acoustic = std::filesystem::exists("results/acoustic.csv");
    if (acoustic) {
        std::vector<Fields> takes;
        for (const Fields &r : read_records("results/acoustic.csv"))
            if (r.at("word") == "DA321" && r.at("device") == "phone") takes.push_back(r);
        std::set<double> dists;
        for (const Fields &r : takes) dists.insert(std::stod(r.at("dist_m")));
        auto mean = [&](const std::string &k, double d) {
            double s = 0;
            int n = 0;
            for (const Fields &r : takes)
                if (std::stod(r.at("dist_m")) == d) s += std::stod(r.at(k)), ++n;
            return s / n;
        };
        auto count = [&](double d) {
            return std::count_if(takes.begin(), takes.end(),
                                 [&](const Fields &r) { return std::stod(r.at("dist_m")) == d; });
        };
        double far = *dists.rbegin();
        std::string body;
        for (double d : dists)
            body += std::format("{} & {} & {} & {} & {} \\\\\n", d, count(d), rnd(mean("auc", d), 4),
                                rnd(mean("ssim", d)), rnd(mean("snr_db", d), 1));
        std::string far_auc = rnd(mean("auc", far), 4);
        A["Abstract"] = std::format("Recordings of a laptop speaker on a phone microphone at up to {}\\,m give a mean "
                                    "stroke/background AUC of {} at that distance.",
                                    far, far_auc);
        A["Contribution"] = "\\item An acoustic validation through a physical speaker, room and phone microphone.";
        A["Conclusion"] =
            std::format("Decoding survives a real speaker and microphone at {}\\,m, with mean AUC {}.", far, far_auc);
        A["Limit"] = "Acoustic results cover one room, one laptop speaker and one phone microphone.";
        A["Future"] = "Future work should extend the acoustic test to other rooms and devices.";
        sec = std::format(
            "\\subsection{{Acoustic validation}}\n"
            "We played the encoded DA321 clip through a laptop speaker and recorded it on a phone at {} distances. "
            "Each recording was aligned by cross-correlation with the encoded signal and decoded with the unchanged "
            "decoder. "
            "Table~\\ref{{tab:acoustic}} reports means over takes. In-band SNR is estimated from room noise before and "
            "after the clip.\n"
            "\\begin{{table}}[t]\n\\centering\n\\caption{{Recorded through a physical speaker and "
            "microphone.}}\n\\label{{tab:acoustic}}\n"
            "\\footnotesize\n\\begin{{tabular}}{{@{{}}ccccc@{{}}}}\n\\toprule\nDistance (m) & Takes & AUC & SSIM & SNR "
            "(dB) \\\\\n\\midrule\n"
            "{}\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
            dists.size(), body);
    }
    std::ofstream("paper/acoustic.tex") << sec;
    for (std::string k : {"Abstract", "Contribution", "Conclusion", "Limit", "Future"})
        nums << std::format("\\newcommand{{\\Acoustic{}}}{{{}}}\n", k, A[k]);
    std::println("{} macros, {} table rows, acoustic data: {}", M.size(), rows.size(), acoustic);
}
