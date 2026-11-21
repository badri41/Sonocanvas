/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Files: WAV through libsndfile, the answer-key PNG through OpenCV, .npz through cnpy.
*/
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sndfile.h>

#include "sonocanvas.hpp"
#include "third_party/cnpy.h"

namespace sc {

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

/*
   cnpy wants one continuous block of floats, hence the clone.
*/
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
