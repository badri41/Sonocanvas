/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Image loading, text rendering, preprocessing and anti-aliased resizing.
*/
#include <cmath>
#include <stdexcept>

#include <opencv2/freetype.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "sonocanvas.hpp"

#ifndef SONOCANVAS_FONT
#define SONOCANVAS_FONT "cpp/fonts/DejaVuSans-Bold.ttf"
#endif

namespace sc {

Image load(const std::string &path)
{
    cv::Mat m = cv::imread(path, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
    if (m.empty()) throw std::runtime_error("cannot read image " + path);
    Image out;
    m.convertTo(out, CV_64F, 1.0 / (m.depth() == CV_16U ? 65535 : 255));
    return out;
}

// The text drawn white at `px` pixels, cropped to its ink: the bounding box of non-zero pixels.
static cv::Mat ink(const std::string &text, int px)
{
    static cv::Ptr<cv::freetype::FreeType2> ft = [] {
        auto f = cv::freetype::createFreeType2();
        f->loadFontData(SONOCANVAS_FONT, 0);  // throws if the font is missing
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
        if (fit) t = ink(text, int(170 * 0.92 * w / t.cols));  // shrink the font
        else w = int(std::ceil(t.cols / 0.92));                // or widen the canvas
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

Image preprocess(const Image &img, const std::string &mode, double gamma, int dilate)
{
    Image out;
    cv::normalize(img, out, 0, 1, cv::NORM_MINMAX);  // a constant image becomes all zeros
    cv::pow(out, gamma, out);
    if (mode == "raw") return out;
    if (mode != "edges") throw std::invalid_argument("unknown mode " + mode);

    Image gx, gy, g;
    cv::Sobel(out, gx, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_REFLECT);
    cv::Sobel(out, gy, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_REFLECT);
    cv::magnitude(gx, gy, g);
    double gmax;
    cv::minMaxLoc(g, nullptr, &gmax);
    cv::Mat edges = g > 0.25 * gmax;  // 0 or 255
    if (dilate > 0) cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_CROSS, {3, 3}), {-1, -1}, dilate);
    edges.convertTo(out, CV_64F, 1.0 / 255);
    return out;
}

// 1-D Gaussian kernel with radius floor(4*sigma + 1/2); the identity when sigma = 0.
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

}  // namespace sc
