/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Unit tests for the core (doctest). Run with `make test`.
*/
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <numbers>
#include <random>

#include "sonocanvas.hpp"

using namespace sc;
constexpr int N = 1024, H = 256, SR = 22050;

TEST_CASE("STFT then ISTFT returns white noise to 1e-8")
{
    // Length is a multiple of H: istft returns H floor(L/H) samples, so a tail would be cut otherwise.
    std::mt19937 rng(0);
    std::normal_distribution<double> g;
    Signal x(200 * H);
    for (double &v : x) v = g(rng);
    auto w = window("hann", N);
    Signal y = istft(stft(x, N, H, w), N, H, w);
    REQUIRE(y.size() == x.size());
    double err = 0;
    for (size_t i = 0; i < x.size(); ++i) err = std::max(err, std::abs(y[i] - x[i]));
    CHECK(err < 1e-8);
}

TEST_CASE("Hann at H = N/4: sum_m w^2[n - mH] = 3N/(8H) = 1.5 away from the edges")
{
    auto w = window("hann", N);
    int M = 40;
    std::vector<double> acc(N + H * (M - 1), 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) acc[m * H + n] += w[n] * w[n];
    for (size_t i = N; i < acc.size() - N; ++i) CHECK(acc[i] == doctest::Approx(3.0 * N / (8 * H)).epsilon(1e-12));
}

TEST_CASE("a 3 kHz tone peaks within one bin of 3 kHz")
{
    Signal x(SR);
    for (int i = 0; i < SR; ++i) x[i] = std::sin(2 * std::numbers::pi * 3000.0 * i / SR);
    Image mag = magnitude(stft(x, N, H, window("hann", N)));
    Image mean;
    cv::reduce(mag, mean, 1, cv::REDUCE_AVG);
    cv::Point peak;
    cv::minMaxLoc(mean, nullptr, nullptr, nullptr, &peak);
    CHECK(std::abs(peak.y * double(SR) / N - 3000.0) <= double(SR) / N);
}

TEST_CASE("AUC equals the pairwise definition, ties included")
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> U;
    Image target(30, 40), rec(30, 40);
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 40; ++j) {
            target(i, j) = U(rng) > 0.7;
            rec(i, j) = std::round((U(rng) + 0.4 * target(i, j)) * 10) / 10;  // rounded: many ties
        }
    double gt = 0, eq = 0, pairs = 0;
    for (auto s = rec.begin(), ts = target.begin(); s != rec.end(); ++s, ++ts)
        for (auto b = rec.begin(), tb = target.begin(); b != rec.end(); ++b, ++tb)
            if (*ts > 0.5 && *tb <= 0.5) {
                ++pairs;
                gt += *s > *b;
                eq += *s == *b;
            }
    CHECK(auc(target, rec) == doctest::Approx((gt + 0.5 * eq) / pairs).epsilon(1e-12));
}

TEST_CASE("AUC limits and invariance to an increasing map")
{
    Image target = Image::zeros(20, 20);
    target(cv::Rect(5, 5, 10, 5)) = 1.0;
    CHECK(auc(target, target) == 1.0);
    CHECK(auc(target, Image(1.0 - target)) == 0.0);
    CHECK(auc(target, Image(Image::ones(20, 20))) == 0.5);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> U;
    Image rec = target.clone(), warped(20, 20);
    for (double &v : rec) v += 0.3 * U(rng);
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j) warped(i, j) = 0.01 * std::pow(rec(i, j), 3);
    CHECK(auc(target, rec) == auc(target, warped));
}

TEST_CASE("DA321 survives encode and decode")
{
    Encoded e = encode(preprocess(render_text("DA321"), "edges"));
    Image rec = decode(e.x);
    CHECK(rec.size() == e.target.size());
    CHECK(rec.rows == 325);
    CHECK(rec.cols == 431);
    CHECK(ssim(e.target, rec) > 0.9);
    CHECK(auc(e.target, rec) > 0.99);
}

TEST_CASE("long text is shrunk to fit the canvas")
{
    Image img = render_text("DATA PROCESSING");
    double left, right, mx;
    cv::minMaxLoc(img.col(0), nullptr, &left);
    cv::minMaxLoc(img.col(img.cols - 1), nullptr, &right);
    cv::minMaxLoc(img, nullptr, &mx);
    CHECK(img.cols == 900);
    CHECK(left == 0);
    CHECK(right == 0);
    CHECK(mx == 1.0);
}

TEST_CASE("fixed font keeps the glyph height and lengthens the clip")
{
    auto height = [](const Image &m) {  // rows containing ink
        cv::Mat rows;
        cv::reduce(m, rows, 1, cv::REDUCE_MAX);
        return cv::countNonZero(rows);
    };
    CHECK(cv::norm(render_text("HELLO", 900, 260, false), render_text("HELLO")) == 0);  // short text: unchanged
    Image wide = render_text("DRAWING WITH SOUND", 900, 260, false);
    CHECK(wide.cols > 900);
    CHECK(height(wide) == height(render_text("HELLO")));
    Image img = preprocess(wide, "edges");
    Params p;
    p.dur = text_duration(img);
    Encoded e = encode(img, p);
    CHECK(std::abs(e.x.size() / double(SR) - p.dur) < 0.05);
    CHECK(auc(e.target, decode(e.x, p)) > 0.99);
}
