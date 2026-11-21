/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Windows, STFT and the overlap-add inverse. I wrote the framing and the overlap-add by hand, since that's
the part of the method I want visible. The FFT itself is cv::dft; my own FFT would add code and teach nothing new.
*/
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "sonocanvas.hpp"

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

}  // namespace sc
