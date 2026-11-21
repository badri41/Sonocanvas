/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Picture -> target magnitude.
*/
#include <cmath>
#include <stdexcept>

#include "sonocanvas.hpp"

namespace sc {

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

}  // namespace sc
