/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Image and spectral quality scores.
*/
#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "sonocanvas.hpp"

namespace sc {

Image ssim_map(const Image &a, const Image &b)
{
    const double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03, n = 49.0 / 48.0;  // n: sample (co)variance
    auto mean = [](const Image &m) {  // local mean over a 7x7 window
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

double auc(const Image &target, const Image &rec)
{
    std::vector<double> t(target.begin(), target.end()), r(rec.begin(), rec.end());
    size_t n = r.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) { return r[i] < r[j]; });

    std::vector<double> rank(n);  // 1-based ranks; tied values share their average rank
    for (size_t i = 0; i < n;) {
        size_t j = i;
        while (j < n && r[order[j]] == r[order[i]]) ++j;
        for (size_t k = i; k < j; ++k) rank[order[k]] = (i + 1 + j) / 2.0;
        i = j;
    }
    double n1 = 0, sum = 0;  // stroke pixel count and their rank sum
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

}  // namespace sc
