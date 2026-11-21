/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

The scores: SSIM, PSNR, AUC and spectral convergence. AUC is the main legibility
score here: SSIM penalises speckle on flat regions even when it doesn't hurt reading.
*/
#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "sonocanvas.hpp"

namespace sc {

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

}  // namespace sc
