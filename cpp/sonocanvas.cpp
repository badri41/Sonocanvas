// The encoder and decoder.
#include <algorithm>
#include <cmath>

#include "sonocanvas.hpp"

namespace sc {

Image band_image(const Image& mag, const Params& p) {
    Image out;
    cv::flip(mag.rowRange(band(p)), out, 0);
    double mx;
    cv::minMaxLoc(out, nullptr, &mx);
    return out / mx;
}

Encoded encode(const Image& img, const Params& p, int n_iter, const std::string& init, unsigned seed) {
    std::vector<double> w = window(p.window, p.n_fft);
    Encoded e;
    e.S = image_to_mag(img, p);
    PhaseResult gl = griffin_lim(e.S, p.n_fft, p.hop, w, n_iter, init, seed);
    double peak = 0;
    for (double v : gl.x) peak = std::max(peak, std::abs(v));
    for (double& v : gl.x) v *= 0.9 / peak;  // x <- 0.9 x / max|x|
    e.x = std::move(gl.x);
    e.errs = std::move(gl.errs);
    e.target = band_image(e.S, p);
    return e;
}

Image decode(const Signal& x, const Params& p) {
    Image rec = band_image(magnitude(stft(x, p.n_fft, p.hop, window(p.window, p.n_fft))), p);
    return resize(rec, rec.rows, p.frames());  // the target grid: band bins x M frames
}

}  // namespace sc
