/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

The encoder and the decoder.
*/
#include <algorithm>
#include <cmath>

#include "sonocanvas.hpp"

namespace sc {

Image band_image(const Image &mag, const Params &p)
{
    Image out;
    cv::flip(mag.rowRange(band(p)), out, 0);
    double mx;
    cv::minMaxLoc(out, nullptr, &mx);
    return out / mx;
}

/*
   I scale the waveform so its peak is 0.9. The WAV never clips, and there's some headroom left
   for whatever the player does to it.
*/
Encoded encode(const Image &img, const Params &p, int n_iter, const std::string &init, unsigned seed)
{
    std::vector<double> w = window(p.window, p.n_fft);
    Encoded e;
    e.S = image_to_mag(img, p);
    Phase_Result gl = griffin_lim(e.S, p.n_fft, p.hop, w, n_iter, init, seed);
    double peak = 0;
    for (double v : gl.x) peak = std::max(peak, std::abs(v));
    for (double &v : gl.x) v *= 0.9 / peak;
    e.x = std::move(gl.x);
    e.errs = std::move(gl.errs);
    e.target = band_image(e.S, p);
    return e;
}

/*
   A WAV that is not exactly the encoder's length (a recording, say) gives a different number of
   frames, so I resize onto the target grid: band bins x M frames. For my own WAVs it's a no-op.
*/
Image decode(const Signal &x, const Params &p)
{
    Image rec = band_image(magnitude(stft(x, p.n_fft, p.hop, window(p.window, p.n_fft))), p);
    return resize(rec, rec.rows, p.frames());
}

}  // namespace sc
