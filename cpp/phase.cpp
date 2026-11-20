// Phase recovery by Griffin-Lim alternating projections.
#include <numbers>
#include <random>
#include <stdexcept>

#include "sonocanvas.hpp"

namespace sc {

PhaseResult griffin_lim(const Image& S, int N, int H, const std::vector<double>& w, int n_iter,
                        const std::string& init, unsigned seed) {
    if (init != "random" && init != "zero") throw std::invalid_argument("unknown init " + init);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    Spectrum X(S.size());
    for (int k = 0; k < S.rows; ++k)
        for (int m = 0; m < S.cols; ++m)
            X(k, m) = init == "random" ? std::polar(S(k, m), 2 * std::numbers::pi * U(rng))
                                       : std::complex<double>(S(k, m), 0.0);

    PhaseResult out;
    for (int i = 0; i < n_iter; ++i) {
        Spectrum Y = stft(istft(X, N, H, w), N, H, w).colRange(0, S.cols);  // project onto real sounds
        out.errs.push_back(spectral_convergence(Y, S));
        for (int k = 0; k < S.rows; ++k)  // keep Y's phase, restore our magnitudes
            for (int m = 0; m < S.cols; ++m) {
                double a = std::abs(Y(k, m));
                X(k, m) = a > 0 ? S(k, m) * Y(k, m) / a : std::complex<double>(S(k, m), 0.0);
            }
    }
    out.x = istft(X, N, H, w);
    return out;
}

}  // namespace sc
