import numpy as np

from sonocanvas import auc


def test_auc_matches_pairwise_definition():
    # P(stroke pixel > background pixel) + P(equal)/2, counted over every pair, ties included.
    rng = np.random.default_rng(0)
    target = rng.random((30, 40)) > 0.7
    rec = np.round(rng.random((30, 40)) + 0.4 * target, 1)
    s, b = rec[target], rec[~target]
    pairs = (s[:, None] > b).mean() + 0.5 * (s[:, None] == b).mean()
    assert abs(auc(target.astype(float), rec) - pairs) < 1e-12


def test_auc_limits_and_scale_invariance():
    target = np.zeros((20, 20))
    target[5:10, 5:15] = 1.0
    assert auc(target, target) == 1.0
    assert auc(target, 1.0 - target) == 0.0
    assert auc(target, np.ones_like(target)) == 0.5  # no separation
    rec = target + 0.3 * np.random.default_rng(1).random(target.shape)
    assert auc(target, rec) == auc(target, 0.01 * rec ** 3)  # any increasing map
