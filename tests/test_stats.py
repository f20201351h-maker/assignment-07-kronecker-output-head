"""R0: uncertainty helpers on hand-checkable cases."""

import math

import numpy as np

from kq5.stats import (bpb_of, bucket_nll, cluster_bootstrap_mean, frequency_buckets, two_level_bootstrap,
                       verdict_interval, window_sums)


def test_window_sums_and_bpb():
    nll = np.array([[math.log(2), 9.0], [math.log(4), math.log(2)]])
    nb = np.array([[1, 0], [2, 1]])
    s, b = window_sums(nll, nb)
    assert np.allclose(s, [math.log(2), math.log(8)]) and np.allclose(b, [1, 3])
    assert abs(bpb_of(s, b) - 1.0) < 1e-12


def test_identical_arms_difference_is_zero_with_zero_width():
    rng = np.random.default_rng(0)
    s = rng.random(50) * 100
    b = np.full(50, 200.0)
    res = two_level_bootstrap({"a": [s], "b": [s.copy()]}, b, lambda v: v["a"] - v["b"], n_boot=200)
    assert res["point"] == 0 and res["ci95"] == [0.0, 0.0]


def test_seed_noise_widens_interval():
    rng = np.random.default_rng(1)
    s = rng.random(200) * 100 + 500
    b = np.full(200, 200.0)
    one = two_level_bootstrap({"a": [s], "b": [s + 5]}, b, lambda v: v["b"] - v["a"], n_boot=500)
    two = two_level_bootstrap({"a": [s, s + 20], "b": [s + 5, s - 15]}, b, lambda v: v["b"] - v["a"], n_boot=500)
    assert one["ci95"][1] - one["ci95"][0] < two["ci95"][1] - two["ci95"][0]


def test_cluster_bootstrap_mean_and_verdicts():
    v = np.array([1.0, 1.0, 3.0, 3.0])
    c = np.array([0, 0, 1, 1])
    r = cluster_bootstrap_mean(v, c, n_boot=500)
    assert r["mean"] == 2.0 and r["n_clusters"] == 2 and 1.0 <= r["ci95"][0] <= r["ci95"][1] <= 3.0
    assert verdict_interval([0.6, 0.9], 0.5) == "SUPPORTED"
    assert verdict_interval([0.1, 0.4], 0.5) == "REJECTED"
    assert verdict_interval([0.4, 0.6], 0.5) == "INCONCLUSIVE"
    assert verdict_interval([-0.5, -0.1], 0.0, higher_is_support=False) == "SUPPORTED"


def test_frequency_buckets_and_bucket_nll():
    idx, labels = frequency_buckets(np.array([0, 1, 9, 10, 150, 2_000_000]))
    assert idx.tolist() == [0, 1, 1, 2, 3, 7] and labels[0] == "[0,1)" and labels[-1] == "[1000000,inf)"
    nll = np.array([[1.0, 2.0, 3.0]])
    targets = np.array([[0, 1, 2]])
    nb = np.array([[1, 1, 0]])
    out = bucket_nll(nll, targets, nb, np.array([0, 1, 1]), 2)
    assert out["nll"][:, 0].tolist() == [1.0, 2.0] and out["count"][:, 0].tolist() == [1, 1]
