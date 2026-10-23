import numpy as np


def ndcg_at(ranks, k=10):
    return np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)


def hit_at(ranks, k=10):
    return (ranks < k).astype(float)


def _metric(metric):
    return ndcg_at if metric == "ndcg" else hit_at


def _boot_means(x, B, seed):
    rng = np.random.default_rng(seed)
    n, means = len(x), np.empty(B)
    for s in range(0, B, 100):
        idx = rng.integers(0, n, (min(100, B - s), n))
        means[s:s + 100] = x[idx].mean(1)
    return means


def mean_ci(x, B=2000, seed=0, alpha=0.05):
    """Trung bình và khoảng tin cậy bootstrap của một mảng giá trị theo từng user."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(_boot_means(x, B, seed), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_metric(ranks, k=10, metric="ndcg", B=2000, seed=0, alpha=0.05):
    """Giá trị trung bình và khoảng tin cậy bootstrap của một metric."""
    return mean_ci(_metric(metric)(np.asarray(ranks), k), B, seed, alpha)


def paired_bootstrap(rank_a, rank_b, k=10, metric="ndcg", B=2000, seed=0, alpha=0.05):
    """Chênh lệch metric (b - a) theo từng user và khoảng tin cậy bootstrap ghép cặp.
    Truyền alpha = 0.05 / m để hiệu chỉnh Bonferroni khi so sánh m biến thể."""
    f = _metric(metric)
    return mean_ci(f(np.asarray(rank_b), k) - f(np.asarray(rank_a), k), B, seed, alpha)
