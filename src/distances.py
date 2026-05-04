"""
distances.py — Distâncias entre distribuições
===============================================

Compara distribuições de métricas entre a rede original e a rede amostrada
usando KL, JS, Wasserstein e KS.
"""

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
import logging

logger = logging.getLogger("graph_sampling")


def normalize_histogram(values, bins=50, range_=None, eps=1e-12):
    """
    Retorna histograma normalizado (soma = 1).

    Parameters
    ----------
    values : array-like
        Valores para o histograma.
    bins : int
        Número de bins.
    range_ : tuple or None
        Range dos bins (min, max). Se None, usa range dos dados.
    eps : float
        Smoothing aditivo.

    Returns
    -------
    np.ndarray
        Histograma normalizado.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.ones(bins) / bins
    hist, _ = np.histogram(values, bins=bins, range=range_, density=False)
    hist = hist.astype(float) + eps
    return hist / hist.sum()


def kl_divergence(p, q, eps=1e-12):
    """
    KL(P || Q) com smoothing.

    Parameters
    ----------
    p, q : array-like
        Distribuições normalizadas.
    eps : float
        Smoothing.

    Returns
    -------
    float
    """
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q, eps=1e-12):
    """
    Jensen-Shannon divergence.

    Returns
    -------
    float
        JSD entre p e q.
    """
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=0) + 0.5 * kl_divergence(q, m, eps=0)


def wasserstein_distance_1d(x, y):
    """
    Earth Mover's Distance 1D.

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float(wasserstein_distance(x, y))


def ks_statistic(x, y):
    """
    Kolmogorov-Smirnov test.

    Returns
    -------
    dict
        {"statistic": float, "pvalue": float}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    stat, pval = ks_2samp(x, y)
    return {"statistic": float(stat), "pvalue": float(pval)}


def compare_distributions(original_dist, sampled_dist, metric_name, bins=50):
    """
    Compara duas distribuições com múltiplas métricas.

    Parameters
    ----------
    original_dist : array-like
        Valores da métrica na rede original.
    sampled_dist : array-like
        Valores da métrica na rede amostrada.
    metric_name : str
        Nome da métrica para labels.
    bins : int
        Número de bins para KL e JS.

    Returns
    -------
    dict
        Contém KL, JS, Wasserstein, KS_statistic, KS_pvalue.
    """
    orig = np.asarray(original_dist, dtype=float)
    samp = np.asarray(sampled_dist, dtype=float)

    # Range compartilhado para histogramas
    if len(orig) > 0 and len(samp) > 0:
        combined = np.concatenate([orig, samp])
        range_ = (combined.min(), combined.max())
    elif len(orig) > 0:
        range_ = (orig.min(), orig.max())
    else:
        range_ = (0, 1)

    # Evitar range degenerado
    if range_[0] == range_[1]:
        range_ = (range_[0] - 0.5, range_[1] + 0.5)

    p = normalize_histogram(orig, bins=bins, range_=range_)
    q = normalize_histogram(samp, bins=bins, range_=range_)

    result = {
        f"{metric_name}_KL": kl_divergence(p, q),
        f"{metric_name}_JS": js_divergence(p, q),
        f"{metric_name}_wasserstein": wasserstein_distance_1d(orig, samp),
    }

    ks = ks_statistic(orig, samp)
    result[f"{metric_name}_KS_stat"] = ks["statistic"]
    result[f"{metric_name}_KS_pvalue"] = ks["pvalue"]

    return result


def compare_all_distributions(original_dists, sampled_dists, bins=50):
    """
    Aplica compare_distributions para todas as métricas de nó.

    Parameters
    ----------
    original_dists : dict[str, array]
        Distribuições da rede original.
    sampled_dists : dict[str, array]
        Distribuições da rede amostrada.

    Returns
    -------
    dict
    """
    result = {}
    metric_names = ["degree", "clustering", "betweenness", "eigenvector", "closeness", "k_core"]

    for name in metric_names:
        orig = original_dists.get(name, np.array([]))
        samp = sampled_dists.get(name, np.array([]))
        try:
            d = compare_distributions(orig, samp, name, bins=bins)
            result.update(d)
        except Exception as e:
            logger.warning(f"Erro comparando {name}: {e}")
            for suffix in ["_KL", "_JS", "_wasserstein", "_KS_stat", "_KS_pvalue"]:
                result[f"{name}{suffix}"] = float("nan")

    return result


def relative_error(original_value, sampled_value):
    """
    Erro relativo robusto.

    Returns
    -------
    float
    """
    try:
        ov = float(original_value)
        sv = float(sampled_value)
    except (TypeError, ValueError):
        return float("nan")

    if np.isnan(ov) or np.isnan(sv):
        return float("nan")

    denom = max(abs(ov), 1e-10)
    return abs(ov - sv) / denom


def compare_global_metrics(original_global, sampled_global):
    """
    Calcula erro relativo para métricas globais.

    Returns
    -------
    dict
        Erro relativo para cada métrica.
    """
    result = {}
    skip_keys = {"n_nodes", "n_edges", "num_connected_components"}

    for key in original_global:
        if key in skip_keys:
            continue
        ov = original_global.get(key)
        sv = sampled_global.get(key)
        if ov is not None and sv is not None:
            result[f"re_{key}"] = relative_error(ov, sv)
        else:
            result[f"re_{key}"] = float("nan")

    return result
