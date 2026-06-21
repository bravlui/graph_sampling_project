#!/usr/bin/env python3
"""
run_stats.py — Testes estatísticos comparativos entre métodos de amostragem
============================================================================

Realiza análise de significância estatística sobre o SPS dos 6 métodos clássicos:

1. Kruskal-Wallis (omnibus): testa se pelo menos um método difere globalmente
   e por modelo de rede. Não-paramétrico; não assume normalidade.

2. Wilcoxon signed-rank pairwise: compara cada par de métodos pareando os
   experimentos por (graph_id, sample_frac). Correção de Bonferroni aplicada
   sobre os C(6,2) = 15 pares possíveis.

3. Tamanho de efeito: correlação rank biserial r = 1 - 2W / (n*(n+1)).
   Interpretado como pequeno (|r| < 0.3), médio (0.3–0.5), grande (> 0.5).

Uso
---
    python scripts/run_stats.py
    python scripts/run_stats.py --config configs/default.yaml
"""

import sys
import os
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, setup_logging, ensure_dirs, results_path, save_csv
from src.visualization import DPI, _save_fig


# ─────────────────────────────────────────────
# Kruskal-Wallis
# ─────────────────────────────────────────────
def run_kruskal_wallis(df, samplers, metric="SPS"):
    rows = []
    scopes = [("global", None)] + [("per_model", m) for m in sorted(df["model"].unique())]
    for scope, model in scopes:
        sub = df if model is None else df[df["model"] == model]
        groups = [sub[sub["sampler"] == s][metric].dropna().values for s in samplers]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        stat, p = stats.kruskal(*groups)
        rows.append({
            "scope": scope,
            "model": model if model else "all",
            "H_statistic": round(stat, 4),
            "p_value": p,
            "significant_a005": p < 0.05,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Wilcoxon pairwise + tamanho de efeito
# ─────────────────────────────────────────────
def run_pairwise_wilcoxon(df, samplers, metric="SPS"):
    pairs = list(itertools.combinations(samplers, 2))
    n_tests = len(pairs)          # 15 pares → Bonferroni × 15
    models = ["all"] + sorted(df["model"].unique())
    rows = []

    for model in models:
        sub = df if model == "all" else df[df["model"] == model]
        for s1, s2 in pairs:
            # Parear por (graph_id, sample_frac): média das repetições
            v1 = (sub[sub["sampler"] == s1]
                  .groupby(["graph_id", "sample_frac"])[metric].mean())
            v2 = (sub[sub["sampler"] == s2]
                  .groupby(["graph_id", "sample_frac"])[metric].mean())
            common = v1.index.intersection(v2.index)
            if len(common) < 5:
                continue
            a = v1.loc[common].values
            b = v2.loc[common].values
            if np.all(a == b):
                continue
            try:
                stat, p_raw = stats.wilcoxon(a, b, alternative="two-sided")
            except Exception:
                continue

            p_bonf = min(p_raw * n_tests, 1.0)

            # Rank biserial: r = 1 - 2W / (n*(n+1))
            n = len(a)
            r = abs(1.0 - (2.0 * stat) / (n * (n + 1)))
            magnitude = ("grande" if r > 0.5 else
                         "médio"  if r > 0.3 else "pequeno")

            rows.append({
                "model": model,
                "sampler_a": s1,
                "sampler_b": s2,
                "mean_a": round(a.mean(), 4),
                "mean_b": round(b.mean(), 4),
                "mean_diff": round(a.mean() - b.mean(), 4),
                "W_statistic": stat,
                "p_raw": p_raw,
                "p_bonferroni": p_bonf,
                "effect_r": round(r, 3),
                "effect_magnitude": magnitude,
                "significant_bonf": p_bonf < 0.05,
                "n_pairs": n,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────
def _build_matrix(df_pairs, samplers, col, fill_diag):
    n = len(samplers)
    idx = {s: i for i, s in enumerate(samplers)}
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, fill_diag)
    for _, row in df_pairs.iterrows():
        i = idx[row["sampler_a"]]
        j = idx[row["sampler_b"]]
        mat[i, j] = row[col]
        mat[j, i] = row[col]
    return mat


def plot_pvalue_heatmap(df_global, samplers, config):
    mat_p = _build_matrix(df_global, samplers, "p_bonferroni", fill_diag=1.0)
    mat_log = -np.log10(np.clip(mat_p, 1e-12, 1.0))
    np.fill_diagonal(mat_log, 0.0)

    labels = [s.replace("_", "\n") for s in samplers]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat_log, cmap="Reds",
                   vmin=0, vmax=max(6.0, np.nanmax(mat_log)))

    ax.set_xticks(range(len(samplers)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(samplers)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(samplers)):
        for j in range(len(samplers)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
            else:
                p = mat_p[i, j]
                stars = ("***" if p < 0.001 else
                         "**"  if p < 0.01  else
                         "*"   if p < 0.05  else "ns")
                color = "white" if mat_log[i, j] > 3 else "black"
                ax.text(j, i, f"{p:.3f}\n{stars}",
                        ha="center", va="center", fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("−log₁₀(p Bonferroni)", fontsize=11)
    ax.set_title(
        "Wilcoxon Pareado — p-valor com Correção de Bonferroni (15 pares)\n"
        "* p<0,05  ** p<0,01  *** p<0,001  ns = não significativo",
        fontsize=12, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    _save_fig(fig, results_path(config, "figures", "stats_pvalue_heatmap.png"))


def plot_effect_heatmap(df_global, samplers, config):
    mat_r = _build_matrix(df_global, samplers, "effect_r", fill_diag=0.0)
    labels = [s.replace("_", "\n") for s in samplers]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat_r, cmap="Blues", vmin=0, vmax=1.0)

    ax.set_xticks(range(len(samplers)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(samplers)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(samplers)):
        for j in range(len(samplers)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
            else:
                r = mat_r[i, j]
                mag = "G" if r > 0.5 else "M" if r > 0.3 else "P"
                color = "white" if r > 0.65 else "black"
                ax.text(j, i, f"{r:.2f}\n({mag})",
                        ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Tamanho de efeito |r| (rank biserial)", fontsize=11)
    ax.set_title(
        "Tamanho de Efeito — Rank Biserial Correlation\n"
        "G = grande (|r|>0,5)  M = médio (0,3–0,5)  P = pequeno (<0,3)",
        fontsize=12, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    _save_fig(fig, results_path(config, "figures", "stats_effect_heatmap.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    df = pd.read_csv(results_path(config, "raw", "distribution_distances.csv"))
    samplers = sorted(df["sampler"].unique())
    logger.info(f"Dataset: {len(df)} linhas, {len(samplers)} samplers")

    # 1. Kruskal-Wallis
    df_kw = run_kruskal_wallis(df, samplers)
    save_csv(df_kw, results_path(config, "processed", "stats_kruskal_wallis.csv"))
    logger.info("Kruskal-Wallis concluído")
    print(df_kw[["scope", "model", "H_statistic", "p_value",
                  "significant_a005"]].to_string(index=False))

    # 2. Wilcoxon pairwise
    df_wilcoxon = run_pairwise_wilcoxon(df, samplers)
    save_csv(df_wilcoxon, results_path(config, "processed",
                                       "stats_pairwise_wilcoxon.csv"))
    logger.info(f"Wilcoxon: {len(df_wilcoxon)} comparações")

    # Resumo global
    df_global = df_wilcoxon[df_wilcoxon["model"] == "all"]
    sig = df_global[df_global["significant_bonf"]]
    print(f"\nPares significativos (Bonferroni a=0.05): {len(sig)}/{len(df_global)}")
    print(sig[["sampler_a", "sampler_b", "mean_diff",
               "p_bonferroni", "effect_r", "effect_magnitude"]].to_string(index=False))

    # 3. Figuras
    plot_pvalue_heatmap(df_global, samplers, config)
    plot_effect_heatmap(df_global, samplers, config)
    logger.info("Figuras geradas.")


if __name__ == "__main__":
    main()
