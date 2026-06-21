"""
visualization.py — Visualizações do benchmark de amostragem
============================================================

Gera as figuras de análise a partir dos CSVs de resultado, usando matplotlib
puro (sem seaborn) para compatibilidade ampla. As figuras são salvas em PNG com
DPI 150, adequado para relatório e web.

Figuras produzidas
------------------
- sps_boxplot.png           : distribuição do SPS por sampler (boxplot com notch)
- sps_by_fraction.png       : curvas SPS médio × fração amostral, facetado por modelo
- heatmap_sampler_metric.png: JS divergence média por sampler e métrica estrutural
- heatmap_model_sampler.png : SPS médio por modelo de rede e sampler
- degree_distributions.png  : histogramas de grau sobrepostos (original vs amostradas)
- pca_samples.png           : projeção PCA das métricas globais das amostras
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

from src.utils import results_path

logger = logging.getLogger("graph_sampling")

# Paleta de cores para samplers
SAMPLER_COLORS = {
    "random_node": "#E63946",
    "random_edge": "#457B9D",
    "snowball": "#2A9D8F",
    "random_walk": "#E9C46A",
    "preferential_random_walk": "#F4A261",
    "metropolis_hastings_rw": "#264653",
    "goas": "#9B59B6",
    "goas_degree": "#8E44AD",
    "goas_clustering": "#6C3483",
    "goas_shortest_path": "#A569BD",
    "goas_centrality": "#BB8FCE",
    "goas_balanced": "#D2B4DE",
    "rl_sampler": "#E74C3C",
}

FIGSIZE = (12, 7)
DPI = 220

# Fontes maiores para legibilidade ao incorporar em relatório PDF
import matplotlib as _mpl
_mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
})


def _get_color(sampler_name):
    return SAMPLER_COLORS.get(sampler_name, "#888888")


def _save_fig(fig, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Figura salva: {filepath}")


# ============================================================================
# 1. Boxplot SPS por sampler
# ============================================================================
def plot_sps_boxplot(df, config, filename="sps_boxplot.png"):
    """Boxplot do SPS por sampler."""
    if "SPS" not in df.columns:
        return

    samplers = sorted(df["sampler"].unique())
    data = [df[df["sampler"] == s]["SPS"].dropna().values for s in samplers]
    colors = [_get_color(s) for s in samplers]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot(data, tick_labels=samplers, patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Structural Preservation Score (SPS)", fontsize=12)
    ax.set_xlabel("Sampler", fontsize=12)
    ax.set_title("SPS por Método de Amostragem", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, results_path(config, "figures", filename))


# ============================================================================
# 2. Linha SPS por fração amostral, facetado por modelo
# ============================================================================
def plot_sps_by_fraction(df, config, filename="sps_by_fraction.png"):
    """Linhas do SPS médio por fração amostral, separado por modelo."""
    if "SPS" not in df.columns:
        return

    models = sorted(df["model"].unique())
    samplers = sorted(df["sampler"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for sname in samplers:
            s_sub = sub[sub["sampler"] == sname]
            means = s_sub.groupby("sample_frac")["SPS"].mean()
            stds = s_sub.groupby("sample_frac")["SPS"].std()
            ax.errorbar(
                means.index, means.values, yerr=stds.values,
                marker="o", label=sname, color=_get_color(sname),
                capsize=3, linewidth=1.5,
            )
        ax.set_title(f"Modelo: {model}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Fração Amostral", fontsize=11)
        ax.set_ylabel("SPS médio" if ax == axes[0] else "", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("SPS por Fração Amostral", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, results_path(config, "figures", filename))


# ============================================================================
# 3. Heatmap: sampler x métrica
# ============================================================================
def plot_heatmap_sampler_metric(df, config, filename="heatmap_sampler_metric.png"):
    """Heatmap: sampler x métrica, mostrando erro médio."""
    metric_cols = [c for c in df.columns if c.endswith("_JS")]
    if not metric_cols:
        return

    samplers = sorted(df["sampler"].unique())
    matrix = np.zeros((len(samplers), len(metric_cols)))

    for i, s in enumerate(samplers):
        for j, m in enumerate(metric_cols):
            val = df[df["sampler"] == s][m].mean()
            matrix[i, j] = val if not np.isnan(val) else 0

    fig, ax = plt.subplots(figsize=(max(10, len(metric_cols)), max(6, len(samplers) * 0.6)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([c.replace("_JS", "") for c in metric_cols], rotation=45, ha="right")
    ax.set_yticks(range(len(samplers)))
    ax.set_yticklabels(samplers)

    # Valores no heatmap
    for i in range(len(samplers)):
        for j in range(len(metric_cols)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="JS Divergence (média)")
    ax.set_title("Erro Médio por Sampler e Métrica", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, results_path(config, "figures", filename))


# ============================================================================
# 4. Heatmap: modelo x sampler com SPS
# ============================================================================
def plot_heatmap_model_sampler(df, config, filename="heatmap_model_sampler.png"):
    """Heatmap: modelo de rede x sampler, mostrando SPS médio."""
    if "SPS" not in df.columns:
        return

    models = sorted(df["model"].unique())
    samplers = sorted(df["sampler"].unique())
    matrix = np.zeros((len(models), len(samplers)))

    for i, m in enumerate(models):
        for j, s in enumerate(samplers):
            val = df[(df["model"] == m) & (df["sampler"] == s)]["SPS"].mean()
            matrix[i, j] = val if not np.isnan(val) else 0

    fig, ax = plt.subplots(figsize=(max(8, len(samplers) * 1.2), max(4, len(models))))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(samplers)))
    ax.set_xticklabels(samplers, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(samplers)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, label="SPS médio")
    ax.set_title("SPS Médio por Modelo e Sampler", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, results_path(config, "figures", filename))


# ============================================================================
# 5. Distribuição de grau: original vs amostrada
# ============================================================================
def plot_degree_distributions(original_degrees, sampled_dict, config,
                               filename="degree_distributions.png"):
    """
    Gráfico de distribuição de grau original vs amostradas.

    Parameters
    ----------
    original_degrees : array-like
        Graus da rede original.
    sampled_dict : dict[str, array-like]
        {sampler_name: graus da amostra}.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Original
    orig = np.asarray(original_degrees)
    bins = np.arange(0, orig.max() + 2) - 0.5
    ax.hist(orig, bins=bins, density=True, alpha=0.4, color="black", label="Original", linewidth=1.5, histtype="step")

    for sname, degrees in sampled_dict.items():
        degrees = np.asarray(degrees)
        ax.hist(degrees, bins=bins, density=True, alpha=0.3, color=_get_color(sname),
                label=sname, histtype="step", linewidth=1.2)

    ax.set_xlabel("Grau", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.set_title("Distribuição de Grau: Original vs Amostradas", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, results_path(config, "figures", filename))


# ============================================================================
# 6. PCA das redes amostradas
# ============================================================================
def plot_pca_samples(df_global, config, filename="pca_samples.png"):
    """
    PCA das redes amostradas usando métricas globais como features.
    Cor por sampler, facet por modelo.
    """
    feature_cols = [c for c in df_global.columns if c.startswith("re_")]
    if not feature_cols or len(df_global) < 3:
        return

    # Limpar NaN
    df_clean = df_global[feature_cols].fillna(0)
    df_clean = df_clean.replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    df_plot = df_global.copy()
    df_plot["PC1"] = coords[:, 0]
    df_plot["PC2"] = coords[:, 1]

    models = sorted(df_plot["model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    samplers = sorted(df_plot["sampler"].unique())

    for ax, model in zip(axes, models):
        sub = df_plot[df_plot["model"] == model]
        for sname in samplers:
            s_sub = sub[sub["sampler"] == sname]
            ax.scatter(s_sub["PC1"], s_sub["PC2"], label=sname,
                      color=_get_color(sname), alpha=0.5, s=20)

        ax.set_title(f"Modelo: {model}", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("PCA das Redes Amostradas", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, results_path(config, "figures", filename))


# ============================================================================
# Runner
# ============================================================================
def generate_all_figures(config):
    """Gera todas as figuras do benchmark a partir dos CSVs de resultado."""
    try:
        df_dist = pd.read_csv(results_path(config, "raw", "distribution_distances.csv"))
        df_global = pd.read_csv(results_path(config, "raw", "global_errors.csv"))
    except FileNotFoundError as e:
        logger.error(f"CSVs não encontrados: {e}")
        return

    plot_sps_boxplot(df_dist, config)
    plot_sps_by_fraction(df_dist, config)
    plot_heatmap_sampler_metric(df_dist, config)
    plot_heatmap_model_sampler(df_dist, config)
    plot_pca_samples(df_global, config)
    logger.info("Todas as figuras do benchmark geradas.")


def generate_all_level1_figures(config):
    """Alias de compatibilidade: lê dos CSVs com prefixo legacy 'level1_'."""
    try:
        df_dist = pd.read_csv(results_path(config, "raw", "level1_distribution_distances.csv"))
        df_global = pd.read_csv(results_path(config, "raw", "level1_global_errors.csv"))
    except FileNotFoundError as e:
        logger.error(f"CSVs não encontrados: {e}")
        return

    plot_sps_boxplot(df_dist, config)
    plot_sps_by_fraction(df_dist, config)
    plot_heatmap_sampler_metric(df_dist, config)
    plot_heatmap_model_sampler(df_dist, config)
    plot_pca_samples(df_global, config)
    logger.info("Todas as figuras geradas.")
