#!/usr/bin/env python3
"""
run_pareto.py — Análise multiobjetivo e fronteira de Pareto
============================================================

Trata a qualidade da amostragem como um problema de otimização multiobjetivo:
em vez de um único score (SPS), analisamos simultaneamente a preservação de grau,
clustering e betweenness. Um sampler é "Pareto-eficiente" se nenhum outro método
consegue melhorar em todas as métricas ao mesmo tempo — ele representa um compromisso
ótimo entre objetivos potencialmente conflitantes.

Exemplo de conflito: um método pode preservar bem a distribuição de grau (baixo
degree_JS) mas distorcer o caminho médio. Identificar a fronteira de Pareto expõe
esses trade-offs explicitamente.

Uso
---
    python scripts/run_pareto.py
    python scripts/run_pareto.py --config configs/default.yaml
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, setup_logging, ensure_dirs, set_global_seed, results_path, save_csv
from src.pareto import (
    compute_pareto_frontier, pareto_dominance_summary,
    compute_hypervolume_proxy, analyze_tradeoffs, analyze_sample_efficiency,
)
from src.visualization import _get_color

# Colunas de "custo" que queremos minimizar (menor = melhor preservação)
LOSS_COLS = [
    "degree_JS", "clustering_JS", "betweenness_JS",
    "re_avg_shortest_path_lcc", "re_avg_clustering", "re_avg_degree",
]


def main():
    parser = argparse.ArgumentParser(
        description="Análise multiobjetivo e fronteira de Pareto dos samplers"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Caminho para arquivo de configuração YAML")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    # Carregar resultados do benchmark principal
    dist_path = results_path(config, "raw", "distribution_distances.csv")
    global_path = results_path(config, "raw", "global_errors.csv")

    try:
        df_dist = pd.read_csv(dist_path)
        df_global = pd.read_csv(global_path)
    except FileNotFoundError:
        logger.error("Execute primeiro o benchmark principal: python scripts/run_benchmark.py")
        return

    # Merge das colunas relevantes para análise multiobjetivo
    merge_cols = ["graph_id", "model", "graph_seed", "sampler", "sample_frac", "rep_seed"]
    available_merge = [c for c in merge_cols if c in df_dist.columns and c in df_global.columns]

    js_cols = [c for c in LOSS_COLS if c.endswith("_JS") and c in df_dist.columns]
    re_cols = [c for c in LOSS_COLS if c.startswith("re_") and c in df_global.columns]

    df = df_dist[available_merge + js_cols].copy()
    if re_cols:
        df_re = df_global[available_merge + re_cols]
        df = df.merge(df_re, on=available_merge, how="left")

    # Incluir resultados GOAS se disponíveis (gerados por run_advanced.py)
    goas_path = results_path(config, "raw", "advanced_results.csv")
    if os.path.exists(goas_path):
        df_goas = pd.read_csv(goas_path)
        available_loss = [c for c in LOSS_COLS if c in df_goas.columns]
        common_cols = [c for c in available_merge if c in df_goas.columns]
        if common_cols and available_loss:
            df = pd.concat([df, df_goas[common_cols + available_loss]], ignore_index=True)
        logger.info("Resultados de samplers avançados incluídos na análise Pareto.")

    available_loss = [c for c in LOSS_COLS if c in df.columns]
    if not available_loss:
        logger.error("Nenhuma coluna de perda disponível no dataset.")
        return

    # 1. Identificar fronteira de Pareto por modelo de rede
    df = compute_pareto_frontier(df, available_loss, group_cols=["model"])
    save_csv(df[df["is_pareto"]], results_path(config, "processed", "pareto_points.csv"))

    # 2. Resumo de dominância: com que frequência cada sampler domina os demais
    summary = pareto_dominance_summary(df)
    save_csv(summary, results_path(config, "processed", "pareto_dominance_summary.csv"))

    # 3. Hypervolume proxy: volume do espaço de objetivos "dominado" pelo sampler
    df = compute_hypervolume_proxy(df, available_loss)

    # 4. Trade-offs: correlação entre pares de objetivos (quais são conflitantes?)
    tradeoffs = analyze_tradeoffs(df, available_loss)
    if tradeoffs:
        save_csv(pd.DataFrame(tradeoffs), results_path(config, "processed", "pareto_tradeoffs.csv"))

    # 5. Eficiência amostral: como o desempenho muda com a fração amostral
    efficiency = analyze_sample_efficiency(df, available_loss)
    if not efficiency.empty:
        save_csv(efficiency, results_path(config, "processed", "pareto_sample_efficiency.csv"))

    # 6. Recomendações Pareto por modelo de rede
    _generate_recommendations(config, df)

    # ===== Visualizações =====
    _plot_pareto_2d(config, df, "degree_JS", "clustering_JS", "pareto_degree_clustering.png")
    _plot_pareto_2d(config, df, "re_avg_shortest_path_lcc", "betweenness_JS",
                    "pareto_shortpath_betweenness.png")
    _plot_pareto_frequency(config, df)
    _plot_pareto_heatmap(config, df)

    logger.info("Análise Pareto concluída.")


def _generate_recommendations(config, df):
    """Salva os pontos Pareto com informação de hypervolume por modelo."""
    if "is_pareto" not in df.columns:
        return
    pareto = df[df["is_pareto"]]
    if pareto.empty:
        return

    rows = [
        {
            "model": row.get("model", ""),
            "sampler": row.get("sampler", ""),
            "sample_frac": row.get("sample_frac", ""),
            "hypervolume_proxy": row.get("hypervolume_proxy", float("nan")),
        }
        for _, row in pareto.iterrows()
    ]
    save_csv(pd.DataFrame(rows), results_path(config, "processed", "pareto_recommendations.csv"))


def _plot_pareto_2d(config, df, col_x, col_y, filename):
    """Scatter 2D: todos os pontos em baixa opacidade, Pareto com estrela."""
    if col_x not in df.columns or col_y not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for sname in sorted(df["sampler"].unique()):
        sub = df[df["sampler"] == sname]
        pareto_mask = sub.get("is_pareto", pd.Series([False] * len(sub)))
        ax.scatter(sub[~pareto_mask][col_x], sub[~pareto_mask][col_y],
                   color=_get_color(sname), alpha=0.3, s=15, label=sname)
        ax.scatter(sub[pareto_mask][col_x], sub[pareto_mask][col_y],
                   color=_get_color(sname), alpha=0.9, s=60, edgecolors="black",
                   linewidths=1.5, marker="*")

    ax.set_xlabel(col_x, fontsize=12)
    ax.set_ylabel(col_y, fontsize=12)
    ax.set_title(f"Fronteira Pareto: {col_x} × {col_y}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(config, fig, filename)


def _plot_pareto_frequency(config, df):
    """Frequência relativa de cada sampler na fronteira de Pareto."""
    if "is_pareto" not in df.columns:
        return

    freq = df.groupby("sampler")["is_pareto"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(freq)), freq.values, color=[_get_color(s) for s in freq.index], alpha=0.8)
    ax.set_xticks(range(len(freq)))
    ax.set_xticklabels(freq.index, rotation=30, ha="right")
    ax.set_ylabel("Frequência na Fronteira Pareto", fontsize=12)
    ax.set_title("Qual sampler aparece mais na Fronteira Pareto?", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(config, fig, "pareto_frequency.png")


def _plot_pareto_heatmap(config, df):
    """Heatmap de frequência Pareto por modelo de rede × sampler."""
    if "is_pareto" not in df.columns:
        return

    pivot = df.groupby(["model", "sampler"])["is_pareto"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Frequência Pareto")
    ax.set_title("Frequência Pareto: Modelo × Sampler", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(config, fig, "pareto_heatmap.png")


def _save_fig(config, fig, filename):
    from src.utils import results_path
    filepath = results_path(config, "figures", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
