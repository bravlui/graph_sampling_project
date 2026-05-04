#!/usr/bin/env python3
"""
run_level3_pareto.py — Structural Preservation Pareto Frontier
==============================================================

Analisa amostragem como problema multiobjetivo com fronteira de Pareto.

Uso:
    python scripts/run_level3_pareto.py --config configs/default.yaml
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
from src.visualization import SAMPLER_COLORS, _get_color

LOSS_COLS = [
    "degree_JS", "clustering_JS", "betweenness_JS",
    "re_avg_shortest_path_lcc", "re_avg_clustering", "re_avg_degree",
]


def main():
    parser = argparse.ArgumentParser(description="Nível 3 C2: Pareto Frontier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    # Carregar resultados do Nível 1
    dist_path = results_path(config, "raw", "level1_distribution_distances.csv")
    global_path = results_path(config, "raw", "level1_global_errors.csv")

    try:
        df_dist = pd.read_csv(dist_path)
        df_global = pd.read_csv(global_path)
    except FileNotFoundError:
        logger.error("Execute primeiro o Nível 1.")
        return

    # Merge das métricas relevantes
    merge_cols = ["graph_id", "model", "graph_seed", "sampler", "sample_frac", "rep_seed"]
    available_merge = [c for c in merge_cols if c in df_dist.columns and c in df_global.columns]

    js_cols = [c for c in LOSS_COLS if c.endswith("_JS") and c in df_dist.columns]
    re_cols = [c for c in LOSS_COLS if c.startswith("re_") and c in df_global.columns]

    df = df_dist[available_merge + js_cols].copy()
    if re_cols:
        df_re = df_global[available_merge + re_cols]
        df = df.merge(df_re, on=available_merge, how="left")

    # Tentar incluir resultados GOAS
    goas_path = results_path(config, "raw", "level3_goas_results.csv")
    if os.path.exists(goas_path):
        df_goas = pd.read_csv(goas_path)
        available_loss = [c for c in LOSS_COLS if c in df_goas.columns]
        common_cols = [c for c in available_merge if c in df_goas.columns]
        if common_cols and available_loss:
            df_goas_sub = df_goas[common_cols + available_loss].copy()
            df = pd.concat([df, df_goas_sub], ignore_index=True)
        logger.info("Resultados GOAS incluídos na análise Pareto.")

    available_loss = [c for c in LOSS_COLS if c in df.columns]
    if not available_loss:
        logger.error("Nenhuma coluna de perda disponível.")
        return

    # 1. Fronteira de Pareto por modelo de rede
    df = compute_pareto_frontier(df, available_loss, group_cols=["model"])
    save_csv(df[df["is_pareto"]], results_path(config, "processed", "level3_pareto_points.csv"))

    # 2. Resumo de dominância
    summary = pareto_dominance_summary(df)
    save_csv(summary, results_path(config, "processed", "level3_pareto_summary.csv"))

    # 3. Hypervolume proxy
    df = compute_hypervolume_proxy(df, available_loss)

    # 4. Trade-offs
    tradeoffs = analyze_tradeoffs(df, available_loss)
    if tradeoffs:
        save_csv(pd.DataFrame(tradeoffs),
                 results_path(config, "processed", "level3_pareto_tradeoffs.csv"))

    # 5. Eficiência amostral
    efficiency = analyze_sample_efficiency(df, available_loss)
    if not efficiency.empty:
        save_csv(efficiency, results_path(config, "processed", "level3_sample_efficiency.csv"))

    # 6. Recomendações Pareto
    _generate_recommendations(config, df, available_loss)

    # ===== Visualizações =====
    _plot_pareto_2d(config, df, "degree_JS", "clustering_JS",
                    "level3_pareto_degree_clustering.png")
    _plot_pareto_2d(config, df, "re_avg_shortest_path_lcc", "betweenness_JS",
                    "level3_pareto_shortpath_betweenness.png")
    _plot_pareto_frequency(config, df)
    _plot_pareto_heatmap(config, df)

    logger.info("Pareto Frontier analysis concluída.")


def _generate_recommendations(config, df, loss_cols):
    """Gera recomendações Pareto por modelo de rede."""
    if "is_pareto" not in df.columns:
        return

    pareto = df[df["is_pareto"]]
    if pareto.empty:
        return

    rows = []
    for model in pareto["model"].unique():
        sub = pareto[pareto["model"] == model]
        for _, row in sub.iterrows():
            rows.append({
                "model": model,
                "sampler": row.get("sampler", ""),
                "sample_frac": row.get("sample_frac", ""),
                "hypervolume_proxy": row.get("hypervolume_proxy", float("nan")),
            })

    save_csv(pd.DataFrame(rows),
             results_path(config, "processed", "level3_pareto_recommendations.csv"))


def _plot_pareto_2d(config, df, col_x, col_y, filename):
    """Scatter 2D com pontos Pareto destacados."""
    if col_x not in df.columns or col_y not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    samplers = sorted(df["sampler"].unique())
    for sname in samplers:
        sub = df[df["sampler"] == sname]
        pareto_mask = sub["is_pareto"] if "is_pareto" in sub.columns else pd.Series([False] * len(sub))

        non_pareto = sub[~pareto_mask]
        pareto = sub[pareto_mask]

        ax.scatter(non_pareto[col_x], non_pareto[col_y],
                  color=_get_color(sname), alpha=0.3, s=15, label=f"{sname}")
        ax.scatter(pareto[col_x], pareto[col_y],
                  color=_get_color(sname), alpha=0.9, s=60, edgecolors="black",
                  linewidths=1.5, marker="*")

    ax.set_xlabel(col_x, fontsize=12)
    ax.set_ylabel(col_y, fontsize=12)
    ax.set_title(f"Pareto Frontier: {col_x} vs {col_y}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    filepath = results_path(config, "figures", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto_frequency(config, df):
    """Frequência de aparição na fronteira por sampler."""
    if "is_pareto" not in df.columns:
        return

    freq = df.groupby("sampler")["is_pareto"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [_get_color(s) for s in freq.index]
    ax.bar(range(len(freq)), freq.values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(freq)))
    ax.set_xticklabels(freq.index, rotation=30, ha="right")
    ax.set_ylabel("Frequência na Fronteira Pareto", fontsize=12)
    ax.set_title("Frequência Pareto por Sampler", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    filepath = results_path(config, "figures", "level3_pareto_frequency.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto_heatmap(config, df):
    """Heatmap sampler x modelo com frequência Pareto."""
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

    filepath = results_path(config, "figures", "level3_pareto_heatmap.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
