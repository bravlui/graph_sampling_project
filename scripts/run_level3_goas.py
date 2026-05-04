#!/usr/bin/env python3
"""
run_level3_goas.py — GOAS vs Baselines
=======================================

Compara Goal-Oriented Adaptive Sampling contra samplers clássicos
com testes estatísticos.

Uso:
    python scripts/run_level3_goas.py --config configs/default.yaml
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon, mannwhitneyu
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, ensure_dirs, set_global_seed, results_path, save_csv
from src.graph_generators import generate_graph_suite
from src.samplers import get_sampler
from src.metrics import compute_node_distributions, compute_global_metrics
from src.distances import compare_all_distributions, compare_global_metrics
from src.experiments import compute_sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GOAS_OBJECTIVES = ["degree", "clustering", "shortest_path", "centrality", "balanced"]


def run_goas_benchmark(config):
    """Executa benchmark GOAS vs baselines."""
    logger = setup_logging()
    logger.info("=== Nível 3 C1: GOAS vs Baselines ===")

    suite = generate_graph_suite(config)
    sampling_cfg = config.get("sampling", {})
    fractions = sampling_cfg.get("fractions", [0.10, 0.20, 0.30])
    repetitions = sampling_cfg.get("repetitions", 5)
    baseline_samplers = sampling_cfg.get("samplers", ["random_node", "snowball", "random_walk"])
    goas_params = config.get("sampler_params", {}).get("goas", {})

    rows = []

    total = len(suite) * len(fractions) * (len(baseline_samplers) + len(GOAS_OBJECTIVES)) * repetitions
    pbar = tqdm(total=total, desc="GOAS Benchmark")

    for graph_info in suite:
        G = graph_info["graph"]
        original_dists = compute_node_distributions(G, approx_betweenness=True, seed=0)
        original_globals = compute_global_metrics(G, approx_betweenness=True, seed=0)

        for frac in fractions:
            # Baselines
            for sname in baseline_samplers:
                sampler_fn = get_sampler(sname)
                for rep in range(repetitions):
                    rep_seed = graph_info["seed"] * 1000 + rep
                    try:
                        G_s, meta = sampler_fn(G=G, sample_frac=frac, seed=rep_seed)
                        s_dists = compute_node_distributions(G_s, approx_betweenness=True, seed=rep_seed)
                        s_globals = compute_global_metrics(G_s, approx_betweenness=True, seed=rep_seed)
                        dist_comp = compare_all_distributions(original_dists, s_dists)
                        global_comp = compare_global_metrics(original_globals, s_globals)
                        sps = compute_sps(dist_comp, global_comp)

                        row = {
                            "graph_id": graph_info["graph_id"],
                            "model": graph_info["model"],
                            "sample_frac": frac,
                            "sampler": sname,
                            "objective": "none",
                            "rep": rep,
                            "SPS": sps,
                            **{k: v for k, v in dist_comp.items() if k.endswith("_JS")},
                            **{k: v for k, v in global_comp.items()},
                        }
                        rows.append(row)
                        
                        # Limpa memória das variáveis pesadas da iteração
                        del G_s, s_dists, s_globals, dist_comp, global_comp
                    except Exception as e:
                        logger.warning(f"Erro {sname}: {e}")
                    pbar.update(1)

            # GOAS com cada objetivo
            goas_fn = get_sampler("goas")
            for obj in GOAS_OBJECTIVES:
                for rep in range(repetitions):
                    rep_seed = graph_info["seed"] * 1000 + rep
                    try:
                        G_s, meta = goas_fn(
                            G=G, sample_frac=frac, seed=rep_seed,
                            objective=obj, **goas_params,
                        )
                        s_dists = compute_node_distributions(G_s, approx_betweenness=True, seed=rep_seed)
                        s_globals = compute_global_metrics(G_s, approx_betweenness=True, seed=rep_seed)
                        dist_comp = compare_all_distributions(original_dists, s_dists)
                        global_comp = compare_global_metrics(original_globals, s_globals)
                        sps = compute_sps(dist_comp, global_comp)

                        row = {
                            "graph_id": graph_info["graph_id"],
                            "model": graph_info["model"],
                            "sample_frac": frac,
                            "sampler": f"goas_{obj}",
                            "objective": obj,
                            "rep": rep,
                            "SPS": sps,
                            **{k: v for k, v in dist_comp.items() if k.endswith("_JS")},
                            **{k: v for k, v in global_comp.items()},
                        }
                        rows.append(row)
                        
                        # Limpa memória das variáveis pesadas da iteração
                        del G_s, s_dists, s_globals, dist_comp, global_comp
                    except Exception as e:
                        logger.warning(f"Erro GOAS-{obj}: {e}")
                    pbar.update(1)

        # Salva o progresso parcial e libera a memória do grafo atual
        df_partial = pd.DataFrame(rows)
        save_csv(df_partial, results_path(config, "raw", "level3_goas_results.csv"))
        
        del G
        del original_dists
        del original_globals
        gc.collect()

    pbar.close()

    df = pd.DataFrame(rows)
    save_csv(df, results_path(config, "raw", "level3_goas_results.csv"))

    # Resumo
    if not df.empty:
        summary = df.groupby(["sampler", "model", "sample_frac"])["SPS"].agg(["mean", "std"]).reset_index()
        save_csv(summary, results_path(config, "processed", "level3_goas_summary.csv"))

    # Testes estatísticos
    _statistical_tests(config, df)

    # Figura
    _plot_goas_vs_baselines(config, df)

    logger.info("GOAS benchmark concluído.")
    return df


def _statistical_tests(config, df):
    """Testes estatísticos GOAS vs melhor baseline."""
    if df.empty:
        return

    results = []
    baselines = [s for s in df["sampler"].unique() if not s.startswith("goas_")]
    goas_variants = [s for s in df["sampler"].unique() if s.startswith("goas_")]

    for model in df["model"].unique():
        for frac in df["sample_frac"].unique():
            for goas_name in goas_variants:
                goas_vals = df[(df["model"] == model) & (df["sample_frac"] == frac) &
                              (df["sampler"] == goas_name)]["SPS"].dropna().values

                # Melhor baseline
                best_bl_name = None
                best_bl_mean = float("inf")
                best_bl_vals = None

                for bl in baselines:
                    bl_vals = df[(df["model"] == model) & (df["sample_frac"] == frac) &
                               (df["sampler"] == bl)]["SPS"].dropna().values
                    if len(bl_vals) > 0 and bl_vals.mean() < best_bl_mean:
                        best_bl_mean = bl_vals.mean()
                        best_bl_name = bl
                        best_bl_vals = bl_vals

                if best_bl_vals is None or len(goas_vals) < 3 or len(best_bl_vals) < 3:
                    continue

                # Teste
                min_len = min(len(goas_vals), len(best_bl_vals))
                try:
                    if min_len >= 6:
                        stat, pval = wilcoxon(goas_vals[:min_len], best_bl_vals[:min_len])
                        test_name = "wilcoxon"
                    else:
                        stat, pval = mannwhitneyu(goas_vals, best_bl_vals, alternative="less")
                        test_name = "mannwhitney"
                except Exception:
                    continue

                improvement = (best_bl_mean - goas_vals.mean()) / max(best_bl_mean, 1e-10) * 100

                results.append({
                    "model": model,
                    "sample_frac": frac,
                    "goas_variant": goas_name,
                    "best_baseline": best_bl_name,
                    "goas_sps_mean": goas_vals.mean(),
                    "baseline_sps_mean": best_bl_mean,
                    "improvement_pct": improvement,
                    "test": test_name,
                    "p_value": pval,
                })

    if results:
        save_csv(pd.DataFrame(results),
                 results_path(config, "processed", "level3_goas_statistical_tests.csv"))
        save_csv(pd.DataFrame(results),
                 results_path(config, "processed", "level3_goas_improvement_table.csv"))


def _plot_goas_vs_baselines(config, df):
    """Barplot GOAS vs baselines."""
    if df.empty:
        return

    means = df.groupby("sampler")["SPS"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#9B59B6" if s.startswith("goas") else "#457B9D" for s in means.index]
    ax.barh(range(len(means)), means.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index, fontsize=10)
    ax.set_xlabel("SPS Médio (menor = melhor)", fontsize=12)
    ax.set_title("GOAS vs Baselines — SPS Médio", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    filepath = results_path(config, "figures", "level3_goas_vs_baselines.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Nível 3 C1: GOAS")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    run_goas_benchmark(config)


if __name__ == "__main__":
    main()
