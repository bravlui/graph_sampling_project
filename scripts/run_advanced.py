#!/usr/bin/env python3
"""
run_advanced.py — Samplers adaptativos: GOAS, Bandits e PSO
============================================================

Avalia métodos de amostragem orientados a objetivo contra os baselines clássicos,
usando duas abordagens distintas de adaptação:

1. GOAS-MH (Goal-Oriented Adaptive Sampling + Metropolis-Hastings):
   Variantes com objetivos específicos (degree, clustering, balanced). Ajusta
   dinamicamente os pesos de cada critério de seleção durante a amostragem.

2. Contextual Bandit:
   Trata cada sampler clássico como um "braço" do bandit e aprende iterativamente
   qual método explorar para uma dada rede, sem supervisão a priori.

3. PSO-GOAS (Particle Swarm Optimization sobre GOAS):
   Otimiza os hiperparâmetros do GOAS (alpha_degree, alpha_clustering, etc.) via
   enxame de partículas, buscando a combinação que minimiza o SPS na rede alvo.

Os resultados são comparados via teste de Wilcoxon de Wilcoxon unilateral
(H₁: advanced < baseline) e taxa de vitórias.

Uso
---
    python scripts/run_advanced.py
    python scripts/run_advanced.py --config configs/default.yaml
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_generators import generate_graph_suite
from src.samplers import get_sampler
from src.bandit_sampler import contextual_bandit_sampling
from src.meta_samplers import pso_goas, aco_sampler
from src.experiments import run_single_experiment, compute_sps
from src.metrics import compute_node_distributions, compute_global_metrics
from src.distances import compare_all_distributions, compare_global_metrics
from src.utils import load_config, setup_logging, ensure_dirs, set_global_seed, results_path, save_csv
from src.pareto import is_pareto_efficient


def _run_one(G, graph_info, sampler_name, sample_frac, rep_seed, orig_dists, orig_globals, **kwargs):
    """Executa um único experimento com sampler avançado ou clássico."""
    try:
        if sampler_name == "contextual_bandit":
            sub, meta = contextual_bandit_sampling(G, sample_frac, seed=rep_seed)
        elif sampler_name == "pso_goas":
            sub, meta = pso_goas(G, sample_frac, seed=rep_seed, n_particles=10, n_iterations=5)
        elif sampler_name == "aco_sampler":
            sub, meta = aco_sampler(G, sample_frac, seed=rep_seed, n_ants=10, n_iterations=5)
        else:
            dist_row, global_row = run_single_experiment(
                G, graph_info, sampler_name, sample_frac, rep_seed,
                orig_dists, orig_globals,
                sampler_params={sampler_name: kwargs} if kwargs else {},
            )
            if dist_row is None:
                return None
            return {**dist_row, **global_row}
    except Exception as e:
        logger.error(f"Sampler {sampler_name} falhou: {e}")
        return None

    try:
        samp_dists = compute_node_distributions(sub, approx_betweenness=True, seed=rep_seed)
        samp_globals = compute_global_metrics(sub, approx_betweenness=True, seed=rep_seed)
        d_comp = compare_all_distributions(orig_dists, samp_dists)
        g_comp = compare_global_metrics(orig_globals, samp_globals)
        sps = compute_sps(d_comp, g_comp)
    except Exception as e:
        logger.error(f"Métricas falharam para {sampler_name}: {e}")
        return None

    base = {
        "graph_id": graph_info["graph_id"],
        "model": graph_info["model"],
        "graph_seed": graph_info["seed"],
        "graph_n": G.number_of_nodes(),
        "sampler": sampler_name,
        "sample_frac": sample_frac,
        "rep_seed": rep_seed,
        "sampled_n": meta.get("sampled_n", 0),
        "SPS": sps,
    }
    if "objective" in kwargs:
        base["objective"] = kwargs["objective"]
    return {**base, **d_comp, **g_comp}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de samplers adaptativos (GOAS-MH, Bandit, PSO)"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    global logger
    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    logger.info("=== Benchmark de Samplers Adaptativos ===")

    suite = generate_graph_suite(config)
    sampling_cfg = config.get("sampling", {})
    fractions = sampling_cfg.get("fractions", [0.10, 0.20])[:2]  # limitar para custo computacional
    n_reps = min(sampling_cfg.get("repetitions", 5), 3)

    baselines = ["random_node", "random_walk", "snowball", "metropolis_hastings_rw"]
    advanced = ["contextual_bandit", "pso_goas"]

    # GOAS-MH com três objetivos para avaliar especialização
    goas_variants = [
        {"sampler": "goas_mh_adaptive", "label": "goas_mh_degree",     "objective": "degree"},
        {"sampler": "goas_mh_adaptive", "label": "goas_mh_clustering",  "objective": "clustering"},
        {"sampler": "goas_mh_adaptive", "label": "goas_mh_balanced",    "objective": "balanced"},
    ]

    rows = []
    pbar = tqdm(total=len(suite) * len(fractions) * n_reps, desc="Advanced")

    for graph_info in suite:
        G = graph_info["graph"]
        try:
            orig_dists = compute_node_distributions(G, approx_betweenness=True, seed=0)
            orig_globals = compute_global_metrics(G, approx_betweenness=True, seed=0)
        except Exception as e:
            logger.error(f"Métricas originais falharam para {graph_info['graph_id']}: {e}")
            pbar.update(len(fractions) * n_reps)
            continue

        for frac in fractions:
            for rep in range(n_reps):
                rep_seed = graph_info["seed"] * 100 + rep

                for sname in baselines:
                    row = _run_one(G, graph_info, sname, frac, rep_seed, orig_dists, orig_globals)
                    if row:
                        rows.append(row)

                for sname in advanced:
                    row = _run_one(G, graph_info, sname, frac, rep_seed, orig_dists, orig_globals)
                    if row:
                        rows.append(row)

                for v in goas_variants:
                    row = _run_one(G, graph_info, v["sampler"], frac, rep_seed,
                                   orig_dists, orig_globals, objective=v["objective"])
                    if row:
                        row["sampler"] = v["label"]
                        rows.append(row)

                pbar.update(1)

    pbar.close()

    if not rows:
        logger.error("Nenhum resultado gerado.")
        return

    df = pd.DataFrame(rows)
    save_csv(df, results_path(config, "raw", "advanced_results.csv"))
    logger.info(f"Resultados salvos: {len(df)} experimentos.")

    # Testes estatísticos (Wilcoxon unilateral): sampler avançado < baseline?
    _run_statistical_tests(config, df, baselines)

    # Fronteira de Pareto com os samplers avançados incluídos
    _run_pareto(config, df)

    logger.info("Benchmark de samplers adaptativos concluído.")


def _run_statistical_tests(config, df, baselines):
    """Testa se variantes GOAS-MH são estatisticamente melhores que baselines."""
    metrics_map = {
        "goas_mh_degree":     "degree_JS",
        "goas_mh_clustering": "clustering_JS",
        "goas_mh_balanced":   "SPS",
    }

    stats_rows = []
    for variant, target_metric in metrics_map.items():
        if variant not in df["sampler"].values or target_metric not in df.columns:
            continue
        var_data = df[df["sampler"] == variant]

        for baseline in baselines:
            if baseline not in df["sampler"].values:
                continue
            base_data = df[df["sampler"] == baseline]

            merged = pd.merge(
                var_data, base_data,
                on=["graph_id", "sample_frac", "rep_seed"],
                suffixes=("_var", "_base"),
            ).dropna(subset=[f"{target_metric}_var", f"{target_metric}_base"])

            if len(merged) < 5:
                continue

            x = merged[f"{target_metric}_var"].values
            y = merged[f"{target_metric}_base"].values
            stat, p_val = wilcoxon(x, y, alternative="less")

            stats_rows.append({
                "variant": variant,
                "baseline": baseline,
                "metric": target_metric,
                "n_samples": len(x),
                "p_value": round(p_val, 4),
                "significant_05": p_val < 0.05,
                "win_rate": float(np.sum(x < y) / len(x)),
            })

    if stats_rows:
        save_csv(pd.DataFrame(stats_rows), results_path(config, "processed", "advanced_wilcoxon_tests.csv"))


def _run_pareto(config, df):
    """Identifica pontos Pareto no espaço (degree_JS, SPS) por modelo."""
    pareto_rows = []
    for (model, frac), group in df.groupby(["model", "sample_frac"]):
        means = group.groupby("sampler")[["degree_JS", "SPS"]].mean().reset_index().dropna()
        if means.empty:
            continue
        costs = means[["degree_JS", "SPS"]].values
        mask = is_pareto_efficient(costs)
        for i, row in means.iterrows():
            if mask[i]:
                pareto_rows.append({
                    "model": model,
                    "sample_frac": frac,
                    "sampler": row["sampler"],
                    "degree_JS": row["degree_JS"],
                    "SPS": row["SPS"],
                })

    if pareto_rows:
        save_csv(pd.DataFrame(pareto_rows), results_path(config, "processed", "advanced_pareto.csv"))


if __name__ == "__main__":
    main()
