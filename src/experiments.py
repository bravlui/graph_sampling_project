"""
experiments.py — Orquestrador do benchmark de amostragem
=========================================================

Executa o benchmark multicritério completo: para cada combinação de rede,
sampler, fração amostral e repetição, gera uma amostra, calcula as métricas
topológicas e compara com a rede original via KL, JS, Wasserstein, KS e SPS.

O Structural Preservation Score (SPS) agrega JS de distribuições de grau,
clustering e betweenness com erros relativos de métricas globais (caminho
médio, clustering médio, grau médio). Quanto menor o SPS, melhor a amostra
preserva a estrutura original.
"""

import os
import time
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from src.graph_generators import generate_graph_suite
from src.samplers import get_sampler
from src.metrics import compute_node_distributions, compute_global_metrics
from src.distances import compare_all_distributions, compare_global_metrics
from src.utils import results_path, save_csv

logger = logging.getLogger("graph_sampling")


def compute_sps(dist_row, global_row):
    """
    Structural Preservation Score (SPS) — métrica agregada de fidelidade da amostra.

    Combina, em pesos iguais, três divergências JS de distribuições (grau, clustering,
    betweenness) com três erros relativos de métricas globais (caminho médio, clustering
    médio, grau médio). O resultado é um único número: quanto menor, mais fiel à
    estrutura original. A escolha de pesos iguais é uma decisão de projeto consciente
    — alternativas ponderadas são exploradas na análise Pareto (ver pareto.py).

    Quanto menor o SPS, melhor preservação estrutural.
    """
    components = []

    # Distribuições JS
    for key in ["degree_JS", "clustering_JS", "betweenness_JS"]:
        val = dist_row.get(key, float("nan"))
        if not np.isnan(val):
            components.append(val)

    # Erros relativos globais
    for key in ["re_avg_shortest_path_lcc", "re_avg_clustering", "re_avg_degree"]:
        val = global_row.get(key, float("nan"))
        if not np.isnan(val):
            components.append(val)

    if len(components) == 0:
        return float("nan")

    return float(np.mean(components))


def run_single_experiment(
    G_original, graph_info, sampler_name, sample_frac,
    rep_seed, original_dists, original_globals,
    sampler_params=None,
):
    """
    Executa um único experimento de amostragem.

    Returns
    -------
    tuple (dist_row, global_row)
        Dicionários com os resultados.
    """
    sampler_fn = get_sampler(sampler_name)

    # Preparar kwargs
    kwargs = {"G": G_original, "sample_frac": sample_frac, "seed": rep_seed}
    if sampler_params and sampler_name in sampler_params:
        kwargs.update(sampler_params[sampler_name])

    # Gerar amostra
    try:
        G_sample, meta = sampler_fn(**kwargs)
    except Exception as e:
        logger.error(f"Sampler {sampler_name} falhou: {e}")
        return None, None

    # Métricas da amostra
    try:
        sampled_dists = compute_node_distributions(G_sample, approx_betweenness=True, seed=rep_seed)
        sampled_globals = compute_global_metrics(G_sample, approx_betweenness=True, seed=rep_seed)
    except Exception as e:
        logger.error(f"Métricas falharam para {sampler_name}: {e}")
        return None, None

    # Comparar distribuições
    dist_comparison = compare_all_distributions(original_dists, sampled_dists)
    global_comparison = compare_global_metrics(original_globals, sampled_globals)

    # Base info
    base = {
        "graph_id": graph_info["graph_id"],
        "model": graph_info["model"],
        "graph_seed": graph_info["seed"],
        "graph_n": G_original.number_of_nodes(),
        "sampler": sampler_name,
        "sample_frac": sample_frac,
        "rep_seed": rep_seed,
        "sampled_n": meta["sampled_n"],
        "sample_frac_actual": meta["sample_frac_actual"],
    }

    dist_row = {**base, **dist_comparison}
    global_row = {**base, **global_comparison}

    # SPS
    sps = compute_sps(dist_comparison, global_comparison)
    dist_row["SPS"] = sps
    global_row["SPS"] = sps

    return dist_row, global_row


def run_sampling_benchmark(config):
    """
    Executa o benchmark multicritério de amostragem de redes.

    Para cada grafo da suíte, calcula as métricas da rede original uma única vez,
    depois itera sobre todas as combinações (sampler, fração, repetição). O resultado
    é salvo em dois CSVs: distâncias entre distribuições e erros de métricas globais.

    Parameters
    ----------
    config : dict
        Configuração completa carregada do YAML (graphs, sampling, paths).

    Returns
    -------
    tuple (pd.DataFrame, pd.DataFrame)
        DataFrames de distâncias de distribuições e erros de métricas globais.
    """
    logger.info("=== Benchmark de Amostragem de Redes ===")

    # Gerar suíte de grafos
    suite = generate_graph_suite(config)
    logger.info(f"Grafos gerados: {len(suite)}")

    # Parâmetros de amostragem
    sampling_cfg = config.get("sampling", {})
    fractions = sampling_cfg.get("fractions", [0.10, 0.20, 0.30])
    repetitions = sampling_cfg.get("repetitions", 5)
    sampler_names = sampling_cfg.get("samplers", ["random_node", "snowball", "random_walk"])
    sampler_params = config.get("sampler_params", {})

    # Total de experimentos
    total = len(suite) * len(fractions) * len(sampler_names) * repetitions
    logger.info(f"Total de experimentos: {total}")

    dist_rows = []
    global_rows = []

    pbar = tqdm(total=total, desc="Benchmark L1")

    for graph_info in suite:
        G = graph_info["graph"]
        logger.info(f"Processando {graph_info['graph_id']} (n={G.number_of_nodes()})")

        # Calcular métricas da rede original
        try:
            original_dists = compute_node_distributions(G, approx_betweenness=True, seed=0)
            original_globals = compute_global_metrics(G, approx_betweenness=True, seed=0)
        except Exception as e:
            logger.error(f"Erro nas métricas originais de {graph_info['graph_id']}: {e}")
            pbar.update(len(fractions) * len(sampler_names) * repetitions)
            continue

        for frac in fractions:
            for sname in sampler_names:
                for rep in range(repetitions):
                    rep_seed = graph_info["seed"] * 1000 + rep

                    dist_row, global_row = run_single_experiment(
                        G, graph_info, sname, frac,
                        rep_seed, original_dists, original_globals,
                        sampler_params=sampler_params,
                    )

                    if dist_row is not None:
                        dist_rows.append(dist_row)
                    if global_row is not None:
                        global_rows.append(global_row)

                    pbar.update(1)

    pbar.close()

    # Criar DataFrames
    df_dist = pd.DataFrame(dist_rows) if dist_rows else pd.DataFrame()
    df_global = pd.DataFrame(global_rows) if global_rows else pd.DataFrame()

    # Salvar CSVs brutos
    if not df_dist.empty:
        save_csv(df_dist, results_path(config, "raw", "distribution_distances.csv"))
    if not df_global.empty:
        save_csv(df_global, results_path(config, "raw", "global_errors.csv"))

    # Gerar resumos
    _generate_summaries(config, df_dist, df_global)

    logger.info("=== Benchmark de amostragem concluído ===")
    return df_dist, df_global


def _generate_summaries(config, df_dist, df_global):
    """Gera CSVs de resumo processados."""
    if df_dist.empty:
        return

    # Summary by sampler
    group_cols = ["sampler"]
    numeric_cols = df_dist.select_dtypes(include=[np.number]).columns.tolist()
    summary = df_dist.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    save_csv(summary, results_path(config, "processed", "summary_by_sampler.csv"))

    # Best sampler by metric
    metric_cols = [c for c in df_dist.columns if c.endswith("_JS") or c.endswith("_KL")]
    if "SPS" in df_dist.columns:
        metric_cols.append("SPS")

    best_rows = []
    for mc in metric_cols:
        if mc in df_dist.columns:
            means = df_dist.groupby("sampler")[mc].mean()
            best = means.idxmin()
            best_rows.append({"metric": mc, "best_sampler": best, "value": means[best]})

    if best_rows:
        df_best = pd.DataFrame(best_rows)
        save_csv(df_best, results_path(config, "processed", "best_sampler_by_metric.csv"))
