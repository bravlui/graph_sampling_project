"""
run_level4_advanced.py — Benchmarking Nível 4 (Samplers Avançados)
===================================================================

Executa bateria de testes comparando samplers baselines contra
GOAS Adaptativo (v2), Contextual Bandit, PSO-GOAS e ACO Sampler.
Inclui avaliações específicas de objetivo e testes estatísticos (Wilcoxon).
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_generators import generate_graph_suite
from src.samplers import get_sampler
from src.bandit_sampler import contextual_bandit_sampling
from src.meta_samplers import pso_goas, aco_sampler
from src.experiments import run_single_experiment
from src.metrics import compute_node_distributions, compute_global_metrics
from src.distances import compare_all_distributions, compare_global_metrics
from src.utils import setup_logging, results_path, save_csv
from src.pareto import is_pareto_efficient

logger = setup_logging()

# Registra novos samplers customizados nativamente caso não estejam no SAMPLER_REGISTRY padrão (wrappers)
def run_custom_sampler(G, graph_info, sampler_name, sample_frac, rep_seed, orig_dists, orig_globals, **kwargs):
    try:
        if sampler_name == "contextual_bandit":
            sub, meta = contextual_bandit_sampling(G, sample_frac, seed=rep_seed)
        elif sampler_name == "pso_goas":
            sub, meta = pso_goas(G, sample_frac, seed=rep_seed, n_particles=10, n_iterations=5)
        elif sampler_name == "aco_sampler":
            sub, meta = aco_sampler(G, sample_frac, seed=rep_seed, n_ants=10, n_iterations=5)
        else:
            dist_row, global_row = run_single_experiment(G, graph_info, sampler_name, sample_frac, rep_seed, orig_dists, orig_globals, sampler_params={sampler_name: kwargs})
            if dist_row is None: return None
            return {**dist_row, **global_row}
    except Exception as e:
        logger.error(f"Erro no sampler {sampler_name}: {e}")
        return None, None
        
    try:
        samp_dists = compute_node_distributions(sub, approx_betweenness=True, seed=rep_seed)
        samp_globals = compute_global_metrics(sub, approx_betweenness=True, seed=rep_seed)
        d_comp = compare_all_distributions(orig_dists, samp_dists)
        g_comp = compare_global_metrics(orig_globals, samp_globals)
        from src.experiments import compute_sps
        sps = compute_sps(d_comp, g_comp)
    except Exception as e:
        logger.error(f"Erro calculando distancias {sampler_name}: {e}")
        return None, None
        
    base = {
        "graph_id": graph_info["graph_id"], "model": graph_info["model"],
        "graph_seed": graph_info["seed"], "graph_n": G.number_of_nodes(),
        "sampler": sampler_name, "sample_frac": sample_frac,
        "rep_seed": rep_seed, "sampled_n": meta["sampled_n"],
        "SPS": sps
    }
    
    if "objective" in kwargs:
        base["objective"] = kwargs["objective"]
        
    if "action_counts" in meta:
        base["action_counts"] = str(meta["action_counts"])
        
    return {**base, **d_comp, **g_comp}


def main():
    logger.info("=== Nível 4: Advanced Samplers Benchmark ===")
    
    config = {
        "project": "graph_sampling",
        "output_dir": "results",
        "graphs": {
            "models": {
                "ER": {"avg_degree": 10},
                "BA": {"m": 4},
                "WS": {"k": 6, "p": 0.1}
            },
            "sizes": [1000, 3000],
            "seeds": [0, 1]
        }
    }
    
    suite = generate_graph_suite(config)
    fractions = [0.1, 0.2]
    seeds = 2
    
    baselines = ["random_node", "random_walk", "snowball", "goas"]
    advanced = ["contextual_bandit", "pso_goas", "aco_sampler"]
    
    # GOAS v2 variantes
    goas_v2_variants = [
        {"name": "goas_mh_degree", "objective": "degree"},
        {"name": "goas_mh_clustering", "objective": "clustering"},
        {"name": "goas_mh_balanced", "objective": "balanced"}
    ]
    
    rows = []
    total_iters = len(suite) * len(fractions) * seeds
    
    pbar = tqdm(total=total_iters, desc="Level 4")
    
    for graph_info in suite:
        G = graph_info["graph"]
        orig_dists = compute_node_distributions(G, approx_betweenness=True, seed=0)
        orig_globals = compute_global_metrics(G, approx_betweenness=True, seed=0)
        
        for frac in fractions:
            for s in range(seeds):
                rep_seed = graph_info["seed"] * 100 + s
                
                # Baselines
                for b in baselines:
                    row = run_custom_sampler(G, graph_info, b, frac, rep_seed, orig_dists, orig_globals)
                    if row: rows.append(row)
                    
                # Advanced
                for adv in advanced:
                    row = run_custom_sampler(G, graph_info, adv, frac, rep_seed, orig_dists, orig_globals)
                    if row: rows.append(row)
                    
                # GOAS v2 com objetivos específicos
                for v in goas_v2_variants:
                    row = run_custom_sampler(G, graph_info, "goas_mh_adaptive", frac, rep_seed, orig_dists, orig_globals, objective=v["objective"])
                    if row:
                        row["sampler"] = v["name"]
                        rows.append(row)
                        
                pbar.update(1)
                
    pbar.close()
    
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    df = pd.DataFrame(rows)
    save_csv(df, results_path(config, "raw", f"level4_all_results_{timestamp}.csv"))
    
    # --- AVALIAÇÃO DE OBJETIVOS ESPECÍFICOS E TESTES ESTATÍSTICOS ---
    logger.info("=== Executando Testes Estatísticos (Wilcoxon) ===")
    stats_rows = []
    
    metrics_map = {
        "goas_mh_degree": "degree_JS",
        "goas_mh_clustering": "clustering_JS",
        "goas_mh_balanced": "SPS"
    }
    
    for variant, target_metric in metrics_map.items():
        if variant not in df["sampler"].values: continue
        variant_data = df[df["sampler"] == variant]
        
        for baseline in baselines:
            if baseline not in df["sampler"].values: continue
            baseline_data = df[df["sampler"] == baseline]
            
            merged = pd.merge(
                variant_data, baseline_data, 
                on=["graph_id", "sample_frac", "rep_seed"], 
                suffixes=('_var', '_base')
            ).dropna(subset=[f"{target_metric}_var", f"{target_metric}_base"])
            
            if len(merged) < 5: continue
            
            x = merged[f"{target_metric}_var"].values
            y = merged[f"{target_metric}_base"].values
            
            # Wilcoxon: testa se variant é menor (melhor) que o baseline
            # Para JS e SPS, valores menores são melhores.
            stat, p_val = wilcoxon(x, y, alternative='less')
            
            # Effect size r = Z / sqrt(N)
            # Aproximação Z usando scipy stats (se N grande), para simplicidade apenas guardamos stat e p_val
            wins = np.sum(x < y)
            win_rate = wins / len(x)
            
            stats_rows.append({
                "variant": variant,
                "baseline": baseline,
                "metric": target_metric,
                "n_samples": len(x),
                "p_value": p_val,
                "significant": p_val < 0.05,
                "win_rate": win_rate
            })
            
    df_stats = pd.DataFrame(stats_rows)
    save_csv(df_stats, results_path(config, "processed", f"level4_objective_specific_results_{timestamp}.csv"))
    
    # --- FRONTEIRA DE PARETO ---
    logger.info("=== Calculando Fronteira de Pareto ===")
    
    pareto_rows = []
    for (model, frac), group in df.groupby(["model", "sample_frac"]):
        means = group.groupby("sampler")[["degree_JS", "clustering_JS", "SPS"]].mean().reset_index()
        means = means.dropna()
        if len(means) == 0: continue
        
        costs = means[["degree_JS", "SPS"]].values
        mask = is_pareto_efficient(costs)
        
        for i, row in means.iterrows():
            if mask[i]:
                pareto_rows.append({
                    "model": model, "sample_frac": frac, "sampler": row["sampler"],
                    "degree_JS": row["degree_JS"], "SPS": row["SPS"]
                })
                
    df_pareto = pd.DataFrame(pareto_rows)
    save_csv(df_pareto, results_path(config, "processed", f"level4_pareto_updated_{timestamp}.csv"))
    
    logger.info("Experimentos Concluídos com sucesso.")

if __name__ == "__main__":
    main()
