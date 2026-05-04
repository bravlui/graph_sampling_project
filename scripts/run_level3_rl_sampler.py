#!/usr/bin/env python3
"""
run_level3_rl_sampler.py — Self-Supervised Graph Sampling via RL
================================================================

Treina DQN, avalia vs baselines, testa generalização e interpretabilidade.

Uso:
    python scripts/run_level3_rl_sampler.py --config configs/default.yaml
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, setup_logging, ensure_dirs, set_global_seed, results_path, save_csv
from src.graph_generators import generate_er_graph, generate_ba_graph, generate_ws_graph
from src.samplers import get_sampler
from src.metrics import compute_node_distributions, compute_global_metrics
from src.distances import compare_all_distributions, compare_global_metrics
from src.experiments import compute_sps
from src.visualization import _get_color

try:
    import torch
    from src.rl_sampler import (
        GraphSamplingEnv, QNetwork, train_dqn_sampler,
        learned_rl_sampling, HAS_TORCH,
    )
except ImportError:
    HAS_TORCH = False


def _make_graph_generator(model, n, seed_base):
    """Cria função geradora de grafos para o treinamento RL."""
    counter = [0]

    def gen():
        counter[0] += 1
        s = seed_base + counter[0]
        if model == "ER":
            return generate_er_graph(n, avg_degree=10, seed=s)
        elif model == "BA":
            return generate_ba_graph(n, m=5, seed=s)
        elif model == "WS":
            return generate_ws_graph(n, k=10, p=0.3, seed=s)
        else:  # Mistura
            choice = np.random.choice(["ER", "BA", "WS"])
            return _make_graph_generator(choice, n, s)()
        return generate_er_graph(n, avg_degree=10, seed=s)

    return gen


def _evaluate_sampler(G, sample_frac, sampler_fn, seed, **kwargs):
    """Avalia um sampler e retorna SPS."""
    G_s, meta = sampler_fn(G=G, sample_frac=sample_frac, seed=seed, **kwargs)
    orig_dists = compute_node_distributions(G, approx_betweenness=True, seed=0)
    orig_globals = compute_global_metrics(G, approx_betweenness=True, seed=0)
    samp_dists = compute_node_distributions(G_s, approx_betweenness=True, seed=0)
    samp_globals = compute_global_metrics(G_s, approx_betweenness=True, seed=0)
    dist_comp = compare_all_distributions(orig_dists, samp_dists)
    global_comp = compare_global_metrics(orig_globals, samp_globals)
    return compute_sps(dist_comp, global_comp)


def main():
    parser = argparse.ArgumentParser(description="Nível 3 C3: RL Sampler")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--eval-only", action="store_true",
                        help="Apenas avaliar modelo existente")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    if not HAS_TORCH:
        logger.error("PyTorch não disponível. Abortando RL sampler.")
        return

    rl_cfg = config.get("rl", {}).get("training", {})
    eval_cfg = config.get("rl", {}).get("evaluation", {})

    model_save_path = results_path(config, "models", "level3_dqn_sampler.pt")

    if not args.eval_only:
        # ===== Treinamento =====
        logger.info("=== Treinando DQN Sampler ===")

        graph_size = rl_cfg.get("graph_size", 300)
        sample_frac = rl_cfg.get("sample_frac", 0.20)
        num_episodes = rl_cfg.get("num_episodes", 500)

        # Treinar em mistura de redes
        gen_fn = _make_graph_generator("MIX", graph_size, seed_base=42)

        q_net, training_log = train_dqn_sampler(
            graph_generator_fn=gen_fn,
            num_episodes=num_episodes,
            sample_frac=sample_frac,
            seed=42,
            gamma=rl_cfg.get("gamma", 0.95),
            lr=rl_cfg.get("lr", 1e-3),
            batch_size=rl_cfg.get("batch_size", 64),
            epsilon_start=rl_cfg.get("epsilon_start", 1.0),
            epsilon_end=rl_cfg.get("epsilon_end", 0.05),
            epsilon_decay=rl_cfg.get("epsilon_decay", 0.995),
            target_update_freq=rl_cfg.get("target_update_freq", 10),
            replay_buffer_size=rl_cfg.get("replay_buffer_size", 10000),
        )

        # Salvar modelo
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(q_net.state_dict(), model_save_path)
        logger.info(f"Modelo salvo: {model_save_path}")

        # Salvar log de treinamento
        df_log = pd.DataFrame(training_log)
        save_csv(df_log, results_path(config, "raw", "level3_rl_training_log.csv"))

        # Plot reward curve
        _plot_reward_curve(config, df_log)

    # ===== Avaliação =====
    logger.info("=== Avaliando RL Sampler ===")

    if not os.path.exists(model_save_path):
        logger.error("Modelo não encontrado. Treine primeiro.")
        return

    eval_results = []
    action_freq_results = []

    graph_sizes = eval_cfg.get("graph_sizes", [300, 500])
    sample_fracs = eval_cfg.get("sample_fracs", [0.10, 0.20, 0.30])
    eval_seeds = eval_cfg.get("seeds", [0, 1, 2])

    baselines = ["random_node", "snowball", "random_walk"]
    models = ["ER", "BA", "WS"]

    total = len(models) * len(graph_sizes) * len(sample_fracs) * len(eval_seeds) * (1 + len(baselines))
    pbar = tqdm(total=total, desc="RL Evaluation")

    for model_name in models:
        for gs in graph_sizes:
            for frac in sample_fracs:
                for es in eval_seeds:
                    # Gerar rede de teste
                    gen_fn = _make_graph_generator(model_name, gs, seed_base=1000 + es)
                    G_test = gen_fn()

                    # RL sampler
                    try:
                        G_rl, meta_rl = learned_rl_sampling(G_test, frac, model_save_path, seed=es)
                        sps_rl = _evaluate_sampler(
                            G_test, frac,
                            lambda G, sample_frac, seed, **kw: learned_rl_sampling(G, sample_frac, model_save_path, seed),
                            seed=es,
                        )
                        eval_results.append({
                            "model": model_name, "graph_size": gs,
                            "sample_frac": frac, "seed": es,
                            "sampler": "rl_sampler", "SPS": sps_rl,
                        })

                        # Frequência de ações
                        action_freq_results.append({
                            "model": model_name, "graph_size": gs,
                            "sample_frac": frac, "seed": es,
                            "action_0_degree": meta_rl["action_counts"][0],
                            "action_1_clustering": meta_rl["action_counts"][1],
                            "action_2_kcore": meta_rl["action_counts"][2],
                            "action_3_random_frontier": meta_rl["action_counts"][3],
                            "action_4_random_jump": meta_rl["action_counts"][4],
                        })
                    except Exception as e:
                        logger.warning(f"RL eval falhou: {e}")
                    pbar.update(1)

                    # Baselines
                    for bl_name in baselines:
                        try:
                            bl_fn = get_sampler(bl_name)
                            sps_bl = _evaluate_sampler(G_test, frac, bl_fn, seed=es)
                            eval_results.append({
                                "model": model_name, "graph_size": gs,
                                "sample_frac": frac, "seed": es,
                                "sampler": bl_name, "SPS": sps_bl,
                            })
                        except Exception as e:
                            logger.warning(f"Baseline {bl_name} eval falhou: {e}")
                        pbar.update(1)

    pbar.close()

    # Salvar resultados
    df_eval = pd.DataFrame(eval_results)
    save_csv(df_eval, results_path(config, "raw", "level3_rl_evaluation.csv"))

    df_actions = pd.DataFrame(action_freq_results)
    save_csv(df_actions, results_path(config, "processed", "level3_rl_action_frequency.csv"))

    # Generalização
    _analyze_generalization(config, df_eval)

    # Figuras
    _plot_rl_vs_baselines(config, df_eval)
    _plot_action_frequency(config, df_actions)

    logger.info("RL Sampler avaliação concluída.")


def _plot_reward_curve(config, df_log):
    """Curva de reward ao longo do treinamento."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Média móvel
    rewards = df_log["total_reward"].values
    window = min(50, len(rewards) // 5 + 1)
    if window > 1:
        ma = pd.Series(rewards).rolling(window).mean()
        ax.plot(df_log["episode"], rewards, alpha=0.3, color="#457B9D")
        ax.plot(df_log["episode"], ma, color="#E63946", linewidth=2, label=f"Média móvel ({window})")
    else:
        ax.plot(df_log["episode"], rewards, color="#457B9D")

    ax.set_xlabel("Episódio", fontsize=12)
    ax.set_ylabel("Reward Total", fontsize=12)
    ax.set_title("Curva de Treinamento DQN", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    filepath = results_path(config, "figures", "level3_rl_reward_curve.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_rl_vs_baselines(config, df_eval):
    """RL vs baselines SPS comparison."""
    if df_eval.empty:
        return

    means = df_eval.groupby("sampler")["SPS"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E74C3C" if s == "rl_sampler" else "#457B9D" for s in means.index]
    ax.barh(range(len(means)), means.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index)
    ax.set_xlabel("SPS Médio (menor = melhor)")
    ax.set_title("RL Sampler vs Baselines", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    filepath = results_path(config, "figures", "level3_rl_vs_baselines.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_action_frequency(config, df_actions):
    """Frequência de ações por modelo de rede."""
    if df_actions.empty:
        return

    action_cols = [c for c in df_actions.columns if c.startswith("action_")]
    models = df_actions["model"].unique()

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    action_labels = ["Degree", "Clustering", "K-core", "Random F.", "Random J."]
    colors_bar = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#264653"]

    for ax, model in zip(axes, models):
        sub = df_actions[df_actions["model"] == model]
        means = sub[action_cols].mean().values
        total = means.sum()
        if total > 0:
            means = means / total

        ax.bar(range(len(action_labels)), means, color=colors_bar, alpha=0.8)
        ax.set_xticks(range(len(action_labels)))
        ax.set_xticklabels(action_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Frequência")
        ax.set_title(f"Modelo: {model}", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Frequência de Ações do RL Sampler", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    filepath = results_path(config, "figures", "level3_rl_action_frequency.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _analyze_generalization(config, df_eval):
    """Analisa generalização do RL sampler entre modelos de rede."""
    if df_eval.empty:
        return

    rl_data = df_eval[df_eval["sampler"] == "rl_sampler"]
    if rl_data.empty:
        return

    gen_rows = []
    for model in rl_data["model"].unique():
        sub = rl_data[rl_data["model"] == model]
        gen_rows.append({
            "test_model": model,
            "sps_mean": sub["SPS"].mean(),
            "sps_std": sub["SPS"].std(),
            "n_samples": len(sub),
        })

    df_gen = pd.DataFrame(gen_rows)
    save_csv(df_gen, results_path(config, "processed", "level3_rl_generalization.csv"))


if __name__ == "__main__":
    main()
