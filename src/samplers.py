"""
samplers.py — Métodos de amostragem de redes
=============================================

Implementa 6 métodos clássicos + GOAS (Goal-Oriented Adaptive Sampling).
Cada sampler retorna um subgrafo induzido relabelado e metadados.
"""

import networkx as nx
import numpy as np
import logging
from collections import deque

logger = logging.getLogger("graph_sampling")


def _finalize_sample(G_original, sampled_nodes, sampler_name):
    """Cria subgrafo induzido, relabela e gera metadados."""
    sampled_nodes = set(sampled_nodes) & set(G_original.nodes())
    if len(sampled_nodes) == 0:
        sampled_nodes = {list(G_original.nodes())[0]}
    subgraph = G_original.subgraph(sampled_nodes).copy()
    subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0)
    metadata = {
        "original_n": G_original.number_of_nodes(),
        "sampled_n": subgraph.number_of_nodes(),
        "sample_frac_actual": subgraph.number_of_nodes() / G_original.number_of_nodes(),
        "sampler": sampler_name,
    }
    return subgraph, metadata


def _target_size(G, sample_frac):
    return max(1, int(round(G.number_of_nodes() * sample_frac)))


# ============================================================================
# 1. Random Node Sampling
# ============================================================================
def random_node_sampling(G, sample_frac, seed):
    """Seleciona nós uniformemente ao acaso e retorna subgrafo induzido."""
    rng = np.random.RandomState(seed)
    target = _target_size(G, sample_frac)
    nodes = list(G.nodes())
    chosen = rng.choice(nodes, size=min(target, len(nodes)), replace=False)
    return _finalize_sample(G, set(chosen), "random_node")


# ============================================================================
# 2. Random Edge Sampling
# ============================================================================
def random_edge_sampling(G, sample_frac, seed):
    """Seleciona arestas ao acaso até cobrir ~sample_frac dos nós."""
    rng = np.random.RandomState(seed)
    target = _target_size(G, sample_frac)
    edges = list(G.edges())
    rng.shuffle(edges)
    sampled_nodes = set()
    for u, v in edges:
        sampled_nodes.add(u)
        sampled_nodes.add(v)
        if len(sampled_nodes) >= target:
            break
    return _finalize_sample(G, sampled_nodes, "random_edge")


# ============================================================================
# 3. Snowball Sampling
# ============================================================================
def snowball_sampling(G, sample_frac, seed, num_seeds=3):
    """Expansão BFS a partir de múltiplos nós semente."""
    rng = np.random.RandomState(seed)
    target = _target_size(G, sample_frac)
    nodes = list(G.nodes())
    seed_nodes = rng.choice(nodes, size=min(num_seeds, len(nodes)), replace=False)
    sampled = set(seed_nodes)
    queue = deque(seed_nodes)
    while len(sampled) < target and queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in sampled:
                sampled.add(neighbor)
                queue.append(neighbor)
                if len(sampled) >= target:
                    break
        if not queue and len(sampled) < target:
            remaining = list(set(nodes) - sampled)
            if remaining:
                new_seed = rng.choice(remaining)
                sampled.add(new_seed)
                queue.append(new_seed)
    return _finalize_sample(G, sampled, "snowball")


# ============================================================================
# 4. Random Walk Sampling
# ============================================================================
def random_walk_sampling(G, sample_frac, seed, restart_prob=0.15):
    """Caminhada aleatória com possibilidade de restart."""
    rng = np.random.RandomState(seed)
    target = _target_size(G, sample_frac)
    nodes = list(G.nodes())
    current = rng.choice(nodes)
    sampled = {current}
    for _ in range(target * 50):
        if len(sampled) >= target:
            break
        if rng.random() < restart_prob:
            current = rng.choice(nodes)
        else:
            neighbors = list(G.neighbors(current))
            if neighbors:
                current = neighbors[rng.randint(len(neighbors))]
            else:
                current = rng.choice(nodes)
        sampled.add(current)
    return _finalize_sample(G, sampled, "random_walk")


# ============================================================================
# 5. Preferential Random Walk Sampling
# ============================================================================
def preferential_random_walk_sampling(G, sample_frac, seed, restart_prob=0.15, alpha=1.0):
    """Caminhada onde P(vizinho v) proporcional a degree(v)^alpha."""
    rng = np.random.RandomState(seed)
    target = _target_size(G, sample_frac)
    nodes = list(G.nodes())
    degree_dict = dict(G.degree())
    current = rng.choice(nodes)
    sampled = {current}
    for _ in range(target * 50):
        if len(sampled) >= target:
            break
        if rng.random() < restart_prob:
            current = rng.choice(nodes)
        else:
            neighbors = list(G.neighbors(current))
            if neighbors:
                weights = np.array([degree_dict[v] ** alpha for v in neighbors], dtype=float)
                total = weights.sum()
                if total > 0:
                    probs = weights / total
                    current = neighbors[rng.choice(len(neighbors), p=probs)]
                else:
                    current = neighbors[rng.randint(len(neighbors))]
            else:
                current = rng.choice(nodes)
        sampled.add(current)
    return _finalize_sample(G, sampled, "preferential_random_walk")


# ============================================================================
# 6. Metropolis-Hastings Random Walk Sampling
# ============================================================================
def metropolis_hastings_random_walk_sampling(G, sample_frac, seed):
    """Random Walk com correção MH: aceitação min(1, deg(u)/deg(v))."""
    rng = np.random.RandomState(seed)
    target = _target_size(G, sample_frac)
    nodes = list(G.nodes())
    degree_dict = dict(G.degree())
    current = rng.choice(nodes)
    sampled = {current}
    for _ in range(target * 100):
        if len(sampled) >= target:
            break
        neighbors = list(G.neighbors(current))
        if not neighbors:
            current = rng.choice(nodes)
            sampled.add(current)
            continue
        candidate = neighbors[rng.randint(len(neighbors))]
        acceptance = min(1.0, max(degree_dict[current], 1) / max(degree_dict[candidate], 1))
        if rng.random() < acceptance:
            current = candidate
        sampled.add(current)
    return _finalize_sample(G, sampled, "metropolis_hastings_rw")


# ============================================================================
# 7. Goal-Oriented Adaptive Sampling (GOAS)
# ============================================================================
def goal_oriented_adaptive_sampling(
    G, sample_frac, seed,
    objective="degree", restart_prob=0.10,
    alpha_degree=1.0, alpha_clustering=1.0, alpha_bridge=1.0,
    exploration=0.05,
):
    """
    Sampler adaptativo orientado ao objetivo estrutural.

    Escolhe próximo nó da fronteira com base em scores ponderados
    pelo objetivo desejado (degree, clustering, shortest_path, centrality, balanced).
    """
    rng = np.random.RandomState(seed)
    target = _target_size(G, sample_frac)
    nodes = list(G.nodes())

    # Pré-computar propriedades
    degree_dict = dict(G.degree())
    max_degree = max(degree_dict.values()) if degree_dict else 1
    clustering_dict = nx.clustering(G)
    core_dict = nx.core_number(G)
    max_core = max(core_dict.values()) if core_dict else 1

    # Pesos por objetivo
    weight_map = {
        "degree":        {"degree": 3.0, "clustering": 0.5, "bridge": 0.5, "centrality": 1.0},
        "clustering":    {"degree": 0.5, "clustering": 3.0, "bridge": 0.5, "centrality": 0.5},
        "shortest_path": {"degree": 0.5, "clustering": 0.5, "bridge": 3.0, "centrality": 1.0},
        "centrality":    {"degree": 1.0, "clustering": 0.5, "bridge": 0.5, "centrality": 3.0},
        "balanced":      {"degree": 1.0, "clustering": 1.0, "bridge": 1.0, "centrality": 1.0},
    }
    weights = weight_map.get(objective, weight_map["balanced"])

    start_node = rng.choice(nodes)
    sampled = {start_node}
    frontier = set()
    for nb in G.neighbors(start_node):
        if nb not in sampled:
            frontier.add(nb)

    random_jumps = 0

    while len(sampled) < target:
        if rng.random() < exploration or not frontier:
            remaining = list(set(nodes) - sampled)
            if not remaining:
                break
            chosen = rng.choice(remaining)
            random_jumps += 1
        elif rng.random() < restart_prob:
            remaining = list(set(nodes) - sampled)
            if not remaining:
                break
            chosen = rng.choice(remaining)
            random_jumps += 1
        else:
            frontier_list = list(frontier)
            scores = np.zeros(len(frontier_list))
            for i, v in enumerate(frontier_list):
                d_score = (degree_dict.get(v, 0) / max_degree) * alpha_degree
                c_score = clustering_dict.get(v, 0.0) * alpha_clustering
                sampled_nb = sum(1 for nb in G.neighbors(v) if nb in sampled)
                total_nb = max(degree_dict.get(v, 1), 1)
                b_score = (1.0 - sampled_nb / total_nb) * alpha_bridge
                k_score = core_dict.get(v, 0) / max_core
                scores[i] = (
                    weights["degree"] * d_score
                    + weights["clustering"] * c_score
                    + weights["bridge"] * b_score
                    + weights["centrality"] * k_score
                )
            scores = scores - scores.max()
            exp_scores = np.exp(scores)
            total = exp_scores.sum()
            probs = exp_scores / total if total > 0 else np.ones(len(frontier_list)) / len(frontier_list)
            idx = rng.choice(len(frontier_list), p=probs)
            chosen = frontier_list[idx]

        sampled.add(chosen)
        frontier.discard(chosen)
        for nb in G.neighbors(chosen):
            if nb not in sampled:
                frontier.add(nb)

    subgraph, metadata = _finalize_sample(G, sampled, "goas")
    metadata.update({"objective": objective, "weights": weights, "random_jumps": random_jumps})
    return subgraph, metadata


# ============================================================================
# Registry
# ============================================================================
_SAMPLER_REGISTRY = {
    "random_node": random_node_sampling,
    "random_edge": random_edge_sampling,
    "snowball": snowball_sampling,
    "random_walk": random_walk_sampling,
    "preferential_random_walk": preferential_random_walk_sampling,
    "metropolis_hastings_rw": metropolis_hastings_random_walk_sampling,
    "goas": goal_oriented_adaptive_sampling,
}


def get_sampler(name):
    """Retorna função de sampler pelo nome."""
    if name not in _SAMPLER_REGISTRY:
        raise ValueError(f"Sampler '{name}' não encontrado. Disponíveis: {list(_SAMPLER_REGISTRY.keys())}")
    return _SAMPLER_REGISTRY[name]


def list_samplers():
    """Retorna lista de nomes de samplers disponíveis."""
    return list(_SAMPLER_REGISTRY.keys())
