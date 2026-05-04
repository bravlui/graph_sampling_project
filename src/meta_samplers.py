"""
meta_samplers.py — Amostradores Baseados em Meta-Heurísticas (Nível 4)
======================================================================

Implementa:
1. PSO-GOAS: Otimização por Enxame de Partículas para hiperparâmetros do GOAS.
2. ACO Sampler: Amostragem de grafos usando Colônia de Formigas baseada em caminhos.
"""

import numpy as np
import networkx as nx
import logging
from src.samplers import goal_oriented_adaptive_sampling
from src.metrics import compute_node_distributions, compute_global_metrics
from src.distances import compare_all_distributions, compare_global_metrics
from src.experiments import compute_sps

logger = logging.getLogger("graph_sampling")

# ============================================================================
# 1. PSO-GOAS (Particle Swarm Optimization for GOAS)
# ============================================================================
def pso_goas(G, sample_frac, seed=42, n_particles=5, n_iterations=3):
    """
    Usa PSO simplificado em uma amostragem proxy (uma fração pequena ou o próprio grafo)
    para encontrar os melhores pesos para o GOAS (alpha_degree, alpha_clustering, alpha_bridge).
    Devido ao custo, deve usar parâmetros baixos.
    """
    rng = np.random.RandomState(seed)
    
    # 3 Dimensions: degree, clustering, bridge
    # Bounds: [0, 5]
    bounds = (0.0, 5.0)
    
    particles = rng.uniform(bounds[0], bounds[1], (n_particles, 3))
    velocities = np.zeros((n_particles, 3))
    
    personal_best_pos = particles.copy()
    personal_best_sps = np.full(n_particles, np.inf)
    
    global_best_pos = None
    global_best_sps = np.inf
    
    # Pre-compute original distributions for SPS
    orig_dists = compute_node_distributions(G, approx_betweenness=True, seed=seed)
    orig_globals = compute_global_metrics(G, approx_betweenness=True, seed=seed)
    
    # PSO Hyperparameters
    w = 0.5 # inertia
    c1 = 1.5 # cognitive
    c2 = 1.5 # social
    
    # Run iterations
    for it in range(n_iterations):
        for i in range(n_particles):
            # Run GOAS with current particle's weights
            p = particles[i]
            sub, _ = goal_oriented_adaptive_sampling(
                G, sample_frac, seed=seed+it+i, objective="balanced",
                alpha_degree=p[0], alpha_clustering=p[1], alpha_bridge=p[2]
            )
            
            # Evaluate SPS
            try:
                samp_dists = compute_node_distributions(sub, approx_betweenness=True, seed=seed)
                samp_globals = compute_global_metrics(sub, approx_betweenness=True, seed=seed)
                d_comp = compare_all_distributions(orig_dists, samp_dists)
                g_comp = compare_global_metrics(orig_globals, samp_globals)
                sps = compute_sps(d_comp, g_comp)
                if np.isnan(sps): sps = np.inf
            except Exception as e:
                logger.warning(f"Erro SPS no PSO: {e}")
                sps = np.inf
                
            # Update personal best
            if sps < personal_best_sps[i]:
                personal_best_sps[i] = sps
                personal_best_pos[i] = p.copy()
                
            # Update global best
            if sps < global_best_sps:
                global_best_sps = sps
                global_best_pos = p.copy()
                
        # Update velocities and positions
        for i in range(n_particles):
            r1, r2 = rng.rand(2)
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (personal_best_pos[i] - particles[i]) + 
                             c2 * r2 * (global_best_pos - particles[i]))
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])

    # Run final sampling with best weights
    if global_best_pos is None: global_best_pos = [1.0, 1.0, 1.0] # Fallback
    
    sub, metadata = goal_oriented_adaptive_sampling(
        G, sample_frac, seed=seed, objective="balanced",
        alpha_degree=global_best_pos[0], 
        alpha_clustering=global_best_pos[1], 
        alpha_bridge=global_best_pos[2]
    )
    
    metadata.update({
        "sampler": "pso_goas",
        "best_weights": list(global_best_pos),
        "best_pso_sps": float(global_best_sps)
    })
    
    return sub, metadata


# ============================================================================
# 2. ACO Sampler (Ant Colony Optimization)
# ============================================================================
def aco_sampler(G, sample_frac, seed=42, n_ants=5, n_iterations=3):
    """
    Amostrador inspirado em Colônia de Formigas.
    Cada formiga constrói uma amostra. O SPS da amostra vira feromônio
    que influencia as próximas formigas.
    """
    rng = np.random.RandomState(seed)
    target_n = max(1, int(round(G.number_of_nodes() * sample_frac)))
    nodes = list(G.nodes())
    
    # Initialize Pheromones on Nodes
    pheromones = {n: 1.0 for n in nodes}
    
    # Heuristics (Degree + Clustering)
    degree_dict = dict(G.degree())
    max_deg = max(degree_dict.values()) if degree_dict else 1
    clust_dict = nx.clustering(G)
    
    heuristics = {}
    for n in nodes:
        heuristics[n] = (degree_dict[n] / max_deg) + clust_dict[n] + 0.1
        
    alpha = 1.0 # Pheromone importance
    beta = 2.0  # Heuristic importance
    evaporation_rate = 0.2
    
    best_sample = None
    best_sps = np.inf
    
    orig_dists = compute_node_distributions(G, approx_betweenness=True, seed=seed)
    orig_globals = compute_global_metrics(G, approx_betweenness=True, seed=seed)
    
    for it in range(n_iterations):
        samples_in_it = []
        sps_in_it = []
        
        for ant in range(n_ants):
            # Formiga constrói solução (random walk bias com feromônio)
            start_node = rng.choice(nodes)
            sampled = {start_node}
            frontier = set(G.neighbors(start_node))
            
            while len(sampled) < target_n:
                if not frontier:
                    # Random jump
                    remaining = list(set(nodes) - sampled)
                    if not remaining: break
                    chosen = rng.choice(remaining)
                else:
                    f_list = list(frontier)
                    probs = []
                    for v in f_list:
                        tau = pheromones[v] ** alpha
                        eta = heuristics[v] ** beta
                        probs.append(tau * eta)
                        
                    probs = np.array(probs)
                    total = probs.sum()
                    if total == 0:
                        probs = np.ones(len(f_list)) / len(f_list)
                    else:
                        probs = probs / total
                        
                    chosen = f_list[rng.choice(len(f_list), p=probs)]
                    
                sampled.add(chosen)
                frontier.discard(chosen)
                for nb in G.neighbors(chosen):
                    if nb not in sampled:
                        frontier.add(nb)
                        
            # Evaluate this ant's sample
            sub = G.subgraph(list(sampled)).copy()
            sub = nx.convert_node_labels_to_integers(sub, first_label=0)
            
            try:
                samp_dists = compute_node_distributions(sub, approx_betweenness=True, seed=seed)
                samp_globals = compute_global_metrics(sub, approx_betweenness=True, seed=seed)
                d_comp = compare_all_distributions(orig_dists, samp_dists)
                g_comp = compare_global_metrics(orig_globals, samp_globals)
                sps = compute_sps(d_comp, g_comp)
                if np.isnan(sps): sps = np.inf
            except:
                sps = np.inf
                
            samples_in_it.append(list(sampled))
            sps_in_it.append(sps)
            
            if sps < best_sps:
                best_sps = sps
                best_sample = list(sampled)
                
        # Evaporate pheromones
        for n in pheromones:
            pheromones[n] *= (1.0 - evaporation_rate)
            
        # Pheromone Update (Deposit) - only the best ant of iteration deposits
        min_sps_idx = np.argmin(sps_in_it)
        best_ant_sps = sps_in_it[min_sps_idx]
        best_ant_sample = samples_in_it[min_sps_idx]
        
        # Deposited amount inversely proportional to SPS
        deposit = 1.0 / (best_ant_sps + 1e-5)
        for n in best_ant_sample:
            pheromones[n] += deposit

    # Return the global best sample
    if best_sample is None:
        best_sample = [nodes[0]] # Fail-safe
        
    subgraph = G.subgraph(best_sample).copy()
    subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0)
    
    metadata = {
        "original_n": G.number_of_nodes(),
        "sampled_n": subgraph.number_of_nodes(),
        "sample_frac_actual": subgraph.number_of_nodes() / G.number_of_nodes(),
        "sampler": "aco_sampler",
        "best_sps": float(best_sps)
    }
    
    return subgraph, metadata
