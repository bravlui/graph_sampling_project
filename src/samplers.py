"""
samplers.py — Métodos de amostragem de redes
=============================================

Implementa seis métodos clássicos de amostragem, cobrindo um espectro que vai
do puramente aleatório (random node) ao orientado a estrutura (MHRW). A escolha
do método impacta diretamente quais propriedades topológicas são preservadas na
amostra — tema central investigado neste projeto (Stumpf et al., 2005).

Métodos implementados
---------------------
1. Random Node Sampling  — seleção i.i.d. de nós; referência imparcial para atributos.
2. Random Edge Sampling  — seleção aleatória de arestas; favorece nós de alto grau.
3. Snowball Sampling     — expansão BFS a partir de sementes; captura vizinhança local.
4. Random Walk Sampling  — caminhada de Markov; estacionária proporcional ao grau.
5. Preferential RW       — caminhada com atração por grau; amplifica viés do RW.
6. Metropolis-Hastings RW — caminhada com correção MH; estacionária aproximadamente uniforme.

Complementos (aprofundamento)
7. GOAS  — exploração orientada a objetivo via softmax sobre fronteira.
8. GOAS-MH  — GOAS com correção MH e adaptação dinâmica dos pesos.

Cada função retorna (subgraph: nx.Graph, metadata: dict).
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
    """
    Seleciona nós uniformemente ao acaso (i.i.d.) e retorna subgrafo induzido.

    Por ser uma amostragem independente e identicamente distribuída, preserva
    distribuições de atributos nodais (e.g., grau esperado), mas não garante
    conectividade nem captura estruturas locais como triângulos ou comunidades.
    Funciona como linha de base neutra para comparação com métodos estruturados.
    """
    rng = np.random.RandomState(seed)
    target = _target_size(G, sample_frac)
    nodes = list(G.nodes())
    chosen = rng.choice(nodes, size=min(target, len(nodes)), replace=False)
    return _finalize_sample(G, set(chosen), "random_node")


# ============================================================================
# 2. Random Edge Sampling
# ============================================================================
def random_edge_sampling(G, sample_frac, seed):
    """
    Seleciona arestas ao acaso até cobrir ~sample_frac dos nós.

    Um nó de grau k tem probabilidade proporcional a k de ser incluído via
    arestas incidentes. Isso introduz viés em favor de hubs, super-representando
    nós de alto grau em relação à distribuição original — um efeito oposto ao
    random node, onde todos os nós têm a mesma chance de seleção.
    """
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
    """
    Expansão BFS a partir de múltiplos nós semente até atingir a fração alvo.

    Preserva bem a estrutura de vizinhança local: triângulos e comunidades densas
    tendem a ser capturados integralmente, pois a expansão segue as arestas reais.
    O efeito colateral é que regiões periféricas da rede ficam sub-representadas —
    o "snowball" pára ao atingir o alvo, sem cobrir toda a rede.
    Quando a BFS esgota componentes antes de atingir o alvo, novos sementes
    aleatórios são adicionados para garantir a fração desejada.
    """
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
    """
    Caminhada aleatória com possibilidade de restart (teleportação).

    A distribuição estacionária de uma caminhada aleatória simples sobre um grafo
    não-direcionado é proporcional ao grau: π(v) ∝ deg(v). Isso significa que nós
    de alto grau são visitados com mais frequência, inflando a média de grau da
    amostra em relação à rede original. O restart (prob=0.15) garante que a
    caminhada não fique presa em componentes isolados, mas não corrige o viés de grau.
    """
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
    """
    Caminhada aleatória com preferência por nós de alto grau (alpha controla o viés).

    A probabilidade de mover para o vizinho v é proporcional a deg(v)^alpha.
    Com alpha=1 (padrão), o viés é análogo ao modelo de Barabási-Albert: hubs
    atraem mais visitas. Alpha > 1 amplifica esse efeito; alpha=0 reduz ao random
    walk padrão. Útil para comparar como a intensidade do viés de grau afeta a
    preservação de distribuições como betweenness e eigenvector centrality.
    """
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
    """
    Random Walk com correção Metropolis-Hastings para estacionária aproximadamente uniforme.

    O viés de grau do random walk simples (π(v) ∝ deg(v)) é corrigido aceitando a
    transição u→v com probabilidade min(1, deg(u)/deg(v)). Se o vizinho candidato
    tem grau maior que o nó atual, a transição pode ser rejeitada — o caminhante
    permanece em u. Isso compensa a maior conectividade de hubs. Na teoria de cadeias
    de Markov, esse critério garante o balanço detalhado com a distribuição uniforme
    (Leskovec & Faloutsos, 2006). Na prática, para grafos esparsos com caudas longas
    de grau (como BA), a convergência é mais lenta do que em redes homogêneas (como ER).
    """
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
    Sampler orientado a objetivo: prioriza nós da fronteira segundo critérios estruturais.

    Mantém um conjunto de fronteira (vizinhos ainda não amostrados) e a cada passo
    escolhe o próximo nó via softmax sobre scores ponderados por objetivo:
    - degree: favorece hubs (útil para preservar distribuição de grau em BA)
    - clustering: favorece triângulos (útil para redes WS com alto clustering)
    - shortest_path: favorece pontes entre componentes (bridge score)
    - centrality: favorece nós com alto k-core (núcleo denso)
    - balanced: pesos iguais para todos os critérios

    O parâmetro `exploration` controla a fração de saltos aleatórios para fora da
    fronteira, evitando que o sampler fique preso em um único cluster denso.
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
# 8. Adaptive GOAS with Metropolis-Hastings (GOAS v2)
# ============================================================================
def goas_mh_adaptive(
    G, sample_frac, seed,
    objective="degree", restart_prob=0.10, exploration=0.05
):
    """
    GOAS v2: Adaptativo no tempo, ciente do objetivo e com correção Metropolis-Hastings.
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
    weights = weight_map.get(objective, weight_map["balanced"]).copy()

    start_node = rng.choice(nodes)
    sampled = {start_node}
    frontier = set()
    for nb in G.neighbors(start_node):
        if nb not in sampled:
            frontier.add(nb)

    random_jumps = 0
    current_node = start_node
    
    # Parâmetros de adaptação
    update_interval = max(1, int(target * 0.10)) # Atualiza a cada 10%
    eta = 0.5 # Learning rate para weights

    def estimate_divergence(sub_nodes):
        """Proxy muito simples para achar a pior métrica atual na amostra"""
        if not sub_nodes: return "degree"
        sub = G.subgraph(list(sub_nodes))
        # Grau médio vs original
        samp_deg = np.mean([degree_dict[v] for v in sub_nodes])
        orig_deg = np.mean(list(degree_dict.values()))
        deg_error = abs(samp_deg - orig_deg) / max(orig_deg, 1)
        
        # Clustering
        try:
            samp_clust = nx.average_clustering(sub)
        except:
            samp_clust = 0
        orig_clust = np.mean(list(clustering_dict.values()))
        clust_error = abs(samp_clust - orig_clust) / max(orig_clust, 0.001)
        
        return "degree" if deg_error > clust_error else "clustering"

    while len(sampled) < target:
        # Adaptação dinâmica a cada intervalo
        if len(sampled) % update_interval == 0:
            worst_metric = estimate_divergence(sampled)
            weights[worst_metric] += eta
            
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
                d_score = (degree_dict.get(v, 0) / max_degree)
                c_score = clustering_dict.get(v, 0.0)
                sampled_nb = sum(1 for nb in G.neighbors(v) if nb in sampled)
                total_nb = max(degree_dict.get(v, 1), 1)
                b_score = (1.0 - sampled_nb / total_nb)
                k_score = core_dict.get(v, 0) / max_core
                
                # Formula de score
                scores[i] = (
                    weights["degree"] * d_score
                    + weights["clustering"] * c_score
                    + weights["bridge"] * b_score
                    + weights["centrality"] * k_score
                )
            
            scores = scores - scores.max()
            exp_scores = np.exp(scores) # T=1
            
            # Correção Metropolis-Hastings (MH(u->v))
            mh_corrections = np.zeros(len(frontier_list))
            for i, v in enumerate(frontier_list):
                mh_corrections[i] = min(1.0, max(degree_dict[current_node], 1) / max(degree_dict[v], 1))
            
            # Probabilidade final
            probs_unnormalized = exp_scores * mh_corrections
            total = probs_unnormalized.sum()
            probs = probs_unnormalized / total if total > 0 else np.ones(len(frontier_list)) / len(frontier_list)
            
            idx = rng.choice(len(frontier_list), p=probs)
            chosen = frontier_list[idx]

        sampled.add(chosen)
        current_node = chosen
        frontier.discard(chosen)
        for nb in G.neighbors(chosen):
            if nb not in sampled:
                frontier.add(nb)

    subgraph, metadata = _finalize_sample(G, sampled, "goas_mh_adaptive")
    metadata.update({"objective": objective, "weights_final": weights, "random_jumps": random_jumps})
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
    "goas_mh_adaptive": goas_mh_adaptive,
}


def get_sampler(name):
    """Retorna função de sampler pelo nome."""
    if name not in _SAMPLER_REGISTRY:
        raise ValueError(f"Sampler '{name}' não encontrado. Disponíveis: {list(_SAMPLER_REGISTRY.keys())}")
    return _SAMPLER_REGISTRY[name]


def list_samplers():
    """Retorna lista de nomes de samplers disponíveis."""
    return list(_SAMPLER_REGISTRY.keys())
