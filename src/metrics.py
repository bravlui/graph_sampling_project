"""
metrics.py — Métricas topológicas de centralidade e estrutura
==============================================================

Para comparar uma rede amostrada com a original, calculamos seis propriedades nodais
(distribuições) e um conjunto de métricas globais escalares. Cada propriedade captura
um aspecto diferente da estrutura:

- Grau            : conectividade local; discrimina ER (Poisson) de BA (lei de potência).
- Clustering      : densidade de triângulos na vizinhança; alto em WS, baixo em ER e BA.
- Betweenness     : fração de caminhos mais curtos que passam por um nó; revela gargalos.
- Eigenvector     : influência global via potência do autovetor dominante; correlaciona com hubs.
- Closeness       : inverso do caminho médio ao restante da rede; alto em nós centrais.
- K-core          : maior subestrutura densa da qual o nó participa; indicador de núcleo.

Betweenness e eigenvector são custosos para redes grandes. Para n > 500, usamos
betweenness aproximada (k=200 pivôs) e aumentamos max_iter do eigenvector.
"""

import numpy as np
import networkx as nx
import logging

logger = logging.getLogger("graph_sampling")


def safe_largest_connected_component(G):
    """Retorna maior componente conectado."""
    if nx.is_connected(G):
        return G
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def compute_node_distributions(G, approx_betweenness=True, seed=0):
    """
    Calcula as seis distribuições nodais usadas na comparação original × amostra.

    Para betweenness com n > 500, usa estimativa por k=200 amostras aleatórias de
    pivôs, com erro padrão O(1/sqrt(k)) — suficiente para comparação de distribuições
    mas não para valores exatos por nó. Eigenvector pode falhar a convergir em redes
    de componentes múltiplos; nesses casos retorna zeros.

    Returns
    -------
    dict[str, np.ndarray]
        Chaves: degree, clustering, betweenness, eigenvector, closeness, k_core.
    """
    n = G.number_of_nodes()
    distributions = {}

    # Degree
    distributions["degree"] = np.array([d for _, d in G.degree()], dtype=float)

    # Clustering
    cc = nx.clustering(G)
    distributions["clustering"] = np.array(list(cc.values()), dtype=float)

    # Betweenness (aproximação para redes grandes)
    try:
        if approx_betweenness and n > 500:
            k = min(200, n)
            bc = nx.betweenness_centrality(G, k=k, seed=seed)
        else:
            bc = nx.betweenness_centrality(G)
        distributions["betweenness"] = np.array(list(bc.values()), dtype=float)
    except Exception as e:
        logger.warning(f"Betweenness falhou: {e}")
        distributions["betweenness"] = np.zeros(n)

    # Eigenvector centrality
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        distributions["eigenvector"] = np.array(list(ec.values()), dtype=float)
    except (nx.NetworkXError, nx.PowerIterationFailedConvergence) as e:
        logger.warning(f"Eigenvector centrality falhou: {e}")
        distributions["eigenvector"] = np.zeros(n)

    # Closeness
    try:
        cl = nx.closeness_centrality(G)
        distributions["closeness"] = np.array(list(cl.values()), dtype=float)
    except Exception as e:
        logger.warning(f"Closeness falhou: {e}")
        distributions["closeness"] = np.zeros(n)

    # K-core
    try:
        kc = nx.core_number(G)
        distributions["k_core"] = np.array(list(kc.values()), dtype=float)
    except Exception as e:
        logger.warning(f"K-core falhou: {e}")
        distributions["k_core"] = np.zeros(n)

    return distributions


def compute_global_metrics(G, approx_betweenness=True, seed=0):
    """
    Calcula métricas globais da rede.

    Returns
    -------
    dict
        Métricas globais: n_nodes, n_edges, density, avg_degree,
        degree_var, degree_entropy, avg_clustering, transitivity,
        assortativity, avg_shortest_path_lcc, diameter_lcc,
        avg_betweenness, avg_eigenvector, max_k_core, num_connected_components
    """
    from scipy.stats import entropy as sp_entropy

    n = G.number_of_nodes()
    m = G.number_of_edges()
    metrics = {}

    metrics["n_nodes"] = n
    metrics["n_edges"] = m
    metrics["density"] = nx.density(G) if n > 1 else 0.0

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    metrics["avg_degree"] = float(degrees.mean()) if len(degrees) > 0 else 0.0
    metrics["degree_var"] = float(degrees.var()) if len(degrees) > 0 else 0.0

    # Degree entropy
    if len(degrees) > 0:
        deg_counts = np.bincount(degrees.astype(int))
        deg_probs = deg_counts / deg_counts.sum()
        deg_probs = deg_probs[deg_probs > 0]
        metrics["degree_entropy"] = float(sp_entropy(deg_probs, base=2))
    else:
        metrics["degree_entropy"] = 0.0

    metrics["avg_clustering"] = nx.average_clustering(G)
    metrics["transitivity"] = nx.transitivity(G)

    try:
        metrics["assortativity"] = nx.degree_assortativity_coefficient(G)
        if np.isnan(metrics["assortativity"]):
            metrics["assortativity"] = 0.0
    except Exception:
        metrics["assortativity"] = 0.0

    # Shortest path e diameter no LCC
    lcc = safe_largest_connected_component(G)
    n_lcc = lcc.number_of_nodes()
    try:
        if n_lcc <= 5000:
            metrics["avg_shortest_path_lcc"] = nx.average_shortest_path_length(lcc)
        else:
            # Aproximação por amostragem
            rng = np.random.RandomState(seed)
            sample_nodes = rng.choice(list(lcc.nodes()), size=min(500, n_lcc), replace=False)
            total = 0.0
            count = 0
            for src in sample_nodes:
                lengths = nx.single_source_shortest_path_length(lcc, src)
                total += sum(lengths.values())
                count += len(lengths)
            metrics["avg_shortest_path_lcc"] = total / max(count, 1)
    except Exception:
        metrics["avg_shortest_path_lcc"] = float("nan")

    try:
        if n_lcc <= 3000:
            metrics["diameter_lcc"] = nx.diameter(lcc)
        else:
            metrics["diameter_lcc"] = float("nan")
    except Exception:
        metrics["diameter_lcc"] = float("nan")

    # Betweenness
    try:
        if approx_betweenness and n > 500:
            k = min(200, n)
            bc = nx.betweenness_centrality(G, k=k, seed=seed)
        else:
            bc = nx.betweenness_centrality(G)
        metrics["avg_betweenness"] = float(np.mean(list(bc.values())))
    except Exception:
        metrics["avg_betweenness"] = 0.0

    # Eigenvector
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        metrics["avg_eigenvector"] = float(np.mean(list(ec.values())))
    except Exception:
        metrics["avg_eigenvector"] = 0.0

    # K-core
    try:
        kc = nx.core_number(G)
        metrics["max_k_core"] = max(kc.values()) if kc else 0
    except Exception:
        metrics["max_k_core"] = 0

    metrics["num_connected_components"] = nx.number_connected_components(G)

    return metrics


def compute_graph_feature_vector(G, seed=0):
    """
    Retorna vetor de features para uso em PCA e meta-modelo.

    Returns
    -------
    dict
        Features numéricas da rede.
    """
    gm = compute_global_metrics(G, approx_betweenness=True, seed=seed)
    # Selecionar features numéricas relevantes
    feature_keys = [
        "n_nodes", "n_edges", "density", "avg_degree", "degree_var",
        "degree_entropy", "avg_clustering", "transitivity", "assortativity",
        "avg_shortest_path_lcc", "max_k_core",
    ]
    features = {}
    for k in feature_keys:
        val = gm.get(k, 0.0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        features[k] = val
    return features
