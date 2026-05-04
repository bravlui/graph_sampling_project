"""
graph_generators.py — Geradores de redes sintéticas
====================================================

Funções para gerar redes Erdős-Rényi, Barabási-Albert, Watts-Strogatz
e LFR Benchmark, usadas no benchmark de amostragem.
"""

import networkx as nx
import numpy as np
import logging

logger = logging.getLogger("graph_sampling")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _largest_connected_component(G: nx.Graph) -> nx.Graph:
    """Retorna o maior componente conectado com nós relabelados de 0 a n-1."""
    if nx.is_connected(G):
        return nx.convert_node_labels_to_integers(G, first_label=0)
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc).copy()
    return nx.convert_node_labels_to_integers(subgraph, first_label=0)


# ---------------------------------------------------------------------------
# Geradores individuais
# ---------------------------------------------------------------------------

def generate_er_graph(n: int, avg_degree: float, seed: int) -> nx.Graph:
    """
    Gera rede Erdős-Rényi G(n, p).

    Parameters
    ----------
    n : int
        Número de nós.
    avg_degree : float
        Grau médio desejado; p = avg_degree / (n - 1).
    seed : int
        Semente para reprodutibilidade.

    Returns
    -------
    nx.Graph
        Maior componente conectado, relabelado de 0 a n-1.
    """
    if n < 2:
        raise ValueError(f"n deve ser >= 2, recebido {n}")
    p = avg_degree / (n - 1)
    p = min(p, 1.0)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return _largest_connected_component(G)


def generate_ba_graph(n: int, m: int, seed: int) -> nx.Graph:
    """
    Gera rede Barabási-Albert.

    Parameters
    ----------
    n : int
        Número de nós.
    m : int
        Número de arestas adicionadas por novo nó.
    seed : int
        Semente para reprodutibilidade.

    Returns
    -------
    nx.Graph
        Grafo conectado não direcionado.
    """
    if n < 2:
        raise ValueError(f"n deve ser >= 2, recebido {n}")
    if m < 1:
        raise ValueError(f"m deve ser >= 1, recebido {m}")
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return _largest_connected_component(G)


def generate_ws_graph(n: int, k: int, p: float, seed: int) -> nx.Graph:
    """
    Gera rede Watts-Strogatz.

    Parameters
    ----------
    n : int
        Número de nós.
    k : int
        Cada nó conecta-se a k vizinhos mais próximos no anel.
    p : float
        Probabilidade de reconexão.
    seed : int
        Semente para reprodutibilidade.

    Returns
    -------
    nx.Graph
        Maior componente conectado (geralmente todo o grafo).
    """
    if n < 2:
        raise ValueError(f"n deve ser >= 2, recebido {n}")
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return _largest_connected_component(G)


def generate_lfr_graph(
    n: int,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.3,
    avg_degree: float = 10,
    min_community: int = 20,
    seed: int = 0,
    max_attempts: int = 10,
) -> nx.Graph:
    """
    Gera rede LFR Benchmark (opcional, pode falhar).

    Tenta até max_attempts vezes com seeds diferentes.

    Returns
    -------
    nx.Graph or None
        Maior componente conectado ou None se todas as tentativas falharem.
    """
    for attempt in range(max_attempts):
        try:
            G = nx.LFR_benchmark_graph(
                n,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=avg_degree,
                min_community=min_community,
                seed=seed + attempt,
            )
            # Remover atributos de comunidade para simplificar
            G = nx.Graph(G)
            return _largest_connected_component(G)
        except (nx.ExceededMaxIterations, nx.NetworkXError) as e:
            logger.warning(f"LFR tentativa {attempt+1}/{max_attempts} falhou: {e}")
    logger.error(f"LFR falhou após {max_attempts} tentativas.")
    return None


# ---------------------------------------------------------------------------
# Suíte de grafos
# ---------------------------------------------------------------------------

def generate_graph_suite(config: dict) -> list:
    """
    Gera suíte completa de redes sintéticas a partir da configuração.

    Parameters
    ----------
    config : dict
        Configuração carregada do YAML (seção 'graphs').

    Returns
    -------
    list[dict]
        Lista de dicionários com:
        - graph_id : str
        - model : str
        - params : dict
        - seed : int
        - graph : nx.Graph
    """
    graph_cfg = config.get("graphs", config)
    models = graph_cfg.get("models", {})
    sizes = graph_cfg.get("sizes", [1000])
    seeds = graph_cfg.get("seeds", [0])

    suite = []

    for size in sizes:
        for seed in seeds:
            # ---------- Erdős-Rényi ----------
            if "ER" in models:
                er_params = models["ER"]
                avg_deg = er_params.get("avg_degree", 10)
                G = generate_er_graph(size, avg_deg, seed)
                suite.append({
                    "graph_id": f"ER_n{size}_s{seed}",
                    "model": "ER",
                    "params": {"n": size, "avg_degree": avg_deg},
                    "seed": seed,
                    "graph": G,
                })

            # ---------- Barabási-Albert ----------
            if "BA" in models:
                ba_params = models["BA"]
                m = ba_params.get("m", 5)
                G = generate_ba_graph(size, m, seed)
                suite.append({
                    "graph_id": f"BA_n{size}_s{seed}",
                    "model": "BA",
                    "params": {"n": size, "m": m},
                    "seed": seed,
                    "graph": G,
                })

            # ---------- Watts-Strogatz ----------
            if "WS" in models:
                ws_params = models["WS"]
                k = ws_params.get("k", 10)
                p = ws_params.get("p", 0.3)
                G = generate_ws_graph(size, k, p, seed)
                suite.append({
                    "graph_id": f"WS_n{size}_s{seed}",
                    "model": "WS",
                    "params": {"n": size, "k": k, "p": p},
                    "seed": seed,
                    "graph": G,
                })

    logger.info(f"Suíte gerada: {len(suite)} grafos.")
    return suite
