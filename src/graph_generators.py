"""
graph_generators.py — Geradores de redes sintéticas
====================================================

Os três modelos escolhidos representam regimes topológicos distintos, o que permite
avaliar se o comportamento dos samplers depende do tipo de rede:

- Erdős-Rényi G(n,p): conexões completamente aleatórias, distribuição de grau Poisson
  — graus homogêneos, sem hubs, alto caminho médio.

- Barabási-Albert: crescimento com apego preferencial (rich-get-richer), distribuição
  de grau em lei de potência P(k) ∝ k^{-3} — hubs dominantes (redes livre de escala).
  Introduzido por Barabási & Albert (1999) como modelo para a Web e redes sociais.

- Watts-Strogatz: "mundo pequeno" com alto coeficiente de clustering e caminho médio
  curto, semelhante a redes de relacionamento social (Watts & Strogatz, 1998).

A combinação ER + BA + WS é o benchmark padrão em estudos de amostragem de redes
(e.g., Stumpf et al., 2005; Leskovec & Faloutsos, 2006).

Todos os grafos são retornados como o maior componente conectado,
relabelado de 0 a n-1 para consistência com as métricas.
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
    Gera rede Erdős-Rényi G(n, p), com p = avg_degree / (n-1).

    Cada par de nós é conectado independentemente com probabilidade p. A distribuição
    de grau resultante é Poisson com média avg_degree para n grande. Acima do limiar
    de percolação p_c = 1/n (avg_degree > 1), a rede possui uma componente gigante;
    usamos apenas essa componente para garantir conectividade.

    Parameters
    ----------
    n : int
        Número de nós.
    avg_degree : float
        Grau médio desejado.
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
    Gera rede Barabási-Albert com crescimento e apego preferencial.

    Cada novo nó conecta-se a m nós existentes com probabilidade proporcional ao
    grau deles (rich-get-richer). O resultado é uma distribuição de grau em lei de
    potência P(k) ∝ k^{-3}, com poucos hubs de grau muito alto e muitos nós de
    grau baixo. Redes BA são sempre conectadas por construção, mas retiramos o LCC
    para consistência com ER e WS.

    Parameters
    ----------
    n : int
        Número de nós.
    m : int
        Número de arestas adicionadas por novo nó (grau mínimo = m).
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
    Gera rede Watts-Strogatz (modelo de mundo pequeno).

    Parte de um anel regular onde cada nó conecta-se a k vizinhos. Cada aresta
    é reconectada aleatoriamente com probabilidade p. Para p intermediário (~0.1–0.3),
    a rede exibe a combinação de mundo pequeno: caminho médio curto (como ER) e alto
    coeficiente de clustering (como o anel regular). A distribuição de grau é estreita
    em torno de k — muito diferente da lei de potência do BA.

    Parameters
    ----------
    n : int
        Número de nós.
    k : int
        Número de vizinhos iniciais no anel; deve ser par e maior que ln(n).
    p : float
        Probabilidade de reconexão (0 = anel regular; 1 = ER aleatório).
    seed : int
        Semente para reprodutibilidade.

    Returns
    -------
    nx.Graph
        Maior componente conectado (geralmente toda a rede para p > 0).
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
