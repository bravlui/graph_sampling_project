"""
pareto.py — Análise multiobjetivo e fronteira de Pareto
========================================================

Trata a amostragem de redes como um problema de otimização multiobjetivo:
ao invés de um único score agregado (SPS), avalia simultaneamente múltiplos
critérios de preservação estrutural — grau, clustering, betweenness, etc.

A fronteira de Pareto identifica configurações (sampler, fração) para as quais
não existe alternativa que melhore em todos os objetivos ao mesmo tempo. Pontos
fora da fronteira são "dominados" — há opção estritamente melhor disponível.

Isso é relevante porque trade-offs existem: um método que preserva bem a
distribuição de grau pode distorcer o clustering local, dependendo do tipo de rede.
Expor esses trade-offs explicitamente é preferível a escondê-los em uma média.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("graph_sampling")


def is_pareto_efficient(points):
    """
    Identifica pontos Pareto-eficientes em problema de minimização (todos os objetivos).

    Um ponto i é dominado por j se j <= i em todos os objetivos E j < i em pelo menos um.
    O algoritmo O(n²) é suficiente para os tamanhos de dataset deste estudo (n < 5000).
    Para datasets maiores, existem implementações O(n log n) via ordenação lexicográfica.

    Parameters
    ----------
    points : np.ndarray, shape (n, m)
        Matriz onde cada linha é um ponto e cada coluna um objetivo a minimizar.

    Returns
    -------
    np.ndarray of bool, shape (n,)
        True para pontos não dominados (pertencentes à fronteira de Pareto).
    """
    n = points.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue
        # Um ponto i é dominado se existe j tal que j <= i em todos os objetivos
        # e j < i em pelo menos um
        for j in range(n):
            if i == j or not is_efficient[j]:
                continue
            # j domina i?
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_efficient[i] = False
                break

    return is_efficient


def compute_pareto_frontier(df, loss_cols, group_cols=None):
    """
    Calcula fronteira de Pareto para cada grupo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas de perda e metadados.
    loss_cols : list[str]
        Colunas de perda (a minimizar).
    group_cols : list[str] or None
        Colunas de agrupamento. Se None, calcula para tudo junto.

    Returns
    -------
    pd.DataFrame
        DataFrame original com coluna 'is_pareto' adicionada.
    """
    df = df.copy()
    df["is_pareto"] = False

    available_cols = [c for c in loss_cols if c in df.columns]
    if not available_cols:
        logger.warning("Nenhuma coluna de perda encontrada.")
        return df

    if group_cols is None:
        points = df[available_cols].fillna(1e6).values
        mask = is_pareto_efficient(points)
        df["is_pareto"] = mask
    else:
        valid_groups = [c for c in group_cols if c in df.columns]
        if not valid_groups:
            points = df[available_cols].fillna(1e6).values
            mask = is_pareto_efficient(points)
            df["is_pareto"] = mask
        else:
            for _, group in df.groupby(valid_groups):
                idx = group.index
                points = group[available_cols].fillna(1e6).values
                if len(points) < 2:
                    df.loc[idx, "is_pareto"] = True
                    continue
                mask = is_pareto_efficient(points)
                df.loc[idx, "is_pareto"] = mask

    return df


def pareto_dominance_summary(df):
    """
    Resumo de dominância Pareto.

    Returns
    -------
    pd.DataFrame
        Contagem de vezes que cada sampler aparece na fronteira.
    """
    if "is_pareto" not in df.columns:
        return pd.DataFrame()

    pareto_df = df[df["is_pareto"]]

    summary = pareto_df.groupby("sampler").agg(
        pareto_count=("is_pareto", "sum"),
    ).reset_index()

    total = len(df.groupby("sampler").size())
    summary["pareto_frequency"] = summary["pareto_count"] / max(len(pareto_df), 1)

    if "sample_frac" in pareto_df.columns:
        min_frac = pareto_df.groupby("sampler")["sample_frac"].min().reset_index()
        min_frac.columns = ["sampler", "min_pareto_frac"]
        summary = summary.merge(min_frac, on="sampler", how="left")

    return summary.sort_values("pareto_count", ascending=False)


def compute_hypervolume_proxy(df, loss_cols):
    """
    Proxy de hypervolume: média normalizada dos objetivos.

    Parameters
    ----------
    df : pd.DataFrame
    loss_cols : list[str]

    Returns
    -------
    pd.DataFrame
        DataFrame com coluna 'hypervolume_proxy' adicionada.
    """
    df = df.copy()
    available = [c for c in loss_cols if c in df.columns]
    if not available:
        df["hypervolume_proxy"] = float("nan")
        return df

    # Normalizar entre 0 e 1
    normalized = df[available].copy()
    for col in available:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.0

    df["hypervolume_proxy"] = normalized.mean(axis=1)
    return df


def analyze_tradeoffs(df, loss_cols):
    """
    Analisa trade-offs entre objetivos.

    Returns
    -------
    list[dict]
        Lista de trade-offs identificados.
    """
    tradeoffs = []
    available = [c for c in loss_cols if c in df.columns]

    if len(available) < 2:
        return tradeoffs

    pareto_df = df[df.get("is_pareto", False)] if "is_pareto" in df.columns else df

    # Correlações entre objetivos na fronteira
    for i, col1 in enumerate(available):
        for col2 in available[i+1:]:
            corr = pareto_df[[col1, col2]].corr().iloc[0, 1]
            if not np.isnan(corr) and corr < -0.3:
                tradeoffs.append({
                    "objective_1": col1,
                    "objective_2": col2,
                    "correlation": corr,
                    "interpretation": f"Trade-off negativo: melhorar {col1} tende a piorar {col2}",
                })

    return tradeoffs


def analyze_sample_efficiency(df, loss_cols):
    """
    Identifica quando amostras menores dominam maiores.

    Returns
    -------
    pd.DataFrame
    """
    if "sample_frac" not in df.columns or "is_pareto" not in df.columns:
        return pd.DataFrame()

    available = [c for c in loss_cols if c in df.columns]
    if not available:
        return pd.DataFrame()

    results = []
    pareto = df[df["is_pareto"]]

    samplers = df["sampler"].unique()
    fracs = sorted(df["sample_frac"].unique())

    for sampler in samplers:
        for i, small_frac in enumerate(fracs):
            for large_frac in fracs[i+1:]:
                small = df[(df["sampler"] == sampler) & (df["sample_frac"] == small_frac)]
                large = df[(df["sampler"] == sampler) & (df["sample_frac"] == large_frac)]

                if small.empty or large.empty:
                    continue

                small_mean = small[available].mean()
                large_mean = large[available].mean()

                # Amostra menor domina maior?
                if np.all(small_mean.values <= large_mean.values):
                    results.append({
                        "sampler": sampler,
                        "small_frac": small_frac,
                        "large_frac": large_frac,
                        "small_mean_loss": small_mean.mean(),
                        "large_mean_loss": large_mean.mean(),
                        "dominates": True,
                    })

    return pd.DataFrame(results)
