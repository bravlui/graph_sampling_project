"""
metamodel.py — Meta-modelo preditivo (Nível 2)
===============================================

Constrói dataset supervisionado a partir dos resultados do Nível 1 e
treina modelos para recomendar o melhor sampler.
"""

import os
import numpy as np
import pandas as pd
import logging
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, mean_squared_error, mean_absolute_error,
)

from src.utils import results_path, save_csv, load_csv
from src.metrics import compute_graph_feature_vector

logger = logging.getLogger("graph_sampling")


# ============================================================================
# Construção do dataset
# ============================================================================

def build_metamodel_dataset(config):
    """
    Constrói dataset supervisionado para o meta-modelo.

    Cada exemplo = (graph_id, sample_frac, objective) → best_sampler.

    Features:
    - Métricas globais da rede original
    - sample_frac
    - graph_model (one-hot)

    Returns
    -------
    pd.DataFrame
    """
    # Carregar resultados do Nível 1
    dist_path = results_path(config, "raw", "level1_distribution_distances.csv")
    global_path = results_path(config, "raw", "level1_global_errors.csv")

    df_dist = pd.read_csv(dist_path)
    df_global = pd.read_csv(global_path)

    # Definir objetivos e suas métricas
    objectives = {
        "preserve_degree": ["degree_JS"],
        "preserve_clustering": ["clustering_JS"],
        "preserve_betweenness": ["betweenness_JS"],
        "preserve_shortest_path": ["re_avg_shortest_path_lcc"],
        "preserve_global_sps": ["SPS"],
    }

    # Para cada objetivo, encontrar o melhor sampler por (graph_id, sample_frac)
    rows = []
    group_cols = ["graph_id", "model", "sample_frac", "graph_n"]

    for obj_name, metric_cols in objectives.items():
        # Usar df_dist para JS e df_global para relative errors
        if metric_cols[0].startswith("re_"):
            df_work = df_global.copy()
        else:
            df_work = df_dist.copy()

        available_cols = [c for c in metric_cols if c in df_work.columns]
        if not available_cols:
            continue

        # Calcular erro médio por (graph_id, sample_frac, sampler)
        agg_cols = group_cols + ["sampler"]
        agg = df_work.groupby(agg_cols)[available_cols].mean().reset_index()
        agg["error"] = agg[available_cols].mean(axis=1)

        # Para cada (graph_id, sample_frac), encontrar melhor sampler
        for _, group in agg.groupby(group_cols):
            best_idx = group["error"].idxmin()
            best_row = group.loc[best_idx]

            row = {
                "graph_id": best_row["graph_id"],
                "model": best_row["model"],
                "graph_n": best_row["graph_n"],
                "sample_frac": best_row["sample_frac"],
                "objective": obj_name,
                "target_sampler": best_row["sampler"],
                "best_error": best_row["error"],
            }
            rows.append(row)

    df_meta = pd.DataFrame(rows)

    # Adicionar features da rede original (métricas globais)
    # Extrair features únicas por graph_id do df_global
    feature_cols_global = [c for c in df_global.columns if c.startswith("re_")]
    # Usaremos as métricas globais raw se disponíveis, senão estimamos
    # Por simplicidade, usamos graph_n e model como features base
    # e adicionamos métricas derivadas do benchmark

    # One-hot para model
    for model_name in df_meta["model"].unique():
        df_meta[f"model_{model_name}"] = (df_meta["model"] == model_name).astype(int)

    # Salvar
    save_csv(df_meta, results_path(config, "processed", "level2_metamodel_dataset.csv"))
    logger.info(f"Dataset do meta-modelo: {len(df_meta)} exemplos")
    return df_meta


# ============================================================================
# Treinamento
# ============================================================================

def train_metamodel(config):
    """
    Treina modelos de classificação e regressão.

    Classification: prever o melhor sampler.
    Regression: prever o SPS de cada sampler.
    """
    dataset_path = results_path(config, "processed", "level2_metamodel_dataset.csv")
    if not os.path.exists(dataset_path):
        logger.info("Construindo dataset do meta-modelo...")
        df = build_metamodel_dataset(config)
    else:
        df = pd.read_csv(dataset_path)

    if df.empty:
        logger.error("Dataset vazio, abortando treinamento.")
        return

    # Features
    feature_cols = ["graph_n", "sample_frac"]
    model_cols = [c for c in df.columns if c.startswith("model_")]
    feature_cols.extend(model_cols)

    X = df[feature_cols].values
    y = df["target_sampler"].values
    groups = df["graph_id"].values

    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # GroupKFold
    n_folds = min(config.get("metamodel", {}).get("cv_folds", 5), len(np.unique(groups)))
    if n_folds < 2:
        n_folds = 2

    gkf = GroupKFold(n_splits=n_folds)

    # Modelos
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }

    results = []
    best_score = -1
    best_model = None
    best_model_name = None

    for name, clf in classifiers.items():
        accs, f1s, baccs = [], [], []

        for train_idx, test_idx in gkf.split(X, y_encoded, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ])

            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                accs.append(accuracy_score(y_test, y_pred))
                f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
                baccs.append(balanced_accuracy_score(y_test, y_pred))
            except Exception as e:
                logger.warning(f"{name} fold falhou: {e}")

        if accs:
            mean_acc = np.mean(accs)
            result = {
                "model": name,
                "accuracy_mean": mean_acc,
                "accuracy_std": np.std(accs),
                "macro_f1_mean": np.mean(f1s),
                "macro_f1_std": np.std(f1s),
                "balanced_accuracy_mean": np.mean(baccs),
                "balanced_accuracy_std": np.std(baccs),
            }
            results.append(result)

            if mean_acc > best_score:
                best_score = mean_acc
                best_model_name = name
                # Retrain on full data
                best_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ])
                best_model.fit(X, y_encoded)

    # Salvar resultados
    df_results = pd.DataFrame(results)
    save_csv(df_results, results_path(config, "processed", "level2_classification_metrics.csv"))

    # Salvar melhor modelo
    if best_model is not None:
        model_path = results_path(config, "models", "level2_best_classifier.joblib")
        joblib.dump({"model": best_model, "label_encoder": le, "feature_cols": feature_cols},
                    model_path)
        logger.info(f"Melhor classificador: {best_model_name} (acc={best_score:.3f})")

        # Confusion matrix
        y_pred_full = best_model.predict(X)
        cm = confusion_matrix(y_encoded, y_pred_full)
        _plot_confusion_matrix(cm, le.classes_, config)

        # Feature importance
        if hasattr(best_model.named_steps["clf"], "feature_importances_"):
            _plot_feature_importance(
                best_model.named_steps["clf"].feature_importances_,
                feature_cols, config,
            )

    # ---- Regressão (prever SPS) ----
    _train_regression(config, df, feature_cols, groups)

    return df_results


def _train_regression(config, df, feature_cols, groups):
    """Treina modelos de regressão para prever SPS."""
    if "best_error" not in df.columns:
        return

    X = df[feature_cols].values
    y = df["best_error"].values
    groups = df["graph_id"].values

    n_folds = min(config.get("metamodel", {}).get("cv_folds", 5), len(np.unique(groups)))
    if n_folds < 2:
        n_folds = 2
    gkf = GroupKFold(n_splits=n_folds)

    regressors = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = []
    best_score = float("inf")
    best_model = None

    for name, reg in regressors.items():
        mses, maes = [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                mses.append(mean_squared_error(y_test, y_pred))
                maes.append(mean_absolute_error(y_test, y_pred))
            except Exception as e:
                logger.warning(f"Reg {name} fold falhou: {e}")

        if mses:
            mean_mse = np.mean(mses)
            results.append({
                "model": name,
                "mse_mean": mean_mse,
                "mse_std": np.std(mses),
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
            })
            if mean_mse < best_score:
                best_score = mean_mse
                best_model = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
                best_model.fit(X, y)

    if results:
        save_csv(pd.DataFrame(results), results_path(config, "processed", "level2_regression_metrics.csv"))

    if best_model:
        joblib.dump(best_model, results_path(config, "models", "level2_best_regressor.joblib"))


def _plot_confusion_matrix(cm, classes, config):
    """Plota e salva confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Confusion Matrix — Meta-modelo", fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    filepath = results_path(config, "figures", "level2_confusion_matrix.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(importances, feature_names, config):
    """Plota e salva feature importance."""
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(importances)), importances[idx], color="#2A9D8F")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_ylabel("Importância")
    ax.set_title("Feature Importance — Meta-modelo", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    filepath = results_path(config, "figures", "level2_feature_importance.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Recomendação
# ============================================================================

def recommend_sampler(G, sample_frac, objective, model_path, config=None):
    """
    Recomenda o melhor sampler para uma rede.

    Parameters
    ----------
    G : nx.Graph
    sample_frac : float
    objective : str
    model_path : str
    config : dict or None

    Returns
    -------
    str
        Nome do sampler recomendado.
    """
    import networkx as nx

    data = joblib.load(model_path)
    model = data["model"]
    le = data["label_encoder"]
    feature_cols = data["feature_cols"]

    # Construir features
    features = {"graph_n": G.number_of_nodes(), "sample_frac": sample_frac}
    # One-hot para model (não sabemos o modelo, então zeros)
    for col in feature_cols:
        if col.startswith("model_") and col not in features:
            features[col] = 0

    X = np.array([[features.get(c, 0) for c in feature_cols]])
    y_pred = model.predict(X)
    return le.inverse_transform(y_pred)[0]


# ============================================================================
# Análise interpretativa
# ============================================================================

def analyze_metamodel_results(config):
    """
    Gera tabelas e gráficos interpretáveis para o relatório.

    Produz:
    - Matriz de decisão (modelo x objetivo → sampler recomendado)
    - Mapa de recomendação
    """
    dataset_path = results_path(config, "processed", "level2_metamodel_dataset.csv")
    if not os.path.exists(dataset_path):
        logger.error("Dataset não encontrado.")
        return

    df = pd.read_csv(dataset_path)

    # Matriz de decisão
    decision_rows = []
    objectives = df["objective"].unique()
    models = df["model"].unique()

    for model in models:
        row = {"model": model}
        for obj in objectives:
            sub = df[(df["model"] == model) & (df["objective"] == obj)]
            if not sub.empty:
                best = sub.groupby("target_sampler").size().idxmax()
                row[obj] = best
            else:
                row[obj] = "N/A"
        decision_rows.append(row)

    df_decision = pd.DataFrame(decision_rows)
    save_csv(df_decision, results_path(config, "processed", "level2_decision_matrix.csv"))

    # Plot do mapa de recomendação
    fig, ax = plt.subplots(figsize=(12, 4))

    # Codificar samplers como números para o heatmap
    all_samplers = sorted(df["target_sampler"].unique())
    sampler_to_num = {s: i for i, s in enumerate(all_samplers)}

    matrix = np.zeros((len(models), len(objectives)))
    for i, model in enumerate(models):
        for j, obj in enumerate(objectives):
            val = df_decision.loc[df_decision["model"] == model, obj]
            if not val.empty and val.values[0] != "N/A":
                matrix[i, j] = sampler_to_num.get(val.values[0], 0)

    im = ax.imshow(matrix, cmap="tab10", aspect="auto",
                   vmin=0, vmax=max(len(all_samplers) - 1, 1))
    ax.set_xticks(range(len(objectives)))
    ax.set_xticklabels(objectives, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # Anotações
    for i, model in enumerate(models):
        for j, obj in enumerate(objectives):
            val = df_decision.loc[df_decision["model"] == model, obj]
            if not val.empty:
                ax.text(j, i, str(val.values[0])[:12], ha="center", va="center",
                       fontsize=7, fontweight="bold")

    ax.set_title("Mapa de Recomendação de Sampler", fontweight="bold", fontsize=13)
    fig.tight_layout()

    filepath = results_path(config, "figures", "level2_sampler_recommendation_map.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Análise do meta-modelo concluída.")
