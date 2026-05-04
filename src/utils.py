"""
utils.py — Funções utilitárias
==============================

Funções de I/O, configuração, logging e controle de seed para
garantir reprodutibilidade dos experimentos.
"""

import os
import random
import logging
import yaml
import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    """Define seed global para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path: str) -> dict:
    """Carrega configuração YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> str:
    """Retorna caminho raiz do projeto."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs(config: dict) -> None:
    """Cria diretórios de resultados se não existirem."""
    root = get_project_root()
    paths = config.get("paths", {})
    for key, relpath in paths.items():
        dirpath = os.path.join(root, relpath)
        os.makedirs(dirpath, exist_ok=True)


def setup_logging(level=logging.INFO) -> logging.Logger:
    """Configura logging padrão do projeto."""
    logger = logging.getLogger("graph_sampling")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_csv(df: pd.DataFrame, filepath: str) -> None:
    """Salva DataFrame em CSV, criando diretórios se necessário."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def load_csv(filepath: str) -> pd.DataFrame:
    """Carrega CSV como DataFrame."""
    return pd.read_csv(filepath)


def results_path(config: dict, subdir: str, filename: str) -> str:
    """Retorna caminho completo para arquivo de resultado."""
    root = get_project_root()
    paths = config.get("paths", {})
    base = paths.get(f"results_{subdir}", f"results/{subdir}")
    return os.path.join(root, base, filename)
