#!/usr/bin/env python3
"""
run_level1_benchmark.py — Benchmark multicritério de amostragem
===============================================================

Executa o benchmark completo do Nível 1 e gera visualizações.

Uso:
    python scripts/run_level1_benchmark.py --config configs/default.yaml
    python scripts/run_level1_benchmark.py --config configs/default.yaml --viz
"""

import sys
import os
import argparse

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, ensure_dirs, set_global_seed
from src.experiments import run_level1_benchmark
from src.visualization import generate_all_level1_figures


def main():
    parser = argparse.ArgumentParser(description="Nível 1: Benchmark de Amostragem")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Caminho para arquivo de configuração YAML")
    parser.add_argument("--viz", action="store_true",
                        help="Gerar visualizações após o benchmark")
    parser.add_argument("--viz-only", action="store_true",
                        help="Apenas gerar visualizações (sem rodar benchmark)")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    if args.viz_only:
        logger.info("Modo: apenas visualizações")
        generate_all_level1_figures(config)
    else:
        logger.info("Iniciando benchmark Nível 1...")
        df_dist, df_global = run_level1_benchmark(config)
        logger.info(f"Resultados: {len(df_dist)} linhas de distâncias, {len(df_global)} linhas de erros globais")

        if args.viz:
            logger.info("Gerando visualizações...")
            generate_all_level1_figures(config)

    logger.info("Concluído!")


if __name__ == "__main__":
    main()
