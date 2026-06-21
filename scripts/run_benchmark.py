#!/usr/bin/env python3
"""
run_benchmark.py — Benchmark multicritério de amostragem de redes
==================================================================

Ponto de entrada principal do estudo. Para cada combinação de modelo de rede
(ER, BA, WS), tamanho (n=1000 e n=5000), sampler (6 métodos) e fração amostral
(configurável), executa `repetitions` réplicas independentes e registra:

  - Divergências entre distribuições nodais (KL, JS, Wasserstein, KS)
  - Erros relativos em métricas globais (caminho médio, clustering, grau médio)
  - Structural Preservation Score (SPS): agregação dessas métricas em um único número

Os CSVs de resultado são salvos em results/raw/ (dados por experimento) e
results/processed/ (resumos por sampler e melhor sampler por métrica).

Uso
---
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --config configs/default.yaml --viz

Argumentos
----------
--config    Caminho para o YAML de configuração (padrão: configs/default.yaml)
--viz       Gerar figuras PNG após o benchmark
--viz-only  Apenas gerar figuras a partir de CSVs já existentes (sem rodar o benchmark)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, ensure_dirs, set_global_seed
from src.experiments import run_sampling_benchmark
from src.visualization import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multicritério de métodos de amostragem de redes"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Caminho para arquivo de configuração YAML"
    )
    parser.add_argument(
        "--viz", action="store_true",
        help="Gerar figuras PNG após o benchmark"
    )
    parser.add_argument(
        "--viz-only", action="store_true",
        help="Apenas gerar figuras (requer CSVs em results/raw/)"
    )
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    if args.viz_only:
        logger.info("Gerando figuras a partir dos CSVs existentes...")
        generate_all_figures(config)
    else:
        df_dist, df_global = run_sampling_benchmark(config)
        logger.info(
            f"Benchmark concluído: {len(df_dist)} experimentos de distribuição, "
            f"{len(df_global)} experimentos de métricas globais."
        )
        if args.viz:
            logger.info("Gerando figuras...")
            generate_all_figures(config)

    logger.info("Concluído.")


if __name__ == "__main__":
    main()
