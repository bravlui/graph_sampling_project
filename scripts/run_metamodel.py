#!/usr/bin/env python3
"""
run_metamodel.py — Metamodelo preditivo de sampler ideal
=========================================================

Constrói um dataset de (features da rede, melhor sampler) a partir dos
resultados do benchmark e treina um Random Forest para recomendar o método
de amostragem mais adequado a uma rede desconhecida, dado seu perfil estrutural
(tamanho, densidade, modelo estimado, etc.).

Esse tipo de abordagem — treinar um modelo sobre o comportamento de outros
algoritmos — é chamado de "algorithm selection" ou "meta-learning" na literatura
de automated machine learning (AutoML).

Uso
---
    python scripts/run_metamodel.py
    python scripts/run_metamodel.py --config configs/default.yaml --analyze-only
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, ensure_dirs, set_global_seed
from src.metamodel import build_metamodel_dataset, train_metamodel, analyze_metamodel_results


def main():
    parser = argparse.ArgumentParser(
        description="Metamodelo de seleção de sampler via Random Forest"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Caminho para arquivo de configuração YAML")
    parser.add_argument("--build-only", action="store_true",
                        help="Apenas construir dataset (sem treinar modelo)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Apenas gerar análise interpretativa (requer modelo treinado)")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    if args.analyze_only:
        logger.info("Gerando análise interpretativa do metamodelo...")
        analyze_metamodel_results(config)
    elif args.build_only:
        logger.info("Construindo dataset para o metamodelo...")
        build_metamodel_dataset(config)
    else:
        logger.info("Pipeline completo do metamodelo...")
        build_metamodel_dataset(config)
        train_metamodel(config)
        analyze_metamodel_results(config)

    logger.info("Concluído.")


if __name__ == "__main__":
    main()
