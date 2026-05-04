#!/usr/bin/env python3
"""
run_level2_metamodel.py — Meta-modelo preditivo
================================================

Constrói dataset, treina modelos e gera análise interpretativa.

Uso:
    python scripts/run_level2_metamodel.py --config configs/default.yaml
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, ensure_dirs, set_global_seed
from src.metamodel import build_metamodel_dataset, train_metamodel, analyze_metamodel_results


def main():
    parser = argparse.ArgumentParser(description="Nível 2: Meta-modelo Preditivo")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--build-only", action="store_true",
                        help="Apenas construir dataset (sem treinar)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Apenas gerar análise interpretativa")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_global_seed(config.get("global_seed", 42))

    if args.analyze_only:
        logger.info("Gerando análise interpretativa...")
        analyze_metamodel_results(config)
    elif args.build_only:
        logger.info("Construindo dataset do meta-modelo...")
        build_metamodel_dataset(config)
    else:
        logger.info("Pipeline completo do Nível 2...")
        build_metamodel_dataset(config)
        train_metamodel(config)
        analyze_metamodel_results(config)

    logger.info("Nível 2 concluído!")


if __name__ == "__main__":
    main()
