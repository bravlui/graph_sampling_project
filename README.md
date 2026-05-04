# Projeto 6 — Amostragem de Redes Complexas

Projeto de pesquisa para comparar, modelar e estender métodos de amostragem de redes complexas.

## Estrutura

```
graph_sampling_project/
├── configs/default.yaml          # Parâmetros configuráveis
├── src/                          # Código-fonte modular
│   ├── graph_generators.py       # Geração de redes sintéticas
│   ├── samplers.py               # Métodos de amostragem (clássicos + GOAS)
│   ├── metrics.py                # Métricas topológicas
│   ├── distances.py              # Distâncias entre distribuições
│   ├── experiments.py            # Lógica do benchmark
│   ├── visualization.py          # Visualizações para relatório
│   ├── metamodel.py              # Meta-modelo preditivo (Nível 2)
│   ├── pareto.py                 # Fronteira de Pareto (Nível 3)
│   ├── rl_sampler.py             # RL sampler com DQN (Nível 3)
│   └── utils.py                  # Utilitários
├── scripts/                      # Scripts executáveis
│   ├── run_level1_benchmark.py
│   ├── run_level2_metamodel.py
│   ├── run_level3_goas.py
│   ├── run_level3_pareto.py
│   └── run_level3_rl_sampler.py
├── results/                      # Resultados gerados
│   ├── raw/                      # CSVs brutos
│   ├── processed/                # CSVs processados
│   ├── figures/                  # Figuras PNG
│   └── models/                   # Modelos salvos
└── notebooks/
    └── Projeto_6_Amostragem_Redes.ipynb
```

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

Cada nível é executável de forma independente. Os níveis 2+ dependem dos resultados do nível anterior.

### Nível 1 — Benchmark multicritério

```bash
python scripts/run_level1_benchmark.py --config configs/default.yaml
```

Compara 6 métodos de amostragem (random node, random edge, snowball, random walk, preferential RW, Metropolis-Hastings RW) em redes ER, BA, WS com diferentes tamanhos e frações amostrais.

**Saídas:**
- `results/raw/level1_distribution_distances.csv`
- `results/raw/level1_global_errors.csv`
- `results/processed/level1_summary_by_sampler.csv`
- `results/processed/level1_best_sampler_by_metric.csv`
- `results/figures/` — boxplots, heatmaps, PCA

### Nível 2 — Meta-modelo preditivo

```bash
python scripts/run_level2_metamodel.py --config configs/default.yaml
```

Treina modelos (RF, GBM, LogReg, MLP) para recomendar o melhor sampler dados as características da rede e o objetivo de preservação.

**Saídas:**
- `results/processed/level2_*.csv`
- `results/figures/level2_*.png`
- `results/models/level2_*.joblib`

### Nível 3 — Contribuições originais

#### C1: Goal-Oriented Adaptive Sampling (GOAS)
```bash
python scripts/run_level3_goas.py --config configs/default.yaml
```

#### C2: Structural Preservation Pareto Frontier
```bash
python scripts/run_level3_pareto.py --config configs/default.yaml
```

#### C3: Self-Supervised Graph Sampling via RL
```bash
python scripts/run_level3_rl_sampler.py --config configs/default.yaml
```

## Reprodutibilidade

Todos os experimentos usam seeds fixas configuráveis via `configs/default.yaml`.
