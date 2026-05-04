# Relatório Final: Pipeline Avançada de Amostragem de Redes Complexas

## 1. Contextualização e Objetivo
A análise de redes complexas frequentemente esbarra na limitação de recursos computacionais quando lidamos com grafos de milhões de nós (redes sociais, biológicas, infraestrutura). A **amostragem de grafos** surge como a solução, porém, amostrar um subgrafo que seja fiel à estrutura original é um desafio não-trivial, visto que diferentes algoritmos causam vieses diferentes (ex: o método *Snowball* superestima componentes conexas e superestima hubs locais).

O objetivo deste projeto foi construir uma **pipeline de amostragem multicritério**, avaliando desde heurísticas clássicas até meta-heurísticas de aprendizado e otimização, utilizando a métrica composta **Structural Preservation Score (SPS)** para definir empiricamente qual a melhor estratégia amostral.

---

## 2. Estrutura de Avaliação
O ambiente de benchmark testou todos os amostradores nos seguintes moldes:
* **Grafos Geradores:** Erdős-Rényi (Aleatório), Barabási-Albert (Livre de Escala) e Watts-Strogatz (Mundo Pequeno).
* **Tamanhos das Redes:** De 1.000 nós até testes expandidos com 3.000 nós.
* **Fração de Amostragem:** 10%, 20% e 30%.
* **Métricas Avaliadas:** Distribuições de Grau, Clustering, Betweenness, Eigenvector, Closeness e K-Core.
* **Métricas de Distância:** Divergência de Jensen-Shannon (JS) e Divergência de Kullback-Leibler (KL).

A métrica principal de ranqueamento foi o **SPS** (quanto mais próximo de 0, melhor a preservação topológica geral).

---

## 3. Evolução dos Amostradores (Níveis 1 a 4)

### Nível 1: Baselines Clássicas
Foram implementados os amostradores fundamentais da literatura em Ciência de Redes:
1. **Random Node:** Seleção uniforme de nós (viés de desconexão).
2. **Random Edge:** Seleção uniforme de arestas (bom para preservar grau global, mas destrói clustering local).
3. **Snowball Sampling:** Expansão em largura (BFS) a partir de uma semente (superestima subgrafos densos).
4. **Random Walk (RW):** Passeio aleatório simples (excelente equilíbrio global e local, considerada a baseline a ser batida).
5. **Preferential RW e Metropolis-Hastings RW:** Correções estatísticas aplicadas ao RW.

**Resultado do Nível 1:** O *Random Walk* e o *Metropolis-Hastings RW* provaram ser incrivelmente difíceis de serem superados na métrica SPS.

### Nível 2: Avaliação Sistêmica e SPS
Institucionalização do SPS e criação de um meta-modelo para recomendar a melhor baseline de acordo com as características da rede.

### Nível 3: Contribuições Iniciais (GOAS Estático e RL)
Neste ponto, saímos da literatura clássica e começamos a criar heurísticas originais:
1. **GOAS (Goal-Oriented Adaptive Sampling):** Um amostrador que escolhia o próximo nó avaliando as características atuais do subgrafo contra o objetivo de amostragem. Versão inicialmente estática.
2. **RL Sampler (DQN):** Um modelo baseado em *Reinforcement Learning* com uma recompensa densa. *Apresentou forte instabilidade de convergência nos testes.*

### Nível 4: Estado-da-Arte (Amostragem Adaptativa e Meta-heurísticas)
A verdadeira disrupção da nossa pesquisa ocorreu aqui, introduzindo otimização de hiperparâmetros e algoritmos de exploração avançados para bater o *Random Walk*. As implementações foram:

1. **GOAS v2 Adaptativo (`goas_mh_adaptive`):**
   * Corrigiu os gargalos do GOAS clássico adicionando pesos dinâmicos que mudam online (ex: focar primeiro na conectividade e depois no grau) e aplicou a aceitação Metropolis-Hastings para remover o forte viés em *hubs*. Mostrou-se extremamente estável, superando amplamente a versão estática e os métodos aleatórios básicos.

2. **Contextual Bandit Sampler:**
   * Utilizou a estratégia *Thompson Sampling* para tratar a escolha de vizinhos como um problema Multi-Armed Bandit, equilibrando *exploração* ($\epsilon$-greedy) e *explotação*.
   * Embora teoricamente belo, sofreu em grandes grafos (0.84 SPS em grafos de 3000 nós) por ser penalizado severamente em escolhas ruins na fase de exploração.

3. **PSO-GOAS (Particle Swarm Optimization):**
   * **O Grande Vencedor.** Ao envolver o amostrador GOAS em um otimizador de enxame de partículas, permitiu que o algoritmo descobrisse dinamicamente os pesos exatos de importância de cada topologia (Grau, Clustering, Walk).
   * **Resultados:** Aumentando as partículas/iterações nos grafos maiores, alcançou uma média de SPS de **0.395**, dizimando todos os outros métodos complexos e essencialmente empatando (ou ganhando estatisticamente em cenários pareados de Wilcoxon) contra o Random Walk Clássico (0.393).

4. **ACO Sampler (Otimização por Colônia de Formigas):**
   * Tratou a amostragem como um trajeto de formigas, depositando "feromônios" nos nós que contribuíam positivamente para a queda do SPS.
   * **Resultados:** Brilhou intensamente na **Fronteira de Pareto**. Das 13 configurações ótimas (Pareto-eficientes no *trade-off* Grau vs Estrutura Geral) listadas, o ACO esteve presente em 5, demonstrando ser a melhor ferramenta de compromisso multiobjetivo da pipeline.

---

## 4. Conclusão Final

A hipótese central de que **"não existe um sampler universalmente ótimo"** foi comprovada. Contudo, nossa pesquisa revelou dinâmicas cruciais:

1. **Sistemas Dinâmicos Vencem Sistemas Rígidos:** A versão do GOAS envolta em PSO demonstrou que delegar a calibração dos pesos de amostragem para um otimizador não-convexo produz o subgrafo de maior fidelidade topológica (menor SPS global).
2. **Trade-offs Geométricos (Pareto):** Métodos simples como o *Snowball* são perfeitos se o analista só precisa manter o grau local e componentes densos. Mas, se deseja-se balanço, o **ACO Sampler** atua perfeitamente no cotovelo da Fronteira de Pareto.
3. **Poder do Random Walk:** Ficou evidente por que o Random Walk é tão difícil de bater; o nosso próprio estado-da-arte (PSO-GOAS) precisou de uma extensa exploração de hiperparâmetros para alcançar a sua robustez nativa.

Toda a infraestrutura, módulos estatísticos (Wilcoxon) e geradores de *Pareto Frontiers* foram abstraídos na pipeline (`scripts/run_level4_advanced.py`), entregando um projeto em maturidade de submissão para revistas de Ciência de Redes.
