"""
bandit_sampler.py — Amostragem via Contextual Bandit (Thompson Sampling)
=========================================================================

Formula a seleção de sampler como um problema de bandit multi-braço contextual:
a cada passo, o algoritmo escolhe um "braço" (random_walk, snowball, etc.) para
adicionar o próximo nó à amostra, usando o contexto da amostra atual como feature.

Thompson Sampling é uma política de exploração-exploração que amostra os pesos de
um modelo Bayesiano linear por braço e escolhe o braço com maior recompensa esperada.
A recompensa é a redução de divergência JS na distribuição de grau parcial.

Vantagem sobre métodos fixos: adapta-se durante a amostragem de um único grafo,
sem precisar de treinamento prévio separado. Desvantagem: overhead computacional
por passo e sensibilidade à qualidade das features de contexto.
"""

import numpy as np
import networkx as nx
import logging
from scipy.stats import entropy
import warnings

logger = logging.getLogger("graph_sampling")

class BayesianLinearRegression:
    """Regressão Linear Bayesiana simples para Thompson Sampling."""
    def __init__(self, n_features, alpha=1.0, lambda_=1.0):
        self.n_features = n_features
        self.alpha = alpha  # Prior variance
        self.lambda_ = lambda_  # Noise variance
        
        self.mean = np.zeros(n_features)
        self.cov_inv = np.eye(n_features) / self.alpha
    
    def update(self, x, y):
        """Atualiza a distribuição posterior com um novo exemplo (x, y)."""
        x = np.array(x).reshape(-1, 1)
        cov_inv_new = self.cov_inv + (x @ x.T) / self.lambda_
        mean_new = np.linalg.inv(cov_inv_new) @ (self.cov_inv @ self.mean.reshape(-1, 1) + (x * y) / self.lambda_)
        
        self.cov_inv = cov_inv_new
        self.mean = mean_new.flatten()
        
    def sample_weights(self, rng):
        """Amostra pesos da distribuição posterior."""
        cov = np.linalg.inv(self.cov_inv)
        # Add small jitter for numerical stability
        cov += np.eye(self.n_features) * 1e-6
        return rng.multivariate_normal(self.mean, cov)


class ContextualBanditSampler:
    def __init__(self, G, sample_frac, seed=42):
        self.G = G
        self.n = G.number_of_nodes()
        self.target_n = max(1, int(round(self.n * sample_frac)))
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.degree_dict = dict(G.degree())
        self.max_degree = max(self.degree_dict.values()) if self.degree_dict else 1
        self.clustering_dict = nx.clustering(G)
        
        self.nodes = list(G.nodes())
        
        self.sampled = set()
        self.frontier = set()
        
        self.n_arms = 7
        self.n_features = 9
        
        # Um modelo linear bayesiano para cada braço
        self.arms = [BayesianLinearRegression(self.n_features) for _ in range(self.n_arms)]
        
        # Original fast proxy metrics
        self.orig_deg_hist, _ = np.histogram(list(self.degree_dict.values()), bins=20, density=True)
        self.orig_clust_hist, _ = np.histogram(list(self.clustering_dict.values()), bins=20, density=True)

    def _fast_js_divergence(self, p, q):
        p = np.asarray(p) + 1e-10
        q = np.asarray(q) + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

    def compute_fast_sps(self):
        """Proxy muito mais rápido que o SPS completo para usar como reward em tempo real."""
        if len(self.sampled) < 2: return 0.0
        
        sub = self.G.subgraph(list(self.sampled))
        degrees = [self.degree_dict[v] for v in self.sampled]
        samp_deg_hist, _ = np.histogram(degrees, bins=20, density=True)
        
        clusts = [self.clustering_dict[v] for v in self.sampled]
        samp_clust_hist, _ = np.histogram(clusts, bins=20, density=True)
        
        deg_js = self._fast_js_divergence(self.orig_deg_hist, samp_deg_hist)
        clust_js = self._fast_js_divergence(self.orig_clust_hist, samp_clust_hist)
        
        # Proxy SPS invertido para ser "quanto maior melhor" na formula original
        # Original SPS = avg JS, so lower is better. Here we want to minimize JS.
        return (deg_js + clust_js) / 2.0

    def get_context(self):
        """Extrai 9 features do estado atual."""
        if not self.sampled:
            return np.zeros(self.n_features)
            
        sub = self.G.subgraph(list(self.sampled))
        degrees = [self.degree_dict[v] for v in self.sampled]
        
        sample_size_ratio = len(self.sampled) / self.n
        avg_deg = np.mean(degrees) / self.max_degree
        
        # Entropia do grau
        deg_counts = np.unique(degrees, return_counts=True)[1]
        deg_probs = deg_counts / deg_counts.sum()
        deg_entropy = entropy(deg_probs)
        
        avg_clust = np.mean([self.clustering_dict[v] for v in self.sampled])
        
        try:
            n_comp = nx.number_connected_components(sub)
        except:
            n_comp = 1
            
        frontier_size = len(self.frontier) / self.n
        
        # Estimativas atuais
        samp_deg_hist, _ = np.histogram(degrees, bins=20, density=True)
        samp_clust_hist, _ = np.histogram([self.clustering_dict[v] for v in self.sampled], bins=20, density=True)
        
        est_deg_js = self._fast_js_divergence(self.orig_deg_hist, samp_deg_hist)
        est_clust_js = self._fast_js_divergence(self.orig_clust_hist, samp_clust_hist)
        
        # Feature dummy para modelo do grafo (não sabemos o modelo aqui de forma simples, usamos 1.0 como constante ou bias)
        graph_model_bias = 1.0
        
        return np.array([
            sample_size_ratio, avg_deg, deg_entropy, avg_clust, 
            n_comp / max(len(self.sampled), 1), frontier_size,
            est_deg_js, est_clust_js, graph_model_bias
        ])

    def take_action(self, action_idx, current_node):
        """Aplica o braço e retorna o novo nó selecionado."""
        frontier_list = list(self.frontier)
        chosen = None
        
        if len(frontier_list) == 0 and action_idx != 6:
            action_idx = 6 # Force random jump
            
        if action_idx == 0: # Random node expansion (frontier)
            chosen = frontier_list[self.rng.randint(len(frontier_list))]
        elif action_idx == 1: # Highest degree
            chosen = max(frontier_list, key=lambda v: self.degree_dict.get(v, 0))
        elif action_idx == 2: # Highest clustering
            chosen = max(frontier_list, key=lambda v: self.clustering_dict.get(v, 0.0))
        elif action_idx == 3: # Random walk step
            neighbors = list(self.G.neighbors(current_node))
            if neighbors:
                chosen = neighbors[self.rng.randint(len(neighbors))]
            else:
                action_idx = 6 # Force jump
        elif action_idx == 4: # Metropolis step
            neighbors = list(self.G.neighbors(current_node))
            if neighbors:
                candidate = neighbors[self.rng.randint(len(neighbors))]
                acceptance = min(1.0, max(self.degree_dict[current_node], 1) / max(self.degree_dict[candidate], 1))
                if self.rng.random() < acceptance:
                    chosen = candidate
                else:
                    chosen = current_node # rejeitou, fica
            else:
                action_idx = 6
        elif action_idx == 5: # Snowball expansion (apenas pega um random vizinho da fronteira)
            # Simplificação snowball: pegar nó mais antigo da fronteira (simulando fila FIFO)
            # Como sets nao tem ordem, pegamos random
            chosen = frontier_list[self.rng.randint(len(frontier_list))]
        
        if action_idx == 6 or chosen is None: # Random jump
            remaining = list(set(self.nodes) - self.sampled)
            if remaining:
                chosen = remaining[self.rng.randint(len(remaining))]
            else:
                chosen = current_node
                
        return chosen, action_idx

    def sample(self):
        """Executa a amostragem completa."""
        start_node = self.rng.choice(self.nodes)
        self.sampled.add(start_node)
        for nb in self.G.neighbors(start_node):
            self.frontier.add(nb)
            
        current_node = start_node
        
        # Parâmetros de recompensa em bloco
        block_size = max(5, int(self.target_n * 0.05)) # a cada 5% avalia
        
        last_sps = self.compute_fast_sps()
        history = []
        action_counts = np.zeros(self.n_arms)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while len(self.sampled) < self.target_n:
                context = self.get_context()
                
                # Thompson Sampling: sample weights for each arm and compute expected reward
                expected_rewards = np.zeros(self.n_arms)
                for a in range(self.n_arms):
                    sampled_weights = self.arms[a].sample_weights(self.rng)
                    expected_rewards[a] = np.dot(context, sampled_weights)
                
                # Escolhe a melhor ação baseada na amostragem bayesiana
                chosen_action = np.argmax(expected_rewards)
                
                # Exploration epsilon-greedy fallback just in case
                if self.rng.random() < 0.05:
                    chosen_action = self.rng.randint(self.n_arms)
                    
                new_node, actual_action = self.take_action(chosen_action, current_node)
                
                self.sampled.add(new_node)
                self.frontier.discard(new_node)
                for nb in self.G.neighbors(new_node):
                    if nb not in self.sampled:
                        self.frontier.add(nb)
                
                current_node = new_node
                action_counts[actual_action] += 1
                
                history.append((context, actual_action))
                
                # Reward evaluation in blocks
                if len(self.sampled) % block_size == 0 or len(self.sampled) >= self.target_n:
                    current_sps = self.compute_fast_sps()
                    
                    # Reward = -(SPS_t - SPS_{t-1})
                    # Como JS divergência, menor é melhor. SPS caindo = recompensa positiva.
                    # Mas na formula acima js divergencia, menor é melhor.
                    # diff < 0 significa que melhorou. Queremos diff < 0 dando recompensa positiva.
                    diff = current_sps - last_sps
                    reward = -diff 
                    
                    # Atualiza modelo online
                    for ctx, act in history:
                        self.arms[act].update(ctx, reward)
                    
                    history = []
                    last_sps = current_sps

        # Retornar subgrafo
        subgraph = self.G.subgraph(list(self.sampled)).copy()
        subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0)
        
        metadata = {
            "original_n": self.n,
            "sampled_n": subgraph.number_of_nodes(),
            "sample_frac_actual": subgraph.number_of_nodes() / self.n,
            "sampler": "contextual_bandit",
            "action_counts": action_counts.tolist(),
        }
        
        return subgraph, metadata

def contextual_bandit_sampling(G, sample_frac, seed=42):
    sampler = ContextualBanditSampler(G, sample_frac, seed=seed)
    return sampler.sample()
