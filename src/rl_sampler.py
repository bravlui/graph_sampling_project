"""
rl_sampler.py — Self-Supervised Graph Sampling via RL (Nível 3 - C3)
=====================================================================

Implementa ambiente de RL e DQN para aprender política de amostragem
de grafos orientada à preservação estrutural.
"""

import numpy as np
import networkx as nx
import logging
from collections import deque

logger = logging.getLogger("graph_sampling")

# PyTorch (importação condicional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch não disponível. RL sampler desabilitado.")


# ============================================================================
# Ambiente de RL
# ============================================================================

class GraphSamplingEnv:
    """
    Ambiente para amostragem de grafos via RL.

    Estado (8 dimensões):
    - sample_size_ratio: fração atual de nós amostrados
    - avg_degree_sample: grau médio da amostra
    - degree_var_sample: variância do grau na amostra
    - avg_clustering_sample: clustering médio da amostra
    - num_components_sample: número de componentes conectados
    - frontier_size: tamanho da fronteira normalizado
    - last_node_degree: grau do último nó adicionado (normalizado)
    - target_sample_frac: fração alvo

    Ações (5):
    0: escolher nó da fronteira com maior grau
    1: escolher nó da fronteira com maior clustering
    2: escolher nó da fronteira com maior k-core
    3: escolher nó aleatório da fronteira
    4: random jump para nó não visitado
    """

    N_ACTIONS = 5
    STATE_DIM = 8

    def __init__(self, G, sample_frac=0.2, seed=0):
        """
        Parameters
        ----------
        G : nx.Graph
            Rede a amostrar.
        sample_frac : float
            Fração alvo de nós.
        seed : int
            Semente.
        """
        self.G = G
        self.n = G.number_of_nodes()
        self.target_n = max(1, int(round(self.n * sample_frac)))
        self.sample_frac = sample_frac
        self.rng = np.random.RandomState(seed)

        # Pré-computar
        self.degree_dict = dict(G.degree())
        self.max_degree = max(self.degree_dict.values()) if self.degree_dict else 1
        self.clustering_dict = nx.clustering(G)
        self.core_dict = nx.core_number(G)
        self.max_core = max(self.core_dict.values()) if self.core_dict else 1
        self.nodes = list(G.nodes())

        self.reset()

    def reset(self):
        """Reinicia o episódio."""
        start = self.rng.choice(self.nodes)
        self.sampled = {start}
        self.frontier = set()
        for nb in self.G.neighbors(start):
            self.frontier.add(nb)
        self.last_degree = self.degree_dict[start]
        self.done = False
        self.steps = 0
        return self.get_state()

    def get_state(self):
        """Retorna vetor de estado."""
        if not self.sampled:
            return np.zeros(self.STATE_DIM)

        sampled_list = list(self.sampled)
        sub = self.G.subgraph(sampled_list)

        degrees = [self.degree_dict[v] for v in sampled_list]
        avg_deg = np.mean(degrees) if degrees else 0
        deg_var = np.var(degrees) if degrees else 0

        # Clustering da amostra
        try:
            avg_clust = nx.average_clustering(sub)
        except Exception:
            avg_clust = 0

        n_components = nx.number_connected_components(sub)

        state = np.array([
            len(self.sampled) / self.n,             # sample_size_ratio
            avg_deg / self.max_degree,               # avg_degree_sample (normalizado)
            min(deg_var / (self.max_degree**2 + 1), 1.0),  # degree_var_sample
            avg_clust,                                # avg_clustering_sample
            n_components / max(len(self.sampled), 1), # num_components (normalizado)
            len(self.frontier) / max(self.n, 1),      # frontier_size
            self.last_degree / self.max_degree,        # last_node_degree
            self.sample_frac,                          # target
        ], dtype=np.float32)

        return state

    def step(self, action):
        """
        Executa uma ação.

        Parameters
        ----------
        action : int
            0-4

        Returns
        -------
        tuple (state, reward, done, info)
        """
        if self.done:
            return self.get_state(), 0.0, True, {}

        chosen = None
        frontier_list = list(self.frontier)

        if len(frontier_list) == 0 and action != 4:
            action = 4  # forçar random jump

        if action == 0:  # maior grau da fronteira
            if frontier_list:
                chosen = max(frontier_list, key=lambda v: self.degree_dict.get(v, 0))
        elif action == 1:  # maior clustering da fronteira
            if frontier_list:
                chosen = max(frontier_list, key=lambda v: self.clustering_dict.get(v, 0))
        elif action == 2:  # maior k-core da fronteira
            if frontier_list:
                chosen = max(frontier_list, key=lambda v: self.core_dict.get(v, 0))
        elif action == 3:  # aleatório da fronteira
            if frontier_list:
                chosen = frontier_list[self.rng.randint(len(frontier_list))]
        elif action == 4:  # random jump
            remaining = list(set(self.nodes) - self.sampled)
            if remaining:
                chosen = remaining[self.rng.randint(len(remaining))]

        if chosen is None:
            self.done = True
            return self.get_state(), 0.0, True, {}

        # Adicionar nó
        self.sampled.add(chosen)
        self.frontier.discard(chosen)
        for nb in self.G.neighbors(chosen):
            if nb not in self.sampled:
                self.frontier.add(nb)
        self.last_degree = self.degree_dict.get(chosen, 0)
        self.steps += 1

        # Verificar término
        if len(self.sampled) >= self.target_n:
            self.done = True
            reward = self.compute_reward()
            return self.get_state(), reward, True, {"final": True}

        # Recompensa intermediária: manter conectividade
        sub = self.G.subgraph(list(self.sampled))
        n_comp = nx.number_connected_components(sub)
        intermediate_reward = -0.01 * n_comp  # penalizar desconexão

        return self.get_state(), intermediate_reward, False, {}

    def compute_reward(self):
        """
        Recompensa final: negative SPS entre original e amostra.
        """
        from src.metrics import compute_node_distributions, compute_global_metrics
        from src.distances import compare_all_distributions, compare_global_metrics
        from src.experiments import compute_sps

        sub = self.G.subgraph(list(self.sampled)).copy()
        sub = nx.convert_node_labels_to_integers(sub, first_label=0)

        try:
            orig_dists = compute_node_distributions(self.G, approx_betweenness=True, seed=0)
            orig_globals = compute_global_metrics(self.G, approx_betweenness=True, seed=0)
            samp_dists = compute_node_distributions(sub, approx_betweenness=True, seed=0)
            samp_globals = compute_global_metrics(sub, approx_betweenness=True, seed=0)

            dist_comp = compare_all_distributions(orig_dists, samp_dists)
            global_comp = compare_global_metrics(orig_globals, samp_globals)
            sps = compute_sps(dist_comp, global_comp)

            if np.isnan(sps):
                return -1.0
            return -sps  # negativo pois queremos minimizar SPS
        except Exception as e:
            logger.warning(f"Erro no cálculo de reward: {e}")
            return -1.0

    def get_sampled_graph(self):
        """Retorna subgrafo amostrado relabelado."""
        sub = self.G.subgraph(list(self.sampled)).copy()
        return nx.convert_node_labels_to_integers(sub, first_label=0)


# ============================================================================
# Rede Q (DQN)
# ============================================================================

if HAS_TORCH:
    class QNetwork(nn.Module):
        """MLP para Q-values."""

        def __init__(self, state_dim=8, action_dim=5, hidden_sizes=(128, 64)):
            super().__init__()
            layers = []
            prev = state_dim
            for h in hidden_sizes:
                layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            layers.append(nn.Linear(prev, action_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    class ReplayBuffer:
        """Buffer de experiências para DQN."""

        def __init__(self, capacity=10000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            states, actions, rewards, next_states, dones = zip(*batch)
            return (
                torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.BoolTensor(dones),
            )

        def __len__(self):
            return len(self.buffer)


# ============================================================================
# Treinamento DQN
# ============================================================================

def train_dqn_sampler(
    graph_generator_fn,
    num_episodes=500,
    sample_frac=0.20,
    seed=42,
    gamma=0.95,
    lr=1e-3,
    batch_size=64,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    target_update_freq=10,
    replay_buffer_size=10000,
):
    """
    Treina DQN para aprender política de amostragem.

    Parameters
    ----------
    graph_generator_fn : callable
        Função que gera uma rede nx.Graph (pode variar a cada episódio).
    num_episodes : int
    sample_frac : float
    seed : int
    gamma, lr, batch_size, etc. : hiperparâmetros DQN.

    Returns
    -------
    tuple (QNetwork, list[dict])
        Modelo treinado e log de treinamento.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch necessário para DQN.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    q_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(replay_buffer_size)

    epsilon = epsilon_start
    training_log = []

    for episode in range(num_episodes):
        G = graph_generator_fn()
        env = GraphSamplingEnv(G, sample_frac=sample_frac, seed=seed + episode)
        state = env.reset()

        total_reward = 0
        ep_steps = 0

        while not env.done:
            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(GraphSamplingEnv.N_ACTIONS)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.argmax(dim=1).item()

            next_state, reward, done, info = env.step(action)
            replay.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            ep_steps += 1

            # Treinar
            if len(replay) >= batch_size:
                states, actions, rewards, next_states, dones = replay.sample(batch_size)

                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_states).max(dim=1)[0]
                    targets = rewards + gamma * next_q * (~dones).float()

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decaimento epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Atualizar target network
        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        training_log.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": ep_steps,
            "epsilon": epsilon,
            "sampled_n": len(env.sampled),
        })

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean([l["total_reward"] for l in training_log[-50:]])
            logger.info(f"Episode {episode+1}/{num_episodes} — "
                       f"avg_reward={avg_reward:.4f}, eps={epsilon:.3f}")

    return q_net, training_log


def learned_rl_sampling(G, sample_frac, model_path, seed=0):
    """
    Usa modelo DQN treinado para amostrar uma rede.

    Parameters
    ----------
    G : nx.Graph
    sample_frac : float
    model_path : str
        Caminho para o modelo .pt salvo.
    seed : int

    Returns
    -------
    tuple (nx.Graph, dict)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch necessário.")

    q_net = QNetwork()
    q_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    q_net.eval()

    env = GraphSamplingEnv(G, sample_frac=sample_frac, seed=seed)
    state = env.reset()
    action_counts = np.zeros(GraphSamplingEnv.N_ACTIONS)

    while not env.done:
        with torch.no_grad():
            q_values = q_net(torch.FloatTensor(state).unsqueeze(0))
            action = q_values.argmax(dim=1).item()

        action_counts[action] += 1
        state, _, done, _ = env.step(action)

    subgraph = env.get_sampled_graph()
    metadata = {
        "original_n": G.number_of_nodes(),
        "sampled_n": subgraph.number_of_nodes(),
        "sample_frac_actual": subgraph.number_of_nodes() / G.number_of_nodes(),
        "sampler": "rl_sampler",
        "action_counts": action_counts.tolist(),
    }

    return subgraph, metadata
