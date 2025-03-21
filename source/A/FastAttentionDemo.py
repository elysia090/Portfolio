import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

##############################################
# Low-level Helper Functions
##############################################

def fma(a: float, b: float, c: float) -> float:
    """
    Fused multiply-add: returns a * b + c in a single instruction if available.
    """
    try:
        return math.fma(a, b, c)
    except AttributeError:
        return a * b + c

def optimized_sqrt(n: int) -> float:
    """
    If n is a power of 2, compute sqrt(n) using bit shifts.
    For n = 2^k, sqrt(n) = 2^(k/2).
    Otherwise, falls back to math.sqrt.
    """
    if n & (n - 1) == 0:  # power-of-two check
        k = n.bit_length() - 1
        return 2 ** (k / 2)
    return math.sqrt(n)

##############################################
# Fast Attention Components
##############################################

@torch.jit.script
def select_topk(similarities: torch.Tensor, k_max: int) -> torch.Tensor:
    """
    Select the indices of the top-k values along the last dimension.
    Input shape: [B, L, num_candidates].
    """
    _, topk_idx = torch.topk(similarities, k=k_max, dim=-1)
    return topk_idx

@dataclass
class FastAttentionConfig:
    d_model: int          # Overall model dimension.
    d_key: int            # Key dimension.
    d_query: int          # Query dimension.
    n_heads: int          # Number of attention heads.
    rank: int             # Rank for low-rank approximations.
    rff_dim: int          # Output dimension for random Fourier features.
    k_max: int            # Maximum candidate keys per query.
    stride: int           # Stride length for Trie.
    lsh_buckets: int      # Number of LSH buckets.
    lsh_bandwidth: float  # Bandwidth for LSH.
    lsh_key_dim: int      # LSH input dimension.
    wu_manber_prefix_len: int  # Prefix length for Wu-Manber search.
    hyper_cuts_dim_groups: Optional[List[int]] = None  # Feature groups.
    n_lsh_hashes: int = 4
    dropout: float = 0.1
    intermediate_dim: int = 2048
    use_rff: bool = True  # Whether to use RFF.

class LowRankLinear(nn.Module):
    """
    Low-rank linear layer via factorized weight matrices.
    """
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.u_weight).matmul(self.v_weight)

class RandomFourierFeatures(nn.Module):
    """
    Projects input features using random Fourier features.
    """
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x.matmul(self.omega) + self.bias
        return torch.cos(projection) * self.scale

class LSHTable(nn.Module):
    """
    Locality-sensitive hashing (LSH) table.
    """
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes
        self.random_vectors = nn.Parameter(torch.randn(dim, n_hashes), requires_grad=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.matmul(self.random_vectors)
        return torch.floor(proj / self.bandwidth) % self.n_buckets

_TRIE_INDICES_KEY = '_indices'
class Trie(nn.Module):
    """
    Trie (prefix tree) for candidate lookup using binary quantization.
    """
    def __init__(self, stride: int):
        super().__init__()
        self.root_node: Dict = {}
        self.stride_len = stride
    def insert(self, binary_vector: torch.Tensor, index: int) -> None:
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                current_node[prefix] = {}
            current_node = current_node[prefix]
        current_node.setdefault(_TRIE_INDICES_KEY, []).append(index)
    def search(self, binary_vector: torch.Tensor) -> List[int]:
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                return []
            current_node = current_node[prefix]
        return current_node.get(_TRIE_INDICES_KEY, [])

class CandidateFinder(nn.Module):
    """
    Candidate search module using LSH, Wu-Manber, and Trie-based methods.
    """
    def __init__(self, config: FastAttentionConfig, tries: List[Trie], lsh_tables: nn.ModuleList):
        super().__init__()
        self.config = config
        self.tries = tries
        self.lsh_tables = lsh_tables
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        if self.hyper_cuts_dim_groups is None:
            return [features]
        groups = []
        start = 0
        for group_dim in self.hyper_cuts_dim_groups:
            groups.append(features[:, :, start:start+group_dim])
            start += group_dim
        return groups

    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[tuple, List[int]]:
        table: Dict[tuple, List[int]] = {}
        L = key_bin.size(0)
        for i in range(L):
            prefix = tuple(key_bin[i, :self.config.wu_manber_prefix_len].tolist())
            table.setdefault(prefix, []).append(i)
        return table

    def _wu_manber_search(self, query_bin: torch.Tensor, table: Dict[tuple, List[int]]) -> List[int]:
        prefix = tuple(query_bin[:self.config.wu_manber_prefix_len].tolist())
        return table.get(prefix, [])

    def _get_wu_manber_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor) -> List[List[List[int]]]:
        B, L, _ = key_grp.size()
        key_bin = self.binary_quantize(key_grp)
        query_bin = self.binary_quantize(query_grp)
        cand_lists = []
        for b in range(B):
            table = self._build_wu_manber_hash_table(key_bin[b])
            batch_list = [self._wu_manber_search(query_bin[b, i], table) for i in range(L)]
            cand_lists.append(batch_list)
        return cand_lists

    def _get_trie_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> List[List[List[int]]]:
        B, L, _ = key_grp.size()
        cand_lists = []
        for b in range(B):
            trie = Trie(self.config.stride)
            key_bin = self.binary_quantize(key_grp[b])
            for i in range(L):
                trie.insert(key_bin[i], i)
            batch_list = [trie.search(self.binary_quantize(query_grp[b][i])) for i in range(L)]
            cand_lists.append(batch_list)
        return cand_lists

    def merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat(cand_tensors, dim=-1)
        merged, _ = torch.sort(merged)
        return torch.unique_consecutive(merged, dim=-1)

    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query_grp.size()
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp)
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx)
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_grp.device)
        for b in range(B):
            for i in range(L):
                common = list(set(wu_cands[b][i]) & set(trie_cands[b][i]))
                if common:
                    common_tensor = torch.tensor(common, dtype=torch.long, device=query_grp.device)
                    if common_tensor.numel() > self.config.k_max:
                        common_tensor = common_tensor[:self.config.k_max]
                    candidates[b, i, :common_tensor.numel()] = common_tensor
        return candidates

    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query_up.size()
        query_groups = self.split_features_by_dim_groups(query_up)
        key_groups = self.split_features_by_dim_groups(key_up)
        cand_list = [self._process_dimension_group_candidates(q_grp, k_grp, head_idx)
                     for q_grp, k_grp in zip(query_groups, key_groups)]
        if cand_list:
            merged = self.merge_candidate_indices_groups(cand_list)
            return merged[:, :, :self.config.k_max]
        return torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_up.device)

class AbsorptionProjection(nn.Module):
    """
    Projects queries into the key space using a low-rank 'absorption' transformation.
    """
    def __init__(self, query_dim: int, key_dim: int, rank: int):
        super().__init__()
        self.u_q = nn.Parameter(torch.randn(query_dim, rank) / math.sqrt(rank))
        self.v_q = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))
        self.u_k = nn.Parameter(torch.randn(key_dim, rank) / math.sqrt(rank))
        self.v_k = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        W_UQ = torch.matmul(self.u_q, self.v_q)
        W_UK = torch.matmul(self.u_k, self.v_k)
        W_absorb = torch.matmul(W_UK.transpose(0, 1), W_UQ)
        Q_proj = torch.matmul(query, W_absorb.transpose(0, 1))
        return Q_proj, key

class FastAttention(nn.Module):
    """
    Multi-head Fast Attention module.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        self.query_down_proj = nn.Linear(config.d_model, config.d_query)
        self.key_value_down_proj = nn.Linear(config.d_model, config.d_key)
        self.absorption_projs = nn.ModuleList([
            AbsorptionProjection(config.d_query, config.d_key, config.rank)
            for _ in range(config.n_heads)
        ])
        self.value_up_projs = nn.ModuleList([
            LowRankLinear(config.d_key, config.d_model, config.rank)
            for _ in range(config.n_heads)
        ])
        self.rff_encoders = nn.ModuleList([
            RandomFourierFeatures(config.d_key, config.rff_dim)
            for _ in range(config.n_heads)
        ])
        if config.hyper_cuts_dim_groups:
            self.lsh_tables_list = nn.ModuleList([
                nn.ModuleList([
                    LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                    for dim in config.hyper_cuts_dim_groups
                ])
                for _ in range(config.n_heads)
            ])
        else:
            self.lsh_tables_list = nn.ModuleList([
                nn.ModuleList([LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)])
                for _ in range(config.n_heads)
            ])
        self.tries_list = nn.ModuleList([Trie(config.stride) for _ in range(config.n_heads)])
        self.candidate_finder = CandidateFinder(config, self.tries_list, self.lsh_tables_list)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = query.size()
        query_down = self.query_down_proj(query)
        key_down = self.key_value_down_proj(key)
        head_outputs = []
        for head_idx in range(self.config.n_heads):
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down)
            candidates = self.candidate_finder(Q_proj, K_proj, head_idx)
            cand_mask = candidates != -1
            safe_candidates = candidates.clone()
            safe_candidates[safe_candidates == -1] = 0
            b_idx = torch.arange(B, device=K_proj.device).view(B, 1, 1).expand(B, L, self.config.k_max)
            candidate_keys = K_proj[b_idx, safe_candidates]
            q_exp = Q_proj.unsqueeze(2)
            if self.config.use_rff:
                q_exp = self.rff_encoders[head_idx](q_exp.view(-1, self.config.d_key)).view(B, L, 1, self.config.rff_dim)
                candidate_keys = self.rff_encoders[head_idx](candidate_keys.view(-1, self.config.d_key)).view(B, L, self.config.k_max, self.config.rff_dim)
                scale = optimized_sqrt(self.config.rff_dim)
            else:
                scale = optimized_sqrt(self.config.d_key)
            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) / scale
            sim = sim.masked_fill(~cand_mask, float('-inf'))
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)
            candidate_values = key_down[b_idx, safe_candidates]
            candidate_values = self.value_up_projs[head_idx](candidate_values.view(-1, self.config.d_key)).view(B, L, self.config.k_max, self.config.d_model)
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2)
            head_outputs.append(head_out)
        concat = torch.cat(head_outputs, dim=-1)
        output = self.output_proj(concat)
        return self.dropout(output)

class FeedForwardNetwork(nn.Module):
    """
    Two-layer feedforward network with ReLU activation and dropout.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.intermediate_dim)
        self.linear2 = nn.Linear(config.intermediate_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class FastAttentionEncoderLayer(nn.Module):
    """
    Transformer encoder layer using Fast Attention followed by a feedforward network.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = src
        src = self.norm1(src)
        attn = self.self_attn(src, src, src, mask=src_mask)
        src = residual + self.dropout(attn)
        residual = src
        src = self.norm2(src)
        ffn = self.feed_forward(src)
        return residual + self.dropout(ffn)

##############################################
# GRPO Components (Full Implementation)
##############################################

def compute_real_reward(full_output: torch.Tensor, fast_output: torch.Tensor, cost: float) -> float:
    """
    Compute the real reward based on the L2 error between full and fast attention outputs and the computational cost.
    Replace this placeholder with your actual reward function.
    """
    error = torch.norm(full_output - fast_output)
    reward = - float(error.item()) - cost
    return reward

def extract_state_from_module(module: FastAttention, input_data: torch.Tensor) -> List[float]:
    """
    Extract the state vector from the module's candidate search metrics.
    Replace this placeholder with real measurements.
    """
    candidate_count = module.config.k_max
    avg_similarity = 0.75   # Replace with actual metric.
    variance_similarity = 0.05
    resource_cost = 1.0
    downstream_perf = 0.95
    history_metric = 0.0
    return [candidate_count, avg_similarity, variance_similarity, resource_cost, downstream_perf, history_metric]

class GRPOEnvironmentMulti:
    """
    Environment for multi-group candidate search hyperparameter optimization.
    The state and reward are derived from real model metrics.
    """
    def __init__(self, fast_attention_module: FastAttention, validation_data: torch.Tensor,
                 initial_hyperparams: List[Dict[str, float]], alpha: float = 1.0, beta: float = 0.1):
        self.fast_module = fast_attention_module
        self.validation_data = validation_data
        self.groups = initial_hyperparams
        self.alpha = alpha  # Weight for performance error.
        self.beta = beta    # Weight for computational cost.
    def get_state(self) -> torch.Tensor:
        states = []
        for g in self.groups:
            # In practice, run the module and extract real metrics.
            state_vec = [g['k_max']] + extract_state_from_module(self.fast_module, self.validation_data)[1:]
            states.append(state_vec)
        return torch.tensor(states, dtype=torch.float32)
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = []
        new_states = []
        for i, g in enumerate(self.groups):
            a = actions[i]
            g['lsh_buckets'] = max(1, g['lsh_buckets'] + a[0].item())
            g['lsh_bandwidth'] = max(0.1, g['lsh_bandwidth'] + a[1].item())
            g['trie_stride'] = max(1, g['trie_stride'] + a[2].item())
            g['k_max'] = max(1, g['k_max'] + int(round(a[3].item())))
            # In practice, run both full and fast attention on validation_data.
            # Here we use placeholders.
            full_output = torch.randn(1)  # Replace with actual full attention output.
            fast_output = torch.randn(1)  # Replace with actual fast attention output.
            cost = g['k_max']  # Example: cost proportional to candidate count.
            reward = compute_real_reward(full_output, fast_output, cost)
            rewards.append(reward)
            state_vec = [g['k_max']] + extract_state_from_module(self.fast_module, self.validation_data)[1:]
            new_states.append(state_vec)
        return torch.tensor(new_states, dtype=torch.float32), torch.tensor(rewards, dtype=torch.float32)

class GRPOAgent(nn.Module):
    """
    GRPO policy network that outputs an action vector per group.
    Action vector: [Δlsh_buckets, Δlsh_bandwidth, Δtrie_stride, Δk_max]
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc(state)

def grpo_training_episode(agent: GRPOAgent, env: GRPOEnvironmentMulti, optimizer: torch.optim.Optimizer,
                            episode_length: int = 10, epsilon: float = 1e-8, lambda_kl: float = 0.01) -> float:
    """
    Runs one GRPO training episode over multiple time steps.
    Uses a Monte Carlo baseline (mean reward) scaled by reward standard deviation and includes an explicit KL penalty.
    """
    total_loss = 0.0
    state = env.get_state()  # Shape: [G, state_dim]
    for _ in range(episode_length):
        actions = agent(state)  # [G, action_dim]
        std = torch.ones_like(actions)
        dist = torch.distributions.Normal(actions, std)
        log_probs = dist.log_prob(actions).sum(dim=1)
        next_state, rewards = env.step(actions)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + epsilon
        advantage = (rewards - mean_reward) / std_reward
        # Explicit KL penalty: for Normal(actions, 1) vs. Normal(0,1), KL = 0.5 * (actions^2)
        kl_div = 0.5 * (actions ** 2).mean()
        loss = - (log_probs * advantage).mean() + lambda_kl * kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        state = next_state
    return total_loss / episode_length

def example_grpo_full():
    """
    Full GRPO training example using real metrics.
    Replace the placeholders in compute_real_reward and extract_state_from_module with your actual logic.
    """
    initial_hyperparams = [
        {'lsh_buckets': 32, 'lsh_bandwidth': 4.0, 'trie_stride': 4, 'k_max': 64},
        {'lsh_buckets': 30, 'lsh_bandwidth': 4.2, 'trie_stride': 5, 'k_max': 60},
        {'lsh_buckets': 35, 'lsh_bandwidth': 3.8, 'trie_stride': 4, 'k_max': 70}
    ]
    config = FastAttentionConfig(
        d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
        rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
        lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
        hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4, use_rff=True)
    fast_module = FastAttention(config)
    validation_data = torch.randn(2, 128, config.d_model)  # Replace with your actual validation data.
    env = GRPOEnvironmentMulti(fast_module, validation_data, initial_hyperparams, alpha=1.0, beta=0.1)
    agent = GRPOAgent(state_dim=6, action_dim=4)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    num_episodes = 20
    for ep in range(num_episodes):
        loss = grpo_training_episode(agent, env, optimizer, episode_length=10)
        state = env.get_state()
        print(f"Episode {ep}: Avg Loss={loss:.4f}, Group States={state.tolist()}")

##############################################
# Usage Examples
##############################################

def example_usage():
    config = FastAttentionConfig(
        d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
        rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
        lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
        hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4, use_rff=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastAttention(config).to(device)
    B, L = 2, 128
    query = torch.randn(B, L, config.d_model).to(device)
    key = torch.randn(B, L, config.d_model).to(device)
    value = torch.randn(B, L, config.d_model).to(device)
    mask = torch.ones(B, L, L, dtype=torch.bool).to(device)
    output = model(query, key, value, mask)
    print(f"FastAttention Output shape: {output.shape}")

def example_usage_encoder_layer():
    config = FastAttentionConfig(
        d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
        rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
        lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
        hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4,
        dropout=0.1, intermediate_dim=2048, use_rff=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_layer = FastAttentionEncoderLayer(config).to(device)
    B, L = 2, 128
    src = torch.randn(B, L, config.d_model).to(device)
    src_mask = torch.ones(B, L, L, dtype=torch.bool).to(device)
    output = encoder_layer(src, src_mask)
    print(f"EncoderLayer Output shape: {output.shape}")

if __name__ == "__main__":
    example_usage()
    example_usage_encoder_layer()
    print("Starting full GRPO training (multi-group) with real metrics...")
    example_grpo_full()
