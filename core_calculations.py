import torch 

from constants import DEFAULT_MEMORY_DIMENSION,DEFAULT_TRAJECTORY_LENGTH,DEFAULT_MEMORY_ENTRIES


# ──────────────────────────────────────────────────────────────
# Helper functions and defaults 
# ──────────────────────────────────────────────────────────────

def validate_memory_size(
        mem_dim=DEFAULT_MEMORY_DIMENSION,
        max_trajectory_len=DEFAULT_TRAJECTORY_LENGTH,
        max_entries= DEFAULT_MEMORY_ENTRIES
    ):
    mem_dim = max(mem_dim,DEFAULT_MEMORY_DIMENSION)
    max_trajectory_len= max(max_trajectory_len,mem_dim)
    max_entries = max(max_entries,mem_dim)
    return mem_dim,max_trajectory_len,max_entries

# ──────────────────────────────────────────────────────────────
# 4. Generalized Advantage Estimation (GAE) and Explained Variance
# ──────────────────────────────────────────────────────────────

def compute_explained_variance(y_pred, y_true):
    """
    Computes explained variance between prediction and ground-truth.
    Used for value function diagnostics in RL.
    """
    var_y = torch.var(y_true)
    if var_y == 0:
        return torch.tensor(0.0)
    return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

def compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    """
    Compute Generalized Advantage Estimation (GAE) for a trajectory.
    Args:
        rewards (torch.Tensor): reward sequence [T]
        values (torch.Tensor): value sequence [T]
        gamma (float): discount factor
        lam (float): GAE lambda
        last_value (float): bootstrap value after final state
    Returns:
        torch.Tensor: advantage sequence [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=values.device)
    gae = 0.0
    # concatenate last value for bootstrap
    values_ext = torch.cat([values, torch.tensor([last_value], dtype=torch.float32, device=values.device)])
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages