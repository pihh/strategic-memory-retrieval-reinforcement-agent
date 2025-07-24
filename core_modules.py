import time
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from collections import defaultdict

# ──────────────────────────────────────────────────────────────
# 2. Reward Normalizer
# ──────────────────────────────────────────────────────────────

class RewardNormalizer:
    """
    Online running mean/variance normalizer for rewards.
    Purpose:
        - Stabilizes reinforcement learning by normalizing reward signals.
        - Maintains running statistics (mean, variance) over time.
    Usage:
        norm = RewardNormalizer()
        norm.update([r1, r2, ...])
        norm.normalize([r1, r2, ...])
    """
    def __init__(self, epsilon=1e-8):
        """
        Args:
            epsilon (float): Small value to avoid division by zero.
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4  # prevents division by zero
        self.epsilon = epsilon

    def update(self, rewards):
        """
        Update the running mean and variance given a batch of new rewards.
        Args:
            rewards (array-like): Sequence of reward values (e.g., list or np.ndarray).
        """
        rewards = np.array(rewards)
        batch_mean = rewards.mean()
        batch_var = rewards.var()
        batch_count = len(rewards)
        # Online update for running statistics (Welford’s algorithm style)
        self.mean = (self.mean * self.count + batch_mean * batch_count) / (self.count + batch_count)
        self.var = (self.var * self.count + batch_var * batch_count) / (self.count + batch_count)
        self.count += batch_count

    def normalize(self, rewards):
        """
        Normalize a batch of rewards using the running mean and variance.
        Args:
            rewards (array-like): List or np.ndarray of reward values.
        Returns:
            list: Normalized rewards (zero mean, unit variance).
        """
        rewards = np.array(rewards)
        return ((rewards - self.mean) / (np.sqrt(self.var) + self.epsilon)).tolist()

# ──────────────────────────────────────────────────────────────
# 3. State Counter for Intrinsic Reward (Exploration Bonus)
# ──────────────────────────────────────────────────────────────

class StateCounter:
    """
    Simple state visitation counter for intrinsic motivation in RL agents.

    Features:
        - Discretizes observations (using rounding) to count state visits.
        - Returns an intrinsic reward bonus: 1 / sqrt(visit count).
        - Used to encourage exploration by rewarding novel states.

    Usage:
        sc = StateCounter()
        reward = sc.intrinsic_reward(obs)
    """
    def __init__(self):
        """
        Initializes the internal state visitation counter.
        """
        self.counts = defaultdict(int)

    def count(self, obs):
        """
        Increment and return the visitation count for a (discretized) observation.
        Args:
            obs (np.ndarray): The observed state (continuous values).
        Returns:
            int: The number of times this (discretized) state has been seen.
        """
        key = tuple(np.round(obs, 2))  # Discretize for generalization and efficiency
        self.counts[key] += 1
        return self.counts[key]

    def intrinsic_reward(self, obs):
        """
        Compute the intrinsic exploration bonus for the current observation.
        The bonus is inversely proportional to the sqrt of the visit count (decreases with repetition).
        Args:
            obs (np.ndarray): Current observation.
        Returns:
            float: Intrinsic exploration reward.
        """
        c = self.count(obs)
        return 1.0 / np.sqrt(c)

# ──────────────────────────────────────────────────────────────
# 7. Random Network Distillation (RND)
# ──────────────────────────────────────────────────────────────

class RNDModule(nn.Module):
    """
    Random Network Distillation (RND) module for intrinsic motivation in RL.

    Purpose:
        - Provides a "novelty" or "surprise" signal for each observation.
        - Novelty is high when the predictor's embedding diverges from the fixed random target network.
        - Used to encourage exploration in hard-exploration RL tasks (see: https://arxiv.org/abs/1810.12894).

    Args:
        obs_dim (int): Dimension of environment observation vector.
        emb_dim (int): Embedding dimension for target and predictor networks.

    Attributes:
        target (nn.Sequential): Fixed random network (parameters are frozen).
        predictor (nn.Sequential): Trainable network that tries to match target's output.

    Usage:
        rnd = RNDModule(obs_dim)
        novelty = rnd(obs_tensor)  # [batch] float: intrinsic reward per observation
    """
    def __init__(self, obs_dim, emb_dim=32):
        super().__init__()
        # Fixed random target network (parameters are not updated after initialization)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        # Predictor network (learns to approximate target's outputs)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        # Freeze target net parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        """
        Computes RND novelty signal for a batch of observations.

        Args:
            obs (torch.Tensor): [batch_size, obs_dim] input observations.

        Returns:
            torch.Tensor: [batch_size] novelty (intrinsic reward) for each observation.
        """
        with torch.no_grad():
            target_emb = self.target(obs)
        pred_emb = self.predictor(obs)
        # MSE per sample, mean over embedding dim (last axis)
        novelty = F.mse_loss(pred_emb, target_emb, reduction='none').mean(dim=-1)
        return novelty
