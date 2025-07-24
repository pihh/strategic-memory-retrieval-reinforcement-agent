
import torch
import torch.nn as nn
import numpy as np

class BaseMemoryBuffer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_last_attention(self):
        """Return the latest attention weights, if available."""
        raise NotImplementedError("Memory module must implement get_last_attention()")

class StrategicMemoryBuffer(BaseMemoryBuffer):
    """
    Episodic memory buffer for RL agents using neural trajectory encoding and
    attention-based retrieval, with learnable usefulness/retention scores.

    Features:
        - Stores episode trajectories and outcomes.
        - Each trajectory is encoded as a vector with a Transformer.
        - Each memory entry gets a trainable usefulness parameter.
        - When full, discards the least-useful entry.
        - Returns soft-attended memory readout given a context trajectory.

    Args:
        obs_dim (int): Observation dimension.
        action_dim (int): Action dimension (scalar=1).
        mem_dim (int): Embedding size for memory entries.
        max_entries (int): Max entries to keep (FIFO with learning).
        device (str): Device for tensors ("cpu" or "cuda").
    """

    __version__     = "1.3.0"
    __description__ = "Memory usefullness is now trainable and retention is based on it"


    def __init__(self, obs_dim, action_dim, mem_dim=32, max_entries=100, device='cpu'):
        super().__init__()
      
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mem_dim = mem_dim
        self.max_entries = max_entries
        self.device = device

        self.embedding_proj = nn.Linear(obs_dim + action_dim + 1, mem_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=mem_dim, nhead=2, batch_first=True),
            num_layers=1
        )

        # Fixed usefulness per slot (size max_entries)
        self.usefulness_vec = nn.Parameter(torch.zeros(max_entries, device=self.device), requires_grad=True)
        self.entries = []
        self._entry_indices = []  # Maps entries to slot indices

    def reset(self):
        """Clears all memory entries (keeps usefulness param, but marks buffer empty)."""
        self.entries = []
        self._entry_indices = []

    def add_entry(self, trajectory, outcome):
        """
        Stores a new trajectory/outcome. If full, replaces the least useful slot.
        """
        traj_np = np.array([np.concatenate([obs, [action], [reward]]) for obs, action, reward in trajectory], dtype=np.float32)
        traj = torch.from_numpy(traj_np).to(self.device)
        traj_proj = self.embedding_proj(traj)
        mem_embed = self.encoder(traj_proj.unsqueeze(0)).mean(dim=1).squeeze(0)
        entry = {
            'trajectory': trajectory,
            'outcome': outcome,
            'embedding': mem_embed.detach()
        }
        if len(self.entries) < self.max_entries:
            # Use next available slot (by index)
            slot_idx = len(self.entries)
            self.entries.append(entry)
            self._entry_indices.append(slot_idx)
        else:
            # Overwrite least-useful slot
            usefulness_scores = self.usefulness_vec.detach().cpu()
            idx_remove = usefulness_scores.argmin().item()
            self.entries[idx_remove] = entry
            self._entry_indices[idx_remove] = idx_remove

    def retrieve(self, context_trajectory):
        """
        Soft-attention retrieval over memory entries.
        """
        if len(self.entries) == 0:
            self.last_attn = None
            return torch.zeros(self.mem_dim, device=self.device), None
        traj_np = np.array([np.concatenate([obs, [action], [reward]]) for obs, action, reward in context_trajectory], dtype=np.float32)
        traj = torch.from_numpy(traj_np).to(self.device)
        traj_proj = self.embedding_proj(traj)
        context_embed = self.encoder(traj_proj.unsqueeze(0)).mean(dim=1).squeeze(0)
        mem_embeddings = torch.stack([e['embedding'] for e in self.entries])  # [N, mem_dim]
        attn_logits = torch.matmul(mem_embeddings, context_embed)
        attn = torch.softmax(attn_logits, dim=0)
        mem_readout = (attn.unsqueeze(1) * mem_embeddings).sum(dim=0)
        self.last_attn = attn.detach().cpu().numpy()
        return mem_readout, attn

    def usefulness_loss(self, attn, reward):
        """
        Compute usefulness loss: encourages usefulness score to match (attn * reward).
        Only updates the active slots (those with entries).
        """
        # attn: [N] attention weights used for the memory readout
        # reward: scalar, or [N]
        N = len(self.entries)
        if N == 0:
            return torch.tensor(0.0, device=self.device)
        idxs = self._entry_indices[:N]
        usefulness_vec = self.usefulness_vec[idxs]  # Only the in-use slots
        if isinstance(reward, (float, int)):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        targets = attn.detach() * reward  # [N]
        loss = ((usefulness_vec - targets) ** 2).mean()
        return loss

    def usefulness_parameters(self):
        return [self.usefulness_vec]

    def get_trainable_parameters(self):
        # Everything that should be trained here
        return list(self.parameters()) + list(self.usefulness_parameters())

    def get_last_attention(self):
        return getattr(self, "last_attn", None)

class StrategicMemoryTransformerPolicy(nn.Module):
    """
    Policy network that leverages a strategic memory buffer for RL.

    Features:
        - Processes sequence of observations with a Transformer encoder.
        - Attends to episodic memory via context-driven retrieval.
        - Supports auxiliary loss heads (plug-and-play).
        - Outputs both action logits and value estimate.

    Args:
        obs_dim (int): Input observation dimension.
        mem_dim (int): Size of feature and memory embedding.
        nhead (int): Number of attention heads in transformer.
        memory (StrategicMemoryBuffer): Episodic memory module (optional).
        aux_modules (list): Auxiliary modules (optional).

    Version: 1.2.0
    """
    __version__     = "1.2.0"
    __description__ = "Hint-free memory retrieval using strategic memory buffer"

    def __init__(self, obs_dim, mem_dim=32, nhead=4, memory=None, aux_modules=None, **kwargs):
        super().__init__()
        self.mem_dim = mem_dim
        self.embed = nn.Linear(obs_dim, mem_dim)
        self.pos_embed = nn.Embedding(256, mem_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mem_dim, nhead=nhead, batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Memory will be concatenated, so double the input size to heads
        self.policy_head = nn.Linear(mem_dim + mem_dim, 2)  # Discrete actions: 2
        self.value_head = nn.Linear(mem_dim + mem_dim, 1)
        self.aux_modules = aux_modules if aux_modules is not None else []
        self.memory = memory

    def forward(self, trajectory, obs_t=None, actions=None, rewards=None):
        """
        Forward pass for action selection and value estimation.

        Args:
            trajectory (torch.Tensor): [T, obs_dim] sequence of observations up to now.
            obs_t (optional): Not used, kept for compatibility.
            actions (optional): List or tensor of past actions for memory context.
            rewards (optional): List or tensor of past rewards for memory context.

        Returns:
            logits (torch.Tensor): [2,] Action logits (for discrete action space).
            value (torch.Tensor): [] State-value estimate.
            aux_preds (dict): Dict of auxiliary head predictions.
        """
        T = trajectory.shape[0]
        x = self.embed(trajectory)                 # [T, mem_dim]
        pos = torch.arange(T, device=trajectory.device)
        x = x + self.pos_embed(pos)                # Add position encoding
        x = x.unsqueeze(0)                         # [1, T, mem_dim]
        x = self.transformer(x)                    # [1, T, mem_dim]
        feat = x[0, -1]                            # [mem_dim], final token for policy

        # PATCH: better history alignment for memory 
        if self.memory is not None and actions is not None and rewards is not None:
            actions_list = actions.tolist()
            rewards_list = rewards.tolist()
            # Pad if actions/rewards are shorter than T (e.g., first step)
            if len(actions_list) < T:
                actions_list = [0] * (T - len(actions_list)) + actions_list
            if len(rewards_list) < T:
                rewards_list = [0.0] * (T - len(rewards_list)) + rewards_list
            context_traj = []
            for i in range(T):
                context_traj.append((
                    trajectory[i].cpu().numpy(),
                    actions_list[i],
                    rewards_list[i]
                ))
            mem_readout, attn = self.memory.retrieve(context_traj)
        else:
            mem_readout = torch.zeros_like(feat)

        # Concatenate memory and main features
        final_feat = torch.cat([feat, mem_readout], dim=-1)
        logits = self.policy_head(final_feat)
        value = self.value_head(final_feat)
        aux_preds = {}
        for aux in self.aux_modules:
            aux_preds[aux.name] = aux.head(final_feat)
        return logits, value.squeeze(-1), aux_preds

