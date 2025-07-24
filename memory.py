
import torch
import torch.nn as nn
import numpy as np

class StrategicMemoryBuffer(nn.Module):
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
        self.reset()
        self.embedding_proj = nn.Linear(obs_dim + action_dim + 1, mem_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=mem_dim, nhead=2, batch_first=True),
            num_layers=1
        )

    def reset(self):
        """
        Clears all memory entries and usefulness parameters.
        """
        self.entries = []
        self.usefulness = nn.ParameterList()  # Each is a nn.Parameter([1], float)

    def add_entry(self, trajectory, outcome):
        """
        Stores a new trajectory/outcome, initializes its usefulness as a trainable parameter.
        If full, discards the least-useful entry.
        """
        traj = torch.tensor(
            [np.concatenate([obs, [action], [reward]]) for obs, action, reward in trajectory],
            dtype=torch.float32, device=self.device
        )
        traj_proj = self.embedding_proj(traj)
        mem_embed = self.encoder(traj_proj.unsqueeze(0)).mean(dim=1).squeeze(0)
        entry = {
            'trajectory': trajectory,
            'outcome': outcome,
            'embedding': mem_embed.detach()  # Detach for safe storage (no gradients!)
        }
        self.entries.append(entry)
        self.usefulness.append(nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=self.device), requires_grad=True))
        # Remove least-useful if buffer is full
        if len(self.entries) > self.max_entries:
            # Pick min usefulness (detached so doesn't block gradient)
            usefulness_scores = torch.stack([p.detach() for p in self.usefulness]).squeeze(-1)
            idx_remove = usefulness_scores.argmin().item()
            del self.entries[idx_remove]
            del self.usefulness[idx_remove]

    def retrieve(self, context_trajectory):
        """
        Retrieves memory with soft-attention and also returns attention for use in loss.
        """
        if len(self.entries) == 0:
            return torch.zeros(self.mem_dim, device=self.device), None
        traj = torch.tensor(
            [np.concatenate([obs, [action], [reward]]) for obs, action, reward in context_trajectory],
            dtype=torch.float32, device=self.device
        )
        traj_proj = self.embedding_proj(traj)
        context_embed = self.encoder(traj_proj.unsqueeze(0)).mean(dim=1).squeeze(0)
        mem_embeddings = torch.stack([e['embedding'] for e in self.entries])  # [N, mem_dim]
        attn_logits = torch.matmul(mem_embeddings, context_embed)  # [N]
        attn = torch.softmax(attn_logits, dim=0)
        mem_readout = (attn.unsqueeze(1) * mem_embeddings).sum(dim=0)
        self.last_attn = attn.detach().cpu().numpy()
        return mem_readout, attn

    def usefulness_loss(self, attn, rewards):
        """
        Computes usefulness loss: encourages retention of useful entries.
        Args:
            attn (torch.Tensor): attention weights used for the decision [N]
            rewards (float): outcome/reward from this trajectory
        Returns:
            torch.Tensor: usefulness loss
        """
        # Each memory has a parameterized usefulness score
        usefulness_vec = torch.cat([u for u in self.usefulness])  # [N]
        # Targets: should match high reward * attended entries (simple heuristic)
        # You can make this more sophisticated as you wish!
        targets = attn.detach() * rewards  # attended entries responsible for outcome
        loss = ((usefulness_vec - targets) ** 2).mean()
        return loss

    def usefulness_parameters(self):
        return self.usefulness.parameters()
