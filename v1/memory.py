
class StrategicMemoryBuffer(nn.Module):
    """
    Episodic memory buffer for RL agents using neural trajectory encoding and
    attention-based retrieval.

    Features:
        - Stores a list of episode trajectories and their outcomes.
        - Encodes each trajectory as a vector embedding using a Transformer encoder.
        - Allows content-based retrieval (soft attention) given a new context trajectory.

    Args:
        obs_dim (int): Dimension of state/observation vector.
        action_dim (int): Dimension of action vector (typically scalar, but supports >1).
        mem_dim (int): Size of embedding for memory entries.
        max_entries (int): Maximum number of memory entries to store (FIFO).
        device (str): Device (e.g., "cpu" or "cuda").
    """
    def __init__(self, obs_dim, action_dim, mem_dim=32, max_entries=100, device='cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mem_dim = mem_dim
        self.max_entries = max_entries
        self.device = device
        self.reset()
        # Projects [obs | action | reward] into mem_dim
        self.embedding_proj = nn.Linear(obs_dim + action_dim + 1, mem_dim)
        # Single-layer transformer encoder for sequence modeling
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=mem_dim, nhead=2, batch_first=True),
            num_layers=1
        )

    def reset(self):
        """
        Clears all memory entries (FIFO buffer).
        """
        self.entries = []

    def add_entry(self, trajectory, outcome):
        """
        Stores a new trajectory and its outcome in the buffer.

        Args:
            trajectory (list of (obs, action, reward)): Episode trace.
            outcome (any): Outcome/label for this episode (can be unused).
        """
        # Convert to torch tensor: each row [obs..., action, reward]
        traj = torch.tensor(
            [np.concatenate([obs, [action], [reward]]) for obs, action, reward in trajectory],
            dtype=torch.float32, device=self.device
        )  # [T, obs_dim + action_dim + 1]
        traj_proj = self.embedding_proj(traj)  # [T, mem_dim]
        # Get a fixed-size embedding for the whole trajectory (mean pooling)
        mem_embed = self.encoder(traj_proj.unsqueeze(0)).mean(dim=1).squeeze(0)  # [mem_dim]
        entry = {
            'trajectory': trajectory,
            'outcome': outcome,
            'embedding': mem_embed.detach()  # Detach for safe storage
        }
        self.entries.append(entry)
        # Enforce buffer size
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def retrieve(self, context_trajectory):
        """
        Retrieves relevant memory by attention-weighted readout over embeddings.

        Args:
            context_trajectory (list of (obs, action, reward)): The current trajectory context.

        Returns:
            mem_readout (torch.Tensor): [mem_dim] attention-weighted memory readout.
            attn (torch.Tensor or None): [num_entries] attention weights or None if empty.
        """
        if len(self.entries) == 0:
            return torch.zeros(self.mem_dim, device=self.device), None

        traj = torch.tensor(
            [np.concatenate([obs, [action], [reward]]) for obs, action, reward in context_trajectory],
            dtype=torch.float32, device=self.device
        )
        traj_proj = self.embedding_proj(traj)
        context_embed = self.encoder(traj_proj.unsqueeze(0)).mean(dim=1).squeeze(0)  # [mem_dim]
        mem_embeddings = torch.stack([e['embedding'] for e in self.entries])  # [N, mem_dim]
        # Attention: inner product similarity
        attn_logits = torch.matmul(mem_embeddings, context_embed)
        attn = torch.softmax(attn_logits, dim=0)
        mem_readout = (attn.unsqueeze(1) * mem_embeddings).sum(dim=0)
        return mem_readout, attn


class MemoryTransformerPolicy(nn.Module):
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
