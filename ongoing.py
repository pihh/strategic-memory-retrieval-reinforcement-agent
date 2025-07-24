
class MotifMemoryBank(nn.Module):

    def __init__(self, obs_dim, action_dim, mem_dim=32, n_motifs=32, motif_len=4, device='cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mem_dim = mem_dim
        self.n_motifs = n_motifs
        self.motif_len = motif_len
        self.device = device

        # Learnable motif memory bank
        self.motif_embeds = nn.Parameter(torch.randn(n_motifs, mem_dim))
        # Neural encoder for extracting motifs from subtrajectories
        self.embedding_proj = nn.Linear(obs_dim + action_dim + 1, mem_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=mem_dim, nhead=2, batch_first=True),
            num_layers=1
        )

    def retrieve(self, context_traj):
    
        if len(context_traj) < self.motif_len:
            pad = [context_traj[0]] * (self.motif_len - len(context_traj))
            motif_traj = pad + context_traj
        else:
            motif_traj = context_traj[-self.motif_len:]

        motif_np = np.array([np.concatenate([obs, [a], [r]]) for obs, a, r in motif_traj], dtype=np.float32)
        motif_input = torch.from_numpy(motif_np).unsqueeze(0).to(self.device)
        motif_embed = self.encoder(self.embedding_proj(motif_input)).mean(dim=1).squeeze(0)  # [mem_dim]
        attn_logits = torch.matmul(self.motif_embeds, motif_embed)
        attn = torch.softmax(attn_logits, dim=0)
        motif_readout = (attn.unsqueeze(1) * self.motif_embeds).sum(dim=0)
        self.last_attn = attn.detach().cpu().numpy()
        return motif_readout, attn

    def motif_parameters(self):
        return [self.motif_embeds]


class CombinedMemoryModule(nn.Module):

    def __init__(self, episodic_buffer, motif_bank):
        super().__init__()
        self.episodic_buffer = episodic_buffer
        self.motif_bank = motif_bank

    def retrieve(self, context_trajectory):
        epi_readout, epi_attn = self.episodic_buffer.retrieve(context_trajectory)
        motif_readout, motif_attn = self.motif_bank.retrieve(context_trajectory)
        combined = torch.cat([epi_readout, motif_readout], dim=-1)
        self.last_attn = (epi_attn, motif_attn)  # For logging/usefulness loss
        return combined, epi_attn, motif_attn


class StrategicMemoryTransformerPolicyV4(nn.Module):
    def __init__(self, obs_dim, mem_dim=32, nhead=4, memory=None, aux_modules=None, **kwargs):
        super().__init__()
        self.mem_dim = mem_dim
        self.embed = nn.Linear(obs_dim, mem_dim)
        self.pos_embed = nn.Embedding(256, mem_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=mem_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.policy_head = nn.Linear(mem_dim + 2 * mem_dim, 2)   # now +2mem_dim (episodic + motif)
        self.value_head = nn.Linear(mem_dim + 2 * mem_dim, 1)
        self.aux_modules = aux_modules if aux_modules is not None else []
        self.memory = memory

    def forward(self, trajectory, obs_t=None, actions=None, rewards=None):
        T = trajectory.shape[0]
        x = self.embed(trajectory)
        pos = torch.arange(T, device=trajectory.device)
        x = x + self.pos_embed(pos)
        x = x.unsqueeze(0)
        x = self.transformer(x)
        feat = x[0, -1]

        mem_feat = torch.zeros(2 * self.mem_dim, device=feat.device)
        epi_attn, motif_attn = None, None
        if self.memory is not None and actions is not None and rewards is not None:
            actions_list = actions.tolist()
            rewards_list = rewards.tolist()
            if len(actions_list) < T:
                actions_list = [0] * (T - len(actions_list)) + actions_list
            if len(rewards_list) < T:
                rewards_list = [0.0] * (T - len(rewards_list)) + rewards_list
            context_traj = [
                (trajectory[i].cpu().numpy(), actions_list[i], rewards_list[i]) for i in range(T)
            ]
            mem_feat, epi_attn, motif_attn = self.memory.retrieve(context_traj)
        final_feat = torch.cat([feat, mem_feat], dim=-1)
        logits = self.policy_head(final_feat)
        value = self.value_head(final_feat)
        aux_preds = {}
        for aux in self.aux_modules:
            aux_preds[aux.name] = aux.head(final_feat)
        return logits, value.squeeze(-1), aux_preds


"""
episodic_buffer = StrategicMemoryBuffer(obs_dim=..., action_dim=..., mem_dim=..., max_entries=..., device=device)
motif_bank = MotifMemoryBank(obs_dim=..., action_dim=..., mem_dim=..., n_motifs=32, motif_len=4, device=device)
combined_memory = CombinedMemoryModule(episodic_buffer, motif_bank)
policy = StrategicMemoryTransformerPolicy(obs_dim=..., mem_dim=..., memory=combined_memory, aux_modules=...)
agent = StrategicMemoryAgent(policy_class=..., env=..., memory=combined_memory, ...)



self.optimizer = optim.Adam(
    list(self.policy.parameters()) +
    list(self.memory.episodic_buffer.usefulness_parameters()) +
    list(self.memory.motif_bank.motif_parameters()),
    lr=learning_rate
) 


"""