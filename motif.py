
import torch
import torch.nn as nn
import numpy as np

from constants import DEFAULT_DEVICE,DEFAULT_MOTIF_LEN, DEFAULT_MEMORY_DIMENSION, DEFAULT_TRAJECTORY_LENGTH
from memory import BaseMemoryBuffer
from sklearn.cluster import KMeans


def mine_motifs_from_buffer(episodic_buffer, motif_bank, motif_len=DEFAULT_MOTIF_LEN, n_motifs=DEFAULT_MEMORY_DIMENSION, min_windows=DEFAULT_TRAJECTORY_LENGTH):
    """
    Mine most frequent motifs from episodic buffer and refresh the motif bank.
    - motif_bank: instance of MotifMemoryBank
    - episodic_buffer: should have .entries (each has 'trajectory')
    """
    subtraj_embeds = []
    for entry in episodic_buffer.entries:
        traj = entry['trajectory']
        if len(traj) >= motif_len:
            for i in range(len(traj) - motif_len + 1):
                window = traj[i:i+motif_len]
                # Convert window to tensor and embed
                window_np = np.array([np.concatenate([obs, [a], [r]]) for obs, a, r in window], dtype=np.float32)
                window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(motif_bank.device)
                with torch.no_grad():
                    embed = motif_bank.encoder(motif_bank.embedding_proj(window_tensor)).mean(dim=1).squeeze(0).cpu().numpy()
                subtraj_embeds.append(embed)
    if len(subtraj_embeds) < max(n_motifs, min_windows):
        print(f"Motif mining: not enough motif windows ({len(subtraj_embeds)}) for {n_motifs} motifs.")
        return
    X = np.stack(subtraj_embeds)
    kmeans = KMeans(n_clusters=n_motifs, random_state=0, n_init="auto").fit(X)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=motif_bank.device)
    with torch.no_grad():
        motif_bank.motif_embeds.copy_(centroids)
    print(f"[Motif mining] Updated motif bank with {n_motifs} clusters from buffer.")

class MotifMemoryBank(BaseMemoryBuffer):
    """
    Motif memory: learnable bank of pattern embeddings, attention-retrieved.

    Features:
        - Stores K motif embeddings, trainable.
        - Neural encoder to encode subtrajectories as motifs.
        - Attention over motifs given current context trajectory.
    """
    def __init__(self, obs_dim, action_dim, mem_dim=DEFAULT_MEMORY_DIMENSION, n_motifs=DEFAULT_MEMORY_DIMENSION, motif_len=DEFAULT_MOTIF_LEN, device=DEFAULT_DEVICE):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mem_dim = mem_dim
        self.n_motifs = n_motifs
        self.motif_len = motif_len
        self.device = device
        self.last_attn = None
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
        return motif_readout, None, attn  # PATCH: None for epi_attn, attn for motif_attn

    def motif_parameters(self):
        return [self.motif_embeds]

    def get_trainable_parameters(self):
        return list(self.parameters()) + list(self.motif_parameters())

    def get_last_attention(self):
        return self.last_attn
