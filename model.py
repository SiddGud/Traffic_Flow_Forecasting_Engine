import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialAttention(nn.Module):
    """
    Multi-head self-attention over the spatial (node) dimension.
    """

    def __init__(self, out_features: int, num_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(out_features, out_features)
        self.k_proj = nn.Linear(out_features, out_features)
        self.v_proj = nn.Linear(out_features, out_features)
        self.norm1  = nn.LayerNorm(out_features)
        self.norm2  = nn.LayerNorm(out_features)
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(x)
        key   = self.k_proj(x)
        value = self.v_proj(x)
        query = torch.cat(torch.split(query, self.head_dim, -1), 0).permute(0, 2, 1, 3)
        key   = torch.cat(torch.split(key,   self.head_dim, -1), 0).permute(0, 2, 3, 1)
        value = torch.cat(torch.split(value, self.head_dim, -1), 0).permute(0, 2, 1, 3)
        attn  = torch.matmul(query, key) / (self.head_dim ** 0.5)
        attn  = torch.softmax(attn, dim=-1)
        out   = torch.matmul(attn, value)
        out = torch.cat(torch.split(out, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        out = self.norm1(out + x)
        out = self.norm2(self.ffn(out) + out)
        return out


class GRUBlock(nn.Module):
    """GRU applied independently at each graph node."""

    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.gru   = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.proj  = nn.Linear(hidden_dim, out_features)
        self.norm  = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out    = self.proj(out)
        return self.norm(out)


class STBlock(nn.Module):
    """One Spatio-Temporal block: spatial attention followed by temporal GRU."""

    def __init__(self, out_features: int, num_heads: int, head_dim: int, gru_hidden: int):
        super().__init__()
        self.spatial  = SpatialAttention(out_features, num_heads, head_dim)
        self.temporal = GRUBlock(out_features, gru_hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        x = self.spatial(x)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        x = self.temporal(x)
        x = x.reshape(B, N, T, F).permute(0, 2, 1, 3)
        return x


class STGNN(nn.Module):
    """Spatio-Temporal Graph Neural Network for traffic flow forecasting."""

    def __init__(self, num_nodes, in_features, out_features, num_heads, head_dim,
                 num_layers, gru_hidden, P, Q, se_dim):
        super().__init__()
        self.input_proj = nn.Linear(in_features + se_dim, out_features)
        self.blocks     = nn.ModuleList([
            STBlock(out_features, num_heads, head_dim, gru_hidden)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(out_features * P, Q)

    def forward(self, x: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        B, P, N, _ = x.shape
        se = se.unsqueeze(0).unsqueeze(0).expand(B, P, N, -1)
        x  = torch.cat([x, se], dim=-1)
        x  = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)
        return x
