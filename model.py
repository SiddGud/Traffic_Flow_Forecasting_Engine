import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


class Transform(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )

        self.d = d

    def forward(self, x):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0,2,1,3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,2,3,1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0,2,1,3)

        A = torch.matmul(query, key)
        A /= (self.d ** 0.5)
        A = torch.softmax(A, -1)

        value = torch.matmul(A ,value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0,2,1,3)
        value += x

        value = self.ln(value)
        x = self.ff(value) + value
        # forgot return here, fix later


class GRUBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.gru  = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.proj(out)


class STGNN(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, num_heads, head_dim, num_layers, gru_hidden, P, Q, se_dim):
        super().__init__()
        self.input_proj  = nn.Linear(in_features + se_dim, out_features)
        self.gru_enc     = GRUBlock(out_features, gru_hidden, out_features)
        # TODO: add stacked spatio-temporal blocks properly
        self.output_proj = nn.Linear(out_features * P, Q)

    def forward(self, x, se):
        B, P, N, _ = x.shape
        se = se.unsqueeze(0).unsqueeze(0).expand(B, P, N, -1)
        x  = torch.cat([x, se], dim=-1)
        x  = self.input_proj(x)
        # TODO: not correct yet, just passing through GRU for now
        x  = x.permute(0, 2, 1, 3).reshape(B * N, P, -1)
        x, _ = self.gru_enc.gru(x)
        x  = self.gru_enc.proj(x)
        x  = x.reshape(B, N, P, -1).permute(0, 2, 1, 3)
        x  = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        x  = self.output_proj(x)
        x  = x.permute(0, 2, 1)
        return x
