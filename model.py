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
