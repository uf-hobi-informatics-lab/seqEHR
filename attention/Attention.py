"""
    We would like to explore various attention mechanism
    https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#a-family-of-attention-mechanisms
"""

import torch
from torch import nn


class SoftAttention(nn.Module):

    def __init__(self):
        super(SoftAttention, self).__init__()

    def forward(self, x):
        pass


class SelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
