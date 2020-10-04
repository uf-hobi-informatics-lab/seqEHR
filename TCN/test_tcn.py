from tcn import TemporalConvNet
import torch
from torch import nn
import pickle


def pkl_load(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


def pkl_save(data, fn):
    with open(fn, "wb") as f:
        pickle.dump(data, f)


class TestModel(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs, labels):
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        loss = nn.functional.binary_cross_entropy_with_logits(labels, o)
        return loss, o, torch.argmax(o, dim=1)