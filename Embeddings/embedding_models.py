"""
a simple model to handle EHR seq data with embeddings
we support LSTM, GRU, TLSTM, and TCN as learning framework
"""


import torch
from torch import nn
import sys
sys.path.append("../")

from TLSTM.tlstm import TLSTMCell
from TCN.tcn import TemporalConvNet
from config import ModelType, ModelLossMode


class SeqEHRConfig:
    def __init__(
            self, input_dim=10, output_dim=1, hidden_dim=128, emb_dim=32,
            model_type=ModelType.M_GRU, drop_prob=0.1):
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.drop_prob = drop_prob

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            s += "{}={}\n".format(k, v)
        return s


class SeqEHR(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model_type = config.model_type
        if self.model_type is ModelType.M_TLSTM:
            # TLSTM hidden state dim = (B, h)
            self.seq_model = TLSTMCell(config.emb_dim, config.hidden_dim)
        elif self.model_type is ModelType.M_LSTM:
            # LSTM hidden state dim = (batch, num_layers * num_directions, hidden_size)
            self.seq_model = nn.LSTM(config.emb_dim, config.hidden_dim, batch_first=True)
        elif self.model_type is ModelType.M_GRU:
            # LSTM hidden state dim = (batch, num_layers * num_directions, hidden_size)
            self.seq_model = nn.GRU(config.emb_dim, config.hidden_dim, batch_first=True)
        else:
            raise NotImplementedError(
                "We only support model lstm, gru, tlstm, tcn but get {}".format(
                    self.model_type.value))


