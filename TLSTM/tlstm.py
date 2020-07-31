#################
# Time-Aware LSTM
# Originally published in "Patient Subtyping via Time-Aware LSTM Networks", KDD, 2017
# This is the Pytorch version re-implementation
# Author: BugFace
#################

import torch
from torch.nn.parameter import Parameter
from torch import nn as N
from torch.nn import functional as F
import numpy as np


class TLSTMConfig:
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, dropoutput_rate):
        self.dropout_prob = dropoutput_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim


class SoftmaxCrossEntropyLoss(N.Module):
    """
    equivalent implementation of tf.nn.softmax_cross_entropy_with_logits
    """

    def __init__(self, weight=None):
        super().__init__()

    def forward(self, inputs, targets):
        if not targets.is_same_size(inputs):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(),
                                                                                           inputs.size()))

        inputs = F.softmax(inputs, dim=1)
        loss = -torch.sum(targets * torch.log(inputs), 1)
        loss = torch.unsqueeze(loss, 1)
        return torch.mean(loss)


class TLSTMCell(N.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wi = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Ui = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bi = Parameter(torch.Tensor(hidden_dim))

        self.Wf = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Uf = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bf = Parameter(torch.Tensor(hidden_dim))

        self.Wog = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Uog = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bog = Parameter(torch.Tensor(hidden_dim))

        self.Wc = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Uc = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bc = Parameter(torch.Tensor(hidden_dim))

        self.W_decomp = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_decomp = Parameter(torch.Tensor(hidden_dim))

        self.init_weights()

        self.c1 = torch.tensor(1)
        self.c2 = torch.tensor(np.e)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                # N.init.xavier_uniform_(p.data)
                N.init.normal_(p.data, mean=0.0, std=0.1)
            else:
                # N.init.zeros_(p.data)
                N.init.ones_(p.data)

    def forward(self, x: torch.Tensor, time: torch.Tensor, prev_hidden_state=None):
        # x has three dim: batch size(pats), seq_size(encounters), v_dim(features)
        bz, seq_sz, v_dim = x.size()
        tbz, tseq_sz, tv_dim = time.size()
        assert bz == tbz and seq_sz == tseq_sz, \
            "feature and time seq have different batch size {}-{} or seq len {}-{}".format(bz, tbz, seq_sz, tseq_sz)

        # init hidden if no previous hidden
        if prev_hidden_state is None:
            h_t, c_t = torch.zeros(self.hidden_dim).to(x.device), torch.zeros(self.hidden_dim).to(x.device)
        else:
            h_t, c_t = prev_hidden_state

        # recurrent loop
        hidden_seq = []
        for i, t in enumerate(range(seq_sz)):
            x_t = x[:, t, :]  # (batch, input)
            # process time difference
            t_t = time[:, t, :]
            T = self.map_elapse_time(t_t)
            C_ST = torch.tanh(c_t @ self.W_decomp + self.b_decomp)
            C_ST_dis = torch.mul(T, C_ST)
            c_t = c_t - C_ST + C_ST_dis
            # input gate
            i_t = torch.sigmoid(x_t @ self.Wi + h_t @ self.Ui + self.bi)
            # forget gate
            f_t = torch.sigmoid(x_t @ self.Wf + h_t @ self.Uf + self.bf)
            # output gate
            o_t = torch.sigmoid(x_t @ self.Wog + h_t @ self.Uog + self.bog)
            # candidate MemCell
            c_h = torch.tanh(x_t @ self.Wc + h_t @ self.Uc + self.bc)
            # current MemCell
            c_t = f_t * c_t + i_t * c_h
            # current hidden state
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))  # create extra dim for later concat (seq, batch, input)

        hidden_seq = torch.cat(hidden_seq, dim=0)  # concat to get the seq dim back (seq, batch, input)
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous()  # (seq, batch, input) => (batch, seq, input)
        return hidden_seq, (h_t, c_t)

    def map_elapse_time(self, t):
        T = torch.div(self.c1, torch.log(t + self.c2))
        grid = torch.ones((1, self.hidden_dim))
        return torch.matmul(T, grid)


class TLSTM(N.Module):

    def __init__(self, config=None):
        super().__init__()
        # self.fc_dim = fc_dim
        # self.output_dim = output_dim
        self.tlstm = TLSTMCell(config.input_dim, config.hidden_dim)
        self.dropout_prob = config.dropout_prob
        self.fc_layer = N.Linear(config.hidden_dim, config.fc_dim)
        self.output_layer = N.Linear(config.fc_dim, config.output_dim)

        # we can use pytorch default implementation of binary loss function - BCEWithLogitsLoss
        # but we found the output did not exactly match the tf.nn.softmax_cross_entropy_with_logits with mean reduce
        # To make sure, results repeatable, we will use SoftmaxCrossEntropyLoss for now
        # self.loss_fct = N.BCEWithLogitsLoss()
        self.loss_fct = SoftmaxCrossEntropyLoss()

    def forward(self, feature, time, labels):
        # get raw logits
        seq, (h_t, c_t) = self.tlstm(feature, time)
        # seq = seq[-1, :, :]  # (seq, batch, input); get the last seq for classification
        last_state = h_t
        last_state = F.relu(self.fc_layer(last_state))
        last_state = F.dropout(last_state, p=self.dropout_prob)
        logits = self.output_layer(last_state)

        # measure loss
        loss = self.loss_fct(logits, labels)
        return loss, logits, torch.argmax(logits, dim=1)