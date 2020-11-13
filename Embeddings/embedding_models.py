"""
a simple model to handle EHR seq data with embeddings
we support LSTM, GRU, TLSTM, and TCN as learning framework
"""


import torch
from torch import nn
import sys
sys.path.append("../")

from TLSTM.tlstm import TLSTMCell
from common_utils.config import ModelType, ModelLossMode, EmbeddingReductionMode


class SeqEmbEHRConfig:
    def __init__(self, input_dim=10, output_dim=1, hidden_dim=128, emb_dim=32, drop_prob=0.1, emb_freeze=False,
                 model_type=ModelType.M_GRU, loss_type=ModelLossMode.BIN, merge_type=EmbeddingReductionMode.SUM):
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.loss_type = loss_type
        self.merge_type = merge_type
        self.drop_prob = drop_prob
        self.emb_freeze=emb_freeze

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            s += "{}={}\n".format(k, v)
        return s


class SeqEmbEHR(nn.Module):

    def __init__(self, config, emb_weights=None):
        super().__init__()

        self.merge_type = config.merge_type
        self.loss_type = config.loss_type
        self.model_type = config.model_type

        self.classifier = nn.Linear(config.hidden_dim, config.output_dim)
        self.drop_output = nn.Dropout(p=config.drop_prob)

        # could be replaced by EmbeddingBag
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.tensor(emb_weights, dtype=torch.float32), freeze=config.emb_freeze)
        self.emb_dim = self.embedding_layer.embedding_dim

        if self.merge_type is EmbeddingReductionMode.AVG:
            # we do not apply adjust linear transformation on average case
            self.adjust_layer = None
        elif self.merge_type is EmbeddingReductionMode.FUSE:
            raise NotImplementedError("TODO: keep all embedding weights as features")
        else:
            self.adjust_layer = nn.Linear(self.emb_dim, self.emb_dim)

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
                "We only support model lstm, gru, tlstm but get {}".format(
                    self.model_type.value))

    def forward(self, seqs, labels, times=None):
        # seqs (B, S, F) - batch, seq, feature as ids
        # labels (B, L)

        # (B, S, F) = > (B, S, F, E)
        x = self.embedding_layer(seqs)

        # merge F and E
        if self.merge_type is EmbeddingReductionMode.SUM:
            x = torch.sum(x, dim=2)
            x = self.adjust_layer(x)
        elif self.merge_type is EmbeddingReductionMode.MAX:
            # torch.max return a tuple: (metrics, indices)
            x = torch.max(x, dim=2)[0]
            x = self.adjust_layer(x)
        elif self.merge_type is EmbeddingReductionMode.AVG:
            x = torch.mean(x, dim=2)
        elif self.merge_type is EmbeddingReductionMode.FUSE:
            raise NotImplementedError("TODO: keep all embedding weights as features")
        else:
            raise ValueError("Not support current mode: {}".format(self.merge_type))

        # sequence model
        if self.model_type is ModelType.M_TLSTM:
            h_f, (h_t, c_t) = self.seq_model(x, times)
        else:
            h_f, (h_t, c_t) = self.seq_model(x)
            h_t = h_t.squeeze(0)

        raw_rep = self.drop_output(h_t)

        # output
        outputs = self.classifier(raw_rep)
        pred_prob = nn.functional.softmax(outputs, dim=-1)

        # calc loss
        if self.loss_type is ModelLossMode.BIN:
            # y dim (B, 2)
            loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels)
        elif self.loss_type is ModelLossMode.MUL:
            # y dim (B, 1)
            loss = nn.functional.cross_entropy(outputs, labels)
        else:
            raise NotImplementedError("loss mode only support bin or mul but get {}".format(self.loss_mode.value))

        return loss, pred_prob, torch.argmax(outputs, dim=-1), raw_rep
