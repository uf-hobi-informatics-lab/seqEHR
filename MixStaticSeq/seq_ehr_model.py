import torch
import torch.nn.functional as F
from torch import nn

import sys
sys.path.append("../")


from common_utils.config import ModelLossMode, ModelType
from TLSTM.tlstm import TLSTMCell

from collections import OrderedDict


class NonSeqModel(nn.Module):
    """
     This is a MLP model mapping OHE features to representations
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_mlp=2, with_non_linearity=False):
        super(NonSeqModel, self).__init__()

        if num_mlp == 1:
            self.mlp = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            for _ in range(num_mlp-2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if with_non_linearity:
                    layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))

            self.mlp = nn.Sequential(
                OrderedDict({str(i): layer for i, layer in enumerate(layers)})
            )

    def forward(self, x):
        return self.mlp(x)


class MixModelConfig(object):

    def __init__(self, seq_input_dim, nonseq_input_dim, dropout_rate=0.1,
                 nonseq_hidden_dim=128, seq_hidden_dim=128, mix_hidden_dim=128,
                 nonseq_output_dim=64, mix_output_dim=2, loss_mode=ModelLossMode.BIN, **kwargs):
        super(MixModelConfig, self).__init__()
        self.seq_input_dim = seq_input_dim
        self.seq_hidden_dim = seq_hidden_dim
        self.nonseq_input_dim = nonseq_input_dim
        self.nonseq_hidden_dim = nonseq_hidden_dim
        self.nonseq_output_dim = nonseq_output_dim
        self.mix_hidden_dim = mix_hidden_dim
        self.dropout_rate = dropout_rate
        self.mix_output_dim = mix_output_dim
        self.loss_mode = loss_mode

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise Warning("Can't set {} with value {} for {}".format(key, value, self))

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            s += "{}={}\n".format(k, v)
        return s


class MixModel(nn.Module):
    """
     The model takes two kinds of features:
        1. static features as demographics which do not change with time
        2. dynamic features as diagnoses or labs which change with time
     The model use a MLP to model feature 1 and use lstm or tlstm to model feature 2
     Then prediction is based on representation learned from both features
    """

    def __init__(self, config, model_type=ModelType.M_LSTM):
        super(MixModel, self).__init__()

        if model_type == ModelType.M_TLSTM:
            # TLSTM hidden state dim = (B, h)
            self.seq_model = TLSTMCell(config.seq_input_dim, config.seq_hidden_dim)
        elif model_type == ModelType.M_LSTM:
            # LSTM hidden state dim = (batch, num_layers * num_directions, hidden_size)
            self.seq_model = nn.LSTM(config.seq_input_dim, config.seq_hidden_dim, batch_first=True)
        elif model_type == ModelType.M_GRU:
            # LSTM hidden state dim = (batch, num_layers * num_directions, hidden_size)
            self.seq_model = nn.GRU(config.seq_input_dim, config.seq_hidden_dim, batch_first=True)
        else:
            raise NotImplementedError("We only support model tlsm, lstm, gru currently but get {}"
                                      .format(model_type.value))

        self.non_seq_model = NonSeqModel(
            config.nonseq_input_dim, config.nonseq_hidden_dim, config.nonseq_output_dim,
            config.mlp_num, config.with_non_linearity)

        self.model_type = model_type
        self.dropout_rate = config.dropout_rate
        self.merge_layer = nn.Linear(config.nonseq_output_dim+config.seq_hidden_dim, config.mix_hidden_dim)
        self.classifier = nn.Linear(config.mix_hidden_dim, config.mix_output_dim)
        self.loss_mode = config.loss_mode
        self.sampling_weight = torch.tensor(config.sampling_weight, dtype=torch.float) \
            if ('sampling_weight' in config.__dict__ and config.sampling_weight) else None

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
                # N.init.normal_(p.data, mean=0.0, std=0.1)
            else:
                nn.init.zeros_(p.data)
                # N.init.ones_(p.data)

    def forward(self, x=None):
        """
        Since the data will be load with dataloader, batch_size=1 but batch dim still exist
        :param x: x[0] non-seq features, x[1] sequence features, x[2] time elapse if using TLSTM
        :param y: binary labels (by default should be float32; when using CrossEntropy we will convert them to Long)
        :return: loss, logits, predicted labels
        """
        y = x[-1]  # the last is always label for input
        non_seq_x = x[0]  # (B, T)
        non_seq_rep = self.non_seq_model(non_seq_x)

        seq_x = x[1]  # (B, S, T)
        if self.model_type is ModelType.M_TLSTM:
            time = x[2]  # (B, S, 1)
            _, (seq_rep, _) = self.seq_model(seq_x, time)
        else:
            # seq rep dim = (1, B, h)
            _, (seq_rep, _) = self.seq_model(seq_x)
            # (B, h)
            seq_rep = seq_rep.squeeze(0)

        # non_seq_rep: (B, h)   seq_rep: (B, h)
        m_rep = torch.cat([non_seq_rep, seq_rep], dim=1)

        # merge (B, h+h)
        raw_rep = self.merge_layer(m_rep)
        m_rep = F.dropout(raw_rep, p=self.dropout_rate)

        # (B, 2)
        logits = self.classifier(m_rep)
        pred_prob = F.softmax(logits, dim=-1)

        if self.loss_mode == ModelLossMode.BIN:
            # y dim (B, 2)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=self.sampling_weight)
        elif self.loss_mode == ModelLossMode.MUL:
            # y dim (B, 1)
            loss = F.cross_entropy(logits, y, weight=self.sampling_weight)
        else:
            raise NotImplementedError("loss mode only support bin or mul but get {}".format(self.loss_mode.value))

        return loss, pred_prob, torch.argmax(pred_prob, dim=-1), m_rep


class MixEmbeddingModel(MixModel):
    """
     This is a model extended on MixModel.
     The MixModel using pre-defined feature representation formats
     such as One-hot encoding, binary encoding or raw numbers
    """

    def __init__(self, config, model_type=ModelType.M_LSTM):
        super(MixEmbeddingModel, self).__init__(config, model_type)
        self.medical_emb = nn.Embedding.from_pretrained(torch.tensor(config.embedding), freeze=False, padding_idx=0)


class MixTCNModel(nn.Module):
    """
        Using TCN instead of LSTM
    """
    pass
