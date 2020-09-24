from TLSTM.tlstm import TLSTMCell
import torch
import torch.nn as N
import torch.nn.functional as F
from config import ModelType, ModelLossMode


class NonSeqModel(N.Module):
    """
     This is a three layer model mapping OHE features to representations
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NonSeqModel, self).__init__()
        self.mlp1 = N.Linear(input_dim, hidden_dim)
        self.mlp2 = N.Linear(hidden_dim, hidden_dim)
        self.mlp3 = N.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.selu(self.mlp2(x))
        return self.mlp3(x)


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


class MixModel(N.Module):

    def __init__(self, config, model_type=ModelType.M_LSTM):
        super(MixModel, self).__init__()

        if model_type is ModelType.M_TLSTM:
            # TLSTM hidden state dim = (B, h)
            self.seq_model = TLSTMCell(config.seq_input_dim, config.seq_hidden_dim)
        elif model_type is ModelType.M_LSTM:
            # LSTM hidden state dim = (batch, num_layers * num_directions, hidden_size)
            self.seq_model = N.LSTM(config.seq_input_dim, config.seq_hidden_dim, batch_first=True)
        else:
            raise NotImplementedError("We only support model ctlsm and clstm but get {}".format(model_type.value))

        self.non_seq_model = NonSeqModel(
            config.nonseq_input_dim, config.nonseq_hidden_dim, config.nonseq_output_dim)

        self.model_type = model_type
        self.dropout_rate = config.dropout_rate
        self.merge_layer = N.Linear(config.nonseq_output_dim+config.seq_hidden_dim, config.mix_hidden_dim)
        self.classifier = N.Linear(config.mix_hidden_dim, config.mix_output_dim)
        self.loss_mode = config.loss_mode

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

        # TODO we need to work on this part of the network: test different non-linear function; test number of layers
        raw_rep = self.merge_layer(m_rep)
        m_rep = torch.tanh(F.dropout(raw_rep, p=self.dropout_rate))

        # (B, 2)
        logits = self.classifier(m_rep)
        pred_prob = F.softmax(logits, dim=-1)

        if self.loss_mode is ModelLossMode.BIN:
            # y dim (B, 2)
            loss = F.binary_cross_entropy(pred_prob, y)
            # loss = F.binary_cross_entropy_with_logits(logits, y)
        elif self.loss_mode is ModelLossMode.MUL:
            # y dim (B, 1)
            # y_hat = y.type(torch.long)
            loss = F.cross_entropy(logits, y)
        else:
            raise NotImplementedError("loss mode only support bin or mul but get {}".format(self.loss_mode.value))

        return loss, pred_prob, torch.argmax(pred_prob, dim=-1), m_rep
