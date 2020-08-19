from TLSTM.tlstm import TLSTMCell
import torch
import torch.nn as N
import torch.nn.functional as F


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


class MixModelConfig():

    def __init__(self, seq_input_dim, nonseq_input_dim, dropout_rate=0.1,
                 nonseq_hidden_dim=128, seq_hidden_dim=128, mix_hidden_dim=128,
                 nonseq_output_dim=64, mix_output_dim=2):
        self.seq_input_dim = seq_input_dim
        self.seq_hidden_dim = seq_hidden_dim
        self.nonseq_input_dim = nonseq_input_dim
        self.nonseq_hidden_dim = nonseq_hidden_dim
        self.nonseq_output_dim = nonseq_output_dim
        self.mix_hidden_dim = mix_hidden_dim
        self.dropout_rate = dropout_rate
        self.mix_output_dim = mix_output_dim


class MixModel(N.Module):

    def __init__(self, config, model_type='ctlsm'):
        super(MixModel, self).__init__()

        if model_type == 'ctlstm':
            self.seq_model = TLSTMCell(config.seq_input_dim, config.seq_hidden_dim)
        elif model_type == 'clstm':
            self.seq_model = N.LSTM(config.seq_input_dim, config.seq_hidden_dim)
        else:
            raise NotImplementedError("We only support model ctlsm and clstm but get {}".format(model_type))

        self.non_seq_model = NonSeqModel(
            config.nonseq_input_dim, config.nonseq_hidden_dim, config.nonseq_output_dim)

        self.dropout_rate = config.dropout_rate
        self.merge_layer = N.Linear(config.nonseq_output_dim+config.seq_hidden_dim, config.mix_hidden_dim)
        self.classifier = N.Linear(config.mix_hidden_dim, config.mix_output_dim)

    def forward(self, x, y):
        """
        :param x: x[0] non-seq features, x[1] sequence features
        :param y: binary labels
        :return: loss, logits, predicted labels
        """
        non_seq_x = x[0]
        seq_x = x[1]

        non_seq_x = self.non_seq_model(non_seq_x)
        seq_x = self.seq_model(seq_x)

        m_x = torch.cat([non_seq_x, seq_x], dim=1)
        m_x = F.tanh(F.dropout(self.merge_layer(m_x), p=self.dropout_rate))

        logits = self.classifier(m_x)
        pred_prob = F.softmax(logits)
        loss = F.binary_cross_entropy(pred_prob, y)

        return loss, pred_prob, torch.argmax(pred_prob)
