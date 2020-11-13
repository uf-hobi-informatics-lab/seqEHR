import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from common_utils.config import ModelType, ModelLossMode, EmbeddingReductionMode


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalConvNetEHRConfig:
    def __init__(self, input_dim=16, hidden_dim=128, output_dim=1, num_tcn_blocks=4, use_emb=False,
                 kernel_size=3, drop_prob=0.1, loss_type=ModelLossMode.BIN, keep_dim=False):
        self.num_inputs = input_dim
        self.kernel_size = kernel_size
        self.drop_prob = drop_prob
        self.output_dim = output_dim
        self.loss_type = loss_type
        # derive num_channels using hidden_dim, num_tcn_blocks, and input_dim
        self.num_channels = [hidden_dim] * (num_tcn_blocks - 1) + [input_dim]
        # if keep dim set to True, the output shape will be (B, S, O) else (B, O)
        # if keep dim results will be returned without loss calculation
        self.keep_dim = keep_dim
        # flag whether to use embedding layer as input
        self.use_emb = use_emb

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            s += "{}={}\n".format(k, v)
        return s


class TemporalConvNetEHR(nn.Module):

    def __init__(self, conf=None, emb_weights=None):
        super().__init__()
        self.loss_type = conf.loss_type
        self.keep_dim = conf.keep_dim

        if conf.use_emb:
            self.embedding_layer = nn.Embedding.from_pretrained(
                torch.tensor(emb_weights, dtype=torch.float32), padding_idx=0)
            emb_dim = self.embedding_layer.embedding_dim
            # if use embedding, the input dim should be the same as emb_dim
            assert emb_dim == conf.num_inputs, \
                "expect embedding dimension is the same as TCN input dims but get emb:{} and input:{}".format(
                    emb_dim, conf.num_inputs)

        self.tcn = TemporalConvNet(
            num_inputs=conf.num_inputs, num_channels=conf.num_channels,
            kernel_size=conf.kernel_size, dropout=conf.drop_prob)
        self.classifier = nn.Linear(conf.num_inputs, conf.output_dim)

    def forward(self, x, y):
        # x shape: (B, S, F)
        # TODO: embedding here
        # input for TCN should be (B, F, S)
        x = x.transpose(1, 2)
        # apply TCN
        x = self.tcn(x)
        # whether keep all sequence outputs or just last time step
        if self.keep_dim:
            return x.transpose(1, 2)
        else:
            # re-transpose (B, O, S) => (B, S, O)
            x = x.transpose(1, 2)
            # (B, S, O) => (B, O)
            x = x[:, -1, :]
        rep = x.clone()

        x = self.classifier(x)
        pred_prob = torch.softmax(x, dim=-1)
        pred_labels = torch.argmax(x, dim=-1)

        # calc loss
        if self.loss_type is ModelLossMode.BIN:
            # y dim (B, 2)
            loss = nn.functional.binary_cross_entropy_with_logits(x, y)
        elif self.loss_type is ModelLossMode.MUL:
            # y dim (B, 1)
            loss = nn.functional.cross_entropy(x, y)
        else:
            raise NotImplementedError("loss mode only support bin or mul but get {}".format(self.loss_mode.value))

        return loss, pred_prob, pred_labels, rep
