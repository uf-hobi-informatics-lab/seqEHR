from tcn import TemporalConvNet
import torch
from torch import nn
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle
import sys
import os
from pathlib import Path
from tqdm import trange


def pkl_load(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


def pkl_save(data, fn):
    with open(fn, "wb") as f:
        pickle.dump(data, f)


class Args:
    def __init__(self):
        self.ksize = 3
        self.epochs = 50
        self.levels = 4
        self.lr = 0.001
        self.hidden_dim = 128

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            s += "{}={}\n".format(k, v)
        return s


class TestTCNModel(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs, labels):
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L) - L is time seq; C is features
        o = self.linear(y1[:, :, -1])
        loss = nn.functional.binary_cross_entropy_with_logits(labels, o)
        return loss, o, torch.argmax(o, dim=1)


def _eval(model, features, times, labels):
    model.eval()

    assert len(features) == len(times) == len(labels), \
        """input data and labels must have same amount of data point but 
        get num of features: {};
        get num of times: {};
        get num of labels: {}.
        """.format(len(features), len(times), len(labels))
    data_idxs = list(range(len(features)))

    y_preds, y_trues, gs_labels, pred_labels = None, None, None, None

    for data_idx in data_idxs:
        # prepare data
        feature = features[data_idx]
        time = times[data_idx]
        time = np.reshape(time, [time.shape[0], time.shape[2], time.shape[1]])
        label = labels[data_idx]
        # to tensor on device
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        time_tensor = torch.tensor(time, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        with torch.no_grad():
            _, logits, y_pred = model(feature_tensor, time_tensor, label_tensor)
            logits = logits.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            if y_preds is None:
                pred_labels = logits
                y_preds = y_pred
                gs_labels = label
                y_trues = label[1]
            else:
                pred_labels = np.concatenate([pred_labels, logits], axis=0)
                y_preds = np.concatenate([y_preds, y_pred], axis=0)
                gs_labels = np.concatenate([gs_labels, label], axis=0)
                y_trues = np.concatenate([y_trues, label[1]], axis=0)

    total_acc = accuracy_score(y_trues, y_preds)
    total_auc = roc_auc_score(gs_labels, pred_labels, average='micro')
    total_auc_macro = roc_auc_score(gs_labels, pred_labels, average='macro')
    print("Train Accuracy = {:.3f}".format(total_acc))
    print("Train AUC = {:.3f}".format(total_auc))
    print("Train AUC Macro = {:.3f}".format(total_auc_macro))


def train(args, model, features, times, labels):
    assert len(features) == len(times) == len(labels), \
        """input data and labels must have same amount of data point but 
        get num of features: {};
        get num of times: {};
        get num of labels: {}.
        """.format(len(features), len(times), len(labels))
    data_idxs = list(range(len(features)))

    # optimizer set up
    # # use adam to follow the original implementation
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # # using AdamW for better generalizability
    # no_decay = {'bias', 'norm'}
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.eps)

    # using fp16 for training rely on Nvidia apex package
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # training loop
    tr_loss = .0
    epoch_iter = trange(int(args.train_epochs), desc="Epoch")
    model.zero_grad()
    for epoch in epoch_iter:
        # shuffle training data
        # np.random.shuffle(data_idxs)
        for data_idx in data_idxs:
            # prepare data
            feature = features[data_idx]
            time = times[data_idx]
            time = np.reshape(time, [time.shape[0], time.shape[2], time.shape[1]])
            label = labels[data_idx]
            # to tensor on device
            feature_tensor = torch.tensor(feature, dtype=torch.float32).to(args.device)
            time_tensor = torch.tensor(time, dtype=torch.float32).to(args.device)
            label_tensor = torch.tensor(label, dtype=torch.float32).to(args.device)

            # training
            model.train()
            loss, _, _ = model(feature_tensor, time_tensor, label_tensor)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            model.zero_grad()
            tr_loss += loss.item()
        args.logger.info("epoch: {}; training loss: {}".format(epoch+1, tr_loss/(epoch+1)))

    _eval(model, features, times, labels)


def test(args, model, features, times, labels):
    assert len(features) == len(times) == len(labels), \
        """input data and labels must have same amount of data point but 
        get num of features: {};
        get num of times: {};
        get num of labels: {}.
        """.format(len(features), len(times), len(labels))
    data_idxs = list(range(len(features)))
    _eval(model, features, times, labels)


def main(args):
    # general set up
    torch.manual_seed(13)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(13)

    train_data = pkl_load("../data/tlstm_sync/data_train.pkl")
    train_elapsed_data = pkl_load("../data/tlstm_sync/elapsed_train.pkl")
    train_labels = pkl_load("../data/tlstm_sync/label_train.pkl")
    # init config
    input_dim = train_data[0].shape[2]
    output_dim = train_labels[0].shape[1]

    # init TLSTM model
    model = TestTCNModel()
    # training
    train(args, model, train_data, train_elapsed_data, train_labels)

    test_data = pkl_load("../data/tlstm_sync/data_test.pkl")
    test_elapsed_data = pkl_load("../data/tlstm_sync/elapsed_test.pkl")
    test_labels = pkl_load("../data/tlstm_sync/label_test.pkl")

    test(args, model, test_data, test_elapsed_data, test_labels)


if __name__ == '__main__':
    main()