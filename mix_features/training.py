import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, auc, roc_curve


class SeqEHRTrainer(object):

    def __init__(self, args):
        super(SeqEHRTrainer, self).__init__()
        self.args = args

    def train(self, train_data_loader):
        pass

    def predict(self, test_data_loader, do_eval=True):
        """
        :param test_data_loader: testing data
        :param do_eval: if true try to run evaluation (GS must be provided)
        """
        pass

    def _eval(self):
        pass

    @staticmethod
    def _get_auc(self, yt, yp):
        """
        :param yt: true labels as numpy array of [[0, 1], [1,0]]
        :param yp: predicted labels as numpy array of [[0.2, 0.8], [0.8, 0.2]]
        :return: auc score, sensitivity, specificity, J-index
        """
        assert yt.shape[-1] == yp.shape[-1] == 2, \
            "expected shape of (?,2) " \
            "but get shape of true labels as {} and predict labels shape as {}".format(yt.shape, yp.shape)
        auc_score_1 = roc_auc_score(average='micro')
        # we only need the trues or probs for positive case (label=1)
        yt, yp = yt[:, 1], yp[:, 1]
        fpr, tpr, th = roc_curve(yt, yp)
        auc_score = auc(fpr, tpr)
        # get specificity and sensitivity
        opt_idx = np.argmax(tpr - fpr)
        sensitivity, specificity = tpr[opt_idx], 1 - fpr[opt_idx]
        J_idx = th[opt_idx]

        return auc_score, auc_score_1, sensitivity, specificity, J_idx

    @staticmethod
    def _get_acc(self, yt, yp):
        return accuracy_score(yt, yp)




def _eval(model, features, times, labels):
    model.eval()
    y_preds, y_trues, gs_labels, pred_labels = None, None, None, None
    check_inputs(features, times, labels)

    data_idxs = list(range(len(features)))
    for data_idx in data_idxs:
        # prepare data
        feature = features[data_idx]
        time = times[data_idx]
        label = labels[data_idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32).to(args.device)
        time_tensor = torch.tensor(time, dtype=torch.float32).to(args.device)
        label_tensor = torch.tensor(label, dtype=torch.float32).to(args.device)

        with torch.no_grad():
            _, logits, y_pred = model(feature_tensor, time_tensor, label_tensor)
            logits = torch.nn.functional.softmax(logits).detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            if y_preds is None:
                pred_labels = logits
                y_preds = y_pred
                gs_labels = label
                y_trues = np.argmax(label, axis=1)
            else:
                pred_labels = np.concatenate([pred_labels, logits], axis=0)
                y_preds = np.concatenate([y_preds, y_pred], axis=0)
                gs_labels = np.concatenate([gs_labels, label], axis=0)
                y_trues = np.concatenate([y_trues, np.argmax(label, axis=1)], axis=0)
        return y_trues, y_preds, gs_labels, pred_labels


def check_inputs(*inputs):
    llen = []
    for each in inputs:
        llen.append(len(each))
    assert len(set(llen)) == 1, \
        """input datas must have same amount of data point but 
        get dims as {}
        """.format(llen)


def train(args, model, features, times, labels):
    check_inputs(features, times, labels)
    data_idxs = list(range(len(features)))
    # optimizer set up
    # # use adam to follow the original implementation
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # # using AdamW for better generalizability
    # no_decay = {'', '', '', ''}
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
    epoch_iter = trange(int(args.train_epochs), desc="Epoch", disable=True)
    model.zero_grad()
    for epoch in epoch_iter:
        # shuffle training data
        np.random.shuffle(data_idxs)
        for data_idx in data_idxs:
            # prepare data
            feature = features[data_idx]
            time = times[data_idx]
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
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            tr_loss += loss.item()
        args.logger.info("epoch: {}; training loss: {}".format(epoch + 1, tr_loss / (epoch + 1)))
