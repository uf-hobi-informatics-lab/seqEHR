import sys

import numpy as np
import torch
from torch import nn

sys.path.append("../")

from sklearn.metrics import accuracy_score, roc_auc_score

from common_utils.config import (EMBEDDING_REDUCTION_MODES, ModelLossMode,
                                 ModelType)
from common_utils.utils import pkl_load
from TCN.tcn import TemporalConvNetEHR, TemporalConvNetEHRConfig

if __name__ == '__main__':
    trsl = pkl_load("../data/tlstm_sync/label_train.pkl")
    trs = pkl_load("../data/tlstm_sync/data_train.pkl")

    tssl = pkl_load("../data/tlstm_sync/label_test.pkl")
    tss = pkl_load("../data/tlstm_sync/data_test.pkl")

    emb_red_mode = EMBEDDING_REDUCTION_MODES["avg"]

    conf = TemporalConvNetEHRConfig(
        input_dim=529, num_tcn_blocks=4, hidden_dim=64, output_dim=2,
        kernel_size=2, drop_prob=0.1, loss_type=ModelLossMode.BIN, use_emb=False,
        reduction_type=emb_red_mode, keep_dim=False)
    model = TemporalConvNetEHR(conf=conf)

    # training config
    lr = 0.001
    epn = 100
    mgn = 2.0

    idxes = list(range(len(trs)))
    tr_loss = .0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    for ep in range(epn):
        np.random.shuffle(idxes)
        for idx in idxes:
            model.zero_grad()
            model.train()

            feature = trs[idx]
            labels = trsl[idx]

            feature_tensor = torch.tensor(feature, dtype=torch.float)
            label_tensor = torch.tensor(labels, dtype=torch.float)

            loss, _, _, _ = model(feature_tensor, label_tensor)

            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("epoch: {}; training loss: {}".format(ep + 1, tr_loss / (ep + 1)))

    # evaluation
    model.eval()
    idxes = list(range(len(tss)))
    y_preds, y_trues, gs_labels, pred_labels = None, None, None, None

    for idx in idxes:
        feature = tss[idx]
        labels = tssl[idx]

        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        with torch.no_grad():
            _, logits, y_pred, _ = model(feature_tensor, label_tensor)

            logits = logits.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()

            if y_preds is None:
                pred_labels = logits
                y_preds = y_pred
                gs_labels = labels
                y_trues = labels[:, 1]
            else:
                pred_labels = np.concatenate([pred_labels, logits], axis=0)
                y_preds = np.concatenate([y_preds, y_pred], axis=0)
                gs_labels = np.concatenate([gs_labels, labels], axis=0)
                y_trues = np.concatenate([y_trues, labels[:, 1]], axis=0)

    total_acc = accuracy_score(y_trues, y_preds)
    total_auc = roc_auc_score(gs_labels, pred_labels, average='micro')
    total_auc_macro = roc_auc_score(gs_labels, pred_labels, average='macro')
    print("Accuracy = {:.3f}".format(total_acc))
    print("AUC = {:.3f}".format(total_auc))
    print("AUC Macro = {:.3f}".format(total_auc_macro))