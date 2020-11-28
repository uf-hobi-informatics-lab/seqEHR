import numpy as np
import torch
from torch import nn
import sys
sys.path.append("../")

from common_utils.utils import pkl_load
from common_utils.config import ModelType, ModelLossMode, EmbeddingReductionMode
from Embeddings.embedding_models import SeqEmbEHR, SeqEmbEHRConfig


def ohe2idx(data):
    # convert OHE to index np.array(0,1,0,1) => [1, 3]
    uniques = set()
    nd = []
    for each in data:
        d1 = []
        for e1 in each:
            d2 = []
            for e2 in e1:
                idxs = list(np.where(e2 == 1)[0])
                for i in idxs:
                    uniques.add(i)
                d2.append(idxs)
            d1.append(np.array(d2))
        nd.append(np.array(d1))
    return nd, uniques


def random_generate_embeddings(vocab, emb_dim=50):
    """
    The function is used to create a random initialized embeddings based on a pre-defined vocab
    :param vocab: a list of medical codes (ICD or RXCUI)
    :return:
    """
    vocab = sorted(list(set(vocab)))

    code2index = dict()
    code2index['pad'] = 0
    code2index['unk'] = len(vocab) + 1

    embeddings = np.zeros(emb_dim).reshape(1, -1)

    for idx, code in enumerate(vocab):
        code2index[code] = idx + 1

    np.random.seed(2)
    embeddings = np.concatenate([embeddings, np.random.rand(len(vocab)+1, emb_dim)], axis=0)

    index2code = {v: k for k, v in code2index.items()}

    return embeddings, code2index, index2code


if __name__ == '__main__':
    trs = pkl_load("../data/tlstm_sync/data_train.pkl")
    ttrs = pkl_load("../data/tlstm_sync/elapsed_train.pkl")
    trsl = pkl_load("../data/tlstm_sync/label_train.pkl")
    ntrs, s1 = ohe2idx(trs)

    tss = pkl_load("../data/tlstm_sync/data_test.pkl")
    ttss = pkl_load("../data/tlstm_sync/elapsed_test.pkl")
    tssl = pkl_load("../data/tlstm_sync/label_test.pkl")
    ntss, s2 = ohe2idx(tss)

    # create a embedding with dim as 10
    emb, c2i, i2c = random_generate_embeddings(s1.union(s2), 10)

    conf = SeqEmbEHRConfig(
        input_dim=10, output_dim=2, hidden_dim=64, emb_dim=10, drop_prob=0.1,
        model_type=ModelType.M_TLSTM, loss_type=ModelLossMode.BIN, merge_type=EmbeddingReductionMode.SUM)
    model = SeqEmbEHR(config=conf, emb_weights=emb)

    lr = 0.001
    epn = 50
    mgn = 2.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    idxes = list(range(len(ntrs)))
    tr_loss = .0

    # ### Training
    for ep in range(epn):
        np.random.shuffle(idxes)
        for idx in idxes:
            model.zero_grad()
            model.train()

            feature = ntrs[idx]
            labels = trsl[idx]
            time = ttrs[idx]
            time = np.reshape(time, [time.shape[0], time.shape[2], time.shape[1]])

            feature_tensor = torch.tensor(feature, dtype=torch.long)
            time_tensor = torch.tensor(time, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.float32)

            loss, _, _, _ = model(feature_tensor, label_tensor, time_tensor)
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("epoch: {}; training loss: {}".format(ep + 1, tr_loss / (ep + 1)))


    # ### evaluation
    model.eval()

    idxes = list(range(len(ntss)))
    y_preds, y_trues, gs_labels, pred_labels = None, None, None, None

    for idx in idxes:
        feature = ntss[idx]
        labels = tssl[idx]
        time = ttss[idx]
        time = np.reshape(time, [time.shape[0], time.shape[2], time.shape[1]])

        feature_tensor = torch.tensor(feature, dtype=torch.long)
        time_tensor = torch.tensor(time, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        with torch.no_grad():
            _, logits, y_pred, _ = model(feature_tensor, label_tensor, time_tensor)

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

    from sklearn.metrics import roc_auc_score, accuracy_score
    total_acc = accuracy_score(y_trues, y_preds)
    total_auc = roc_auc_score(gs_labels, pred_labels, average='micro')
    total_auc_macro = roc_auc_score(gs_labels, pred_labels, average='macro')
    print("Accuracy = {:.3f}".format(total_acc))
    print("AUC = {:.3f}".format(total_auc))
    print("AUC Macro = {:.3f}".format(total_auc_macro))