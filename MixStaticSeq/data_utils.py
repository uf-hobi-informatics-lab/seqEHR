"""
To generate training and test data, we must include the label
If in test we do not have true labels (may need to evaluate later),
    then we need to generate fake labels.
In this case, evaluate loss will be meaningless.

In case patients have different length of observations, we have to fix different length of seq data (dim 0)
Our proposal: padding seq for making tensor then add collate_fn to remove all paddings

We may need to fix batch size at 1; need test
"""

import sys
sys.path.append("../")


import torch
from torch import float32, long, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from common_utils.config import ModelLossMode, ModelType, UNIVERSE_PAD


def remove_paddings(seq_data):
    new_seq_data = []
    for data in seq_data:
        if data[0] == UNIVERSE_PAD:
            continue
        new_seq_data.append(data.numpy())
    return new_seq_data


def collate_fn(batch):
    # we know batch size is 1
    dim = len(batch[0])

    if dim == 3:
        # this is for no time-aware
        non_seq, seq, label = batch[0]
        new_seq = remove_paddings(seq)
        return torch.unsqueeze(non_seq, 0), torch.unsqueeze(tensor(new_seq), 0), torch.unsqueeze(label, 0)
    elif dim == 4:
        # this is for time-aware
        non_seq, seq, time, label = batch[0]
        new_seq = remove_paddings(seq)
        new_time = remove_paddings(time)
        return torch.unsqueeze(non_seq, 0), torch.unsqueeze(tensor(new_seq), 0), \
            torch.unsqueeze(tensor(new_time), 0), torch.unsqueeze(label, 0)
    else:
        raise RuntimeError("expect batch has 3 or 4 dim but get {}".format(dim))


class SeqEHRDataLoader:

    def __init__(self, data, model_type, loss_mode, batch_size, task='train', various_seq_len=False):
        self.data = data
        self.task = task
        self.model_type = model_type
        self.batch_size = batch_size
        self.loss_mode = loss_mode
        self.various_seq_len = various_seq_len

    def __create_tensor_dataset(self):
        nonseq, seq, label = [], [], []

        for each in self.data:
            nonseq.append(each[0])
            seq.append(each[1])
            label.append(each[2])

        if self.loss_mode is ModelLossMode.BIN:
            return TensorDataset(
                tensor(nonseq, dtype=float32),
                # tensor(seq, dtype=float32),
                pad_sequence([tensor(s, dtype=float32) for s in seq],
                             batch_first=True,
                             padding_value=UNIVERSE_PAD
                             ),
                tensor(label, dtype=float32)
            )
        else:
            return TensorDataset(
                tensor(nonseq, dtype=float32),
                # tensor(seq, dtype=float32),
                pad_sequence([tensor(s, dtype=float32) for s in seq],
                             batch_first=True,
                             padding_value=UNIVERSE_PAD
                             ),
                tensor(label, dtype=long)
            )

    def __create_tensor_dataset_with_time(self):
        nonseq, seq, time, label = [], [], [], []

        for each in self.data:
            nonseq.append(each[0])
            seq.append(each[1])
            time.append(each[2])
            label.append(each[3])

        if self.loss_mode is ModelLossMode.BIN:
            return TensorDataset(
                tensor(nonseq, dtype=float32),
                # tensor(seq, dtype=float32),
                # tensor(time, dtype=float32),
                pad_sequence([tensor(s, dtype=float32) for s in seq],
                             batch_first=True,
                             padding_value=UNIVERSE_PAD
                             ),
                pad_sequence([tensor(t, dtype=float32) for t in time],
                             batch_first=True,
                             padding_value=UNIVERSE_PAD
                             ),
                tensor(label, dtype=float32)
            )
        else:
            return TensorDataset(
                tensor(nonseq, dtype=float32),
                # tensor(seq, dtype=float32),
                # tensor(time, dtype=float32),
                pad_sequence([tensor(s, dtype=float32) for s in seq],
                             batch_first=True,
                             padding_value=UNIVERSE_PAD
                             ),
                pad_sequence([tensor(t, dtype=float32) for t in time],
                             batch_first=True,
                             padding_value=UNIVERSE_PAD
                             ),
                tensor(label, dtype=long)
            )

    def create_data_loader(self):
        if self.model_type is ModelType.M_TLSTM:
            dataset = self.__create_tensor_dataset_with_time()
        else:
            dataset = self.__create_tensor_dataset()

        if self.task == 'train':
            sampler = RandomSampler(dataset)
        elif self.task == 'test':
            sampler = SequentialSampler(dataset)
        else:
            raise ValueError('task argument only support train or test but get {}'.format(self.task))
        if self.various_seq_len:
            return DataLoader(dataset, sampler=sampler, batch_size=1, pin_memory=True, collate_fn=collate_fn)
        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, pin_memory=True)
