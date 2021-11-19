"""
To generate training and test data, we must include the label
If in test we do not have true labels (may need to evaluate later),
    then we need to generate fake labels.
In this case, evaluate loss will be meaningless.

functions:
  1. load and save data (checked)
  2. merge non-seq and seq data (checked)
  3. convert data to tensor (checked)
"""

import sys

from torch import float32, long, tensor
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from common_utils.config import ModelLossMode, ModelType

sys.path.append("../")


class SeqEHRDataLoader:

    def __init__(self, data, model_type, loss_mode, batch_size, task='train'):
        self.data = data
        self.task = task
        self.model_type = model_type
        self.batch_size = batch_size
        self.loss_mode = loss_mode

    def __create_tensor_dataset(self):
        nonseq, seq, label = [], [], []

        for each in self.data:
            nonseq.append(each[0])
            seq.append(each[1])
            label.append(each[2])

        if self.loss_mode is ModelLossMode.BIN:
            return TensorDataset(
                tensor(nonseq, dtype=float32),
                tensor(seq, dtype=float32),
                tensor(label, dtype=float32)
            )
        else:
            return TensorDataset(
                tensor(nonseq, dtype=float32),
                tensor(seq, dtype=float32),
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
                tensor(seq, dtype=float32),
                tensor(time, dtype=float32),
                tensor(label, dtype=float32)
            )
        else:
            return TensorDataset(
                tensor(nonseq, dtype=float32),
                tensor(seq, dtype=float32),
                tensor(time, dtype=float32),
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

        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, pin_memory=True)
