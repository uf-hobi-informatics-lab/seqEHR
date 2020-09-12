"""
To generate training and test data, we must include the label
If in test we do not have true labels (may need to evaluate later),
    then we need to generate fake labels.
In this case, evaluate loss will be meaningless.

functions:
  1. load and save data (checked)
  2. merge non-seq and seq data (checked)
  3. convert data to tensor (checked)
  4. prepare 5-CV (todo)
"""
import torch
from torch import tensor, float32
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from config import ModelType


class SeqEHRDataLoader:

    def __init__(self, data, model_type, task='train'):
        # TODO switch to pad_packed_seq and pack_padded_seq then we can use batch size
        self.batch_size = 1
        self.data = data
        self.task = task
        self.model_type = model_type

    def __create_tensor_dataset(self):
        nonseq, seq, label = [], [], []

        for each in self.data:
            nonseq.append(each[0])
            seq.append(torch.FloatTensor(each[1]))
            print(seq[0].shape)
            label.append(each[2])
        padded_sequence = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)

        return TensorDataset(
            tensor(nonseq, dtype=float32),
            tensor(padded_sequence, dtype=float32),
            tensor(label, dtype=float32)
        )

    def __create_tensor_dataset_with_time(self):
        nonseq, seq, time, label = [], [], [], []

        for each in self.data:
            nonseq.append(each[0])
            seq.append(each[1])
            time.append(each[2])
            label.append(each[3])

        return TensorDataset(
            tensor(nonseq, dtype=float32),
            tensor(seq, dtype=float32),
            tensor(time, dtype=float32),
            tensor(label, dtype=float32)
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