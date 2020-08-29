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

from torch import tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


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
            seq.append(each[1])
            label.append(each[2])

        return TensorDataset(tensor(nonseq), tensor(seq), tensor(label))

    def __create_tensor_dataset_with_time(self):
        nonseq, seq, time, label = [], [], [], []

        for each in self.data:
            nonseq.append(each[0])
            seq.append(each[1])
            time.append(each[2])
            label.append(each[3])

        return TensorDataset(tensor(nonseq), tensor(seq), tensor(time), tensor(label))

    def create_data_loader(self):
        if self.model_type == "ctlstm":
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