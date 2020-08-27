import torch
import numpy as np
import argparse
from tqdm import trange
import random

from mix_features.mix_model import MixModelConfig, MixModel
from TLSTM.utils import SeqEHRLogger, pkl_load
from training import SeqEHRTrainer
from data_utils import SeqEHRDataLoader


def main(args):
    # general set up
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(13)

    model_type = args.model
    assert model_type in {'clstm', 'ctlstm'}, \
        "we support: lstm, tlstm, clstm, and ctlstm but get {}".format(model_type)
    conf = "{}.conf".format(model_type)

    # load data
    # if using TLSMT the data have 4 components as non-seq, seq, time elapse, label
    # if using LSTM the data have 3 components as non-seq, seq, label
    # seq data can have different seq length but encoded feature dim must be the same
    # The data should be in format as tuple of list of numpy arrays as [(np.array, np.array, np.array, np.array), ...]
    train_data = pkl_load(args.train_data_path)
    test_data = pkl_load(args.test_data_path)
    # collect input dim for model init (batch, seq, dim)
    args.nonseq_input_dim = train_data[0].shape
    args.seq_input_dim = train_data[1].shape

    # check training data
    # if model_type == "ctlstm":
    #     assert len(train_data[0]) == 4, "Using TLSTM model, the input data must have 4 components."
    # else:
    #     assert len(train_data[0]) in {3, 4}, "Using LSTM model, the input data must have 3 or 4 components"

    train_data_loader = SeqEHRDataLoader(train_data, model_type, task='train').create_data_loader()
    test_data_loader = SeqEHRDataLoader(test_data, model_type, task='test').create_data_loader()

    # init model
    config = MixModelConfig(seq_input_dim=args.seq_input_dim[-1], nonseq_input_dim=args.nonseq_input_dim[-1],
                            dropout_rate=args.dropout_rate, nonseq_hidden_dim=args.nonseq_hidden_dim,
                            seq_hidden_dim=args.seq_hidden_dim, mix_hidden_dim=args.mix_hidden_dim,
                            nonseq_output_dim=args.nonseq_representation_dim, mix_output_dim=args.mix_output_dim,
                            loss_mode=args.loss_mode)
    model = MixModel(config=config, model_type=model_type)
    args.config = config
    args.model = model

    # init task runner
    task_runner = SeqEHRTrainer(args)

    # training
    if args.do_train:
        args.logger.info("start training...")
        task_runner.train(train_data_loader)

    # prediction
    if args.do_test:
        args.logger.info("start test...")
        task_runner.predict(test_data_loader, do_eval=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='clstm', type=str,
                        help="which model used for experiment. We have clstm, and ctlstm")
    parser.add_argument("--train_data_path", default=None, type=str,
                        help="training data dir, should contain a feature, time, and label pickle files")
    parser.add_argument("--test_data_path", default=None, type=str,
                        help="test data dir, should contain a feature, time, and label pickle files")
    parser.add_argument("--model_path", default="./model", type=str, help='where to save the trained model')
    parser.add_argument("--config_path", default=None, type=str, help='where to save the config file')
    parser.add_argument("--log_file", default=None, type=str, help='log file')
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--has_test_label", default=True, type=bool,
                        help="If the test data have the ground truth labels")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help='values [-mgn, mgn] to clip gradient')
    parser.add_argument("--weight_decay", default=0.0, type=float, help='weight decay used in AdamW')
    parser.add_argument("--eps", default=1e-8, type=float, help='eps for AdamW')
    parser.add_argument("--learning_rate", default=1e-3, type=float, help='learning_rate')
    parser.add_argument("--dropout_rate", default=0.1, type=float, help='drop probability')
    parser.add_argument("--train_epochs", default=50, type=int, help='number of epochs for training')
    parser.add_argument("--nonseq_hidden_dim", default=128, type=int, help='MLP hidden layer size')
    parser.add_argument("--seq_hidden_dim", default=128, type=int, help='LSTM or TLSTM hidden layer size')
    parser.add_argument("--mix_hidden_dim", default=64, type=int, help='fully connected layer size for mix model')
    parser.add_argument("--nonseq_representation_dim", default=64, type=int,
                        help='representation dim for nonseq features')
    parser.add_argument("--mix_output_dim", default=64, type=int, help='mix model output dim')
    parser.add_argument("--loss_mode", default='bin', type=str,
                        help='using "bin" for Softmax+BCELoss or "clz" for CrossEntropyLoss')
    # TODO: ensemble two TLSTM for handling different data encoding format
    parser.add_argument("--do_ensemble", default=0, type=int,
                        help='whether use ensemble model to process OHE + numeric')
    # TODO: enable mix-percision training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger = SeqEHRLogger(logger_file=args.log_file, logger_level='i').get_logger()
    if args.config_path is None:
        args.config_path = args.model_path
    main(args)
