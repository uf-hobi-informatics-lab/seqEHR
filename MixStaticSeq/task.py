import torch
import numpy as np
import argparse 
import random
from training import SeqEHRTrainer
from data_utils import SeqEHRDataLoader
from common_utils.config import MODEL_TYPE_FLAGS, MODEL_LOSS_MODES

import sys
sys.path.append("../")
from common_utils.utils import SeqEHRLogger, pkl_load


def main(args):
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        args.model_type = MODEL_TYPE_FLAGS[args.model_type]
    except ValueError:
        raise RuntimeError("we support: lstm, tlstm but get {}".format(args.model_type))

    try:
        args.loss_mode = MODEL_LOSS_MODES[args.loss_mode]
    except ValueError:
        raise RuntimeError("we support: lstm, tlstm but get {}".format(args.loss_mode))

    # load data
    # if using TLSMT the data have 4 components as non-seq, seq, time elapse, label
    # if using LSTM the data have 3 components as non-seq, seq, label
    # seq data can have different seq length (encounter numbers) but encoded feature dim must be the same
    # The data should be in format as tuple of list of numpy arrays as [(np.array, np.array, np.array, np.array), ...]
    train_data_loader = None
    if args.do_train:
        train_data = pkl_load(args.train_data_path)
        train_data_loader = SeqEHRDataLoader(
            train_data, args.model_type, args.loss_mode, args.batch_size, task='train').create_data_loader()
        args.total_step = len(train_data_loader)
        # collect input dim for model init (seq, dim)
        args.nonseq_input_dim = train_data[0][0].shape
        args.seq_input_dim = train_data[0][1].shape

        if args.sampling_weight:
            # the data should be a 1-D numpy array of  1/ratio of each class
            # (class with more samples should have low weights)
            args.sampling_weight = pkl_load(args.sampling_weight)
            args.logger.info("using sample weights as {}".format(args.sampling_weight))

    test_data_loader = None
    if args.do_test:
        test_data = pkl_load(args.test_data_path)
        # create data loader (pin_memory is set to True) -> (B, S, T)
        test_data_loader = SeqEHRDataLoader(
            test_data, args.model_type, args.loss_mode, args.batch_size, task='test').create_data_loader()

    # init task runner
    task_runner = SeqEHRTrainer(args)

    # training
    if args.do_train:
        args.logger.info("start training...")
        task_runner.train(train_data_loader)

    # prediction
    if args.do_test:
        args.logger.info("start test...")
        task_runner.predict(test_data_loader, do_eval=args.do_eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='clstm', type=str,
                        help="which model used for experiment. We have clstm, and ctlstm")
    parser.add_argument("--train_data_path", default=None, type=str,
                        help="training data dir, should contain a feature, time, and label pickle files")
    parser.add_argument("--test_data_path", default=None, type=str,
                        help="test data dir, should contain a feature, time, and label pickle files")
    parser.add_argument("--sampling_weight", default=None, type=str,
                        help="pickle file contain the weights for each sampling based on their distributions (opt)")
    parser.add_argument("--new_model_path", default="./model", type=str, help='where to save the trained model')
    parser.add_argument("--log_file", default=None, type=str, help='log file')
    parser.add_argument("--result_path", default=None, type=str,
                        help='path to save raw and evaluation results; if none, report only evaluations by log')
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="""Whether to run evaluation on test. 
                        Using this flag, the labels for the test set must be real labels""")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--do_warmup", action='store_true',
                        help="Whether to use learning rate warm up strategy")
    parser.add_argument("--optim", default="adam", type=str, help='the optimizer used for training')
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help='values [-mgn, mgn] to clip gradient')
    parser.add_argument("--weight_decay", default=0.0, type=float, help='weight decay used in AdamW')
    parser.add_argument("--eps", default=1e-8, type=float, help='eps for AdamW')
    parser.add_argument("--learning_rate", default=1e-3, type=float, help='learning_rate')
    parser.add_argument("--dropout_rate", default=0.1, type=float, help='drop probability')
    parser.add_argument("--train_epochs", default=50, type=int, help='number of epochs for training')
    parser.add_argument("--seed", default=13, type=int, help='random seed')
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help='percentage of warm up steps in the total steps per epoch (must be in [0, 1)')
    parser.add_argument("--nonseq_hidden_dim", default=128, type=int, help='MLP hidden layer size')
    parser.add_argument("--seq_hidden_dim", default=128, type=int, help='LSTM or TLSTM hidden layer size')
    parser.add_argument("--mix_hidden_dim", default=64, type=int, help='fully connected layer size for mix model')
    parser.add_argument("--nonseq_representation_dim", default=64, type=int,
                        help='representation dim for nonseq features')
    parser.add_argument("--log_gradients", action='store_true',
                        help="Whether to log intermediate gradients with loss")
    parser.add_argument("--log_step", default=-1, type=int,
                        help='steps before logging after run training. If -1, log every epoch')
    parser.add_argument("--mix_output_dim", default=2, type=int, help='mix model output dim')
    parser.add_argument("--batch_size", default=1, type=int, help='how many patients data we feed in each iteration')
    parser.add_argument("--loss_mode", default='bin', type=str,
                        help='using "bin" for Softmax+BCELoss or "mul" for CrossEntropyLoss')
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision (PyTorch 1.6 naive implementation)")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger = SeqEHRLogger(logger_file=args.log_file, logger_level='i').get_logger()
    main(args)
