import sys
sys.path.append("../")


from pathlib import Path

import numpy as np
import torch

from seq_ehr_model import MixModel, MixModelConfig
from sklearn.metrics import (accuracy_score, auc,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from tqdm import tqdm, trange

from common_utils.config import ModelLossMode, ModelOptimizers
from common_utils.utils import pkl_load, pkl_save


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class SeqEHRTrainer(object):

    def __init__(self, args):
        super(SeqEHRTrainer, self).__init__()
        self.args = args
        if self.args.do_train:
            self._init_new_model()
        else:
            self._load_model()
        self.args.logger.info("Config Information:")
        self.args.logger.info(self.config)

    def train(self, train_data_loader):
        tr_loss = .0
        global_step = 0

        epoch_iter = trange(int(self.args.train_epochs), desc="Epoch", disable=False)
        for epoch in epoch_iter:
            batch_iter = tqdm(iterable=train_data_loader, desc='Batch', disable=False)
            for step, batch in enumerate(batch_iter):
                self.model.train()
                self.model.zero_grad()
                # load batch to GPU or CPU
                batch = tuple(b.to(self.args.device) for b in batch)
                # the last element is label
                if self.args.fp16:
                    with self.autocast:
                        loss, _, _, _ = self.model(batch)
                else:
                    loss, _, _, _ = self.model(batch)

                if self.args.fp16:
                    loss = self.scaler.scale(loss)
                    loss.backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                if self.args.do_warmup:
                    self.scheduler.step()

                tr_loss += loss.item()
                global_step += 1

                if self.args.log_step > 0 and (step + 1) % self.args.log_step == 0:
                    self.args.logger.info("epoch: {}; step: {}; current loss: {:.4f}; "
                                          "total loss: {:.4f}; average loss: {:.4f}"
                                          .format((epoch+1), global_step, loss, tr_loss, (tr_loss/global_step)))
                    if self.args.log_gradients:
                        for name, parms in self.model.named_parameters():
                            self.args.logger.info(
                                '-->name:', name,
                                '-->grad_requirs:', parms.requires_grad,
                                '--weight', torch.mean(parms.data),
                                '-->grad_value:', torch.mean(parms.grad))

            if self.args.log_step == -1:
                self.args.logger.info("epoch: {}; total loss: {:.4f}; average loss: {:.4f}"
                                      .format((epoch+1), tr_loss, (tr_loss/global_step)))
                if self.args.log_gradients:
                    for name, parms in self.model.named_parameters():
                        self.args.logger.info(
                            '-->name:', name,
                            '-->grad_requirs:', parms.requires_grad,
                            '--weight', torch.mean(parms.data),
                            '-->grad_value:', torch.mean(parms.grad))

        # save model and config
        self._save_model()

    def predict(self, test_data_loader, do_eval=True):
        """
        :param test_data_loader: testing data
        :param do_eval: if true try to run evaluation (GS must be provided)
        """
        batch_iter = tqdm(iterable=test_data_loader, desc='Batch', disable=False)
        yt_probs, yp_probs, yt_tags, yp_tags, eval_loss, representations = self._eval(batch_iter)

        res_path = None
        if self.args.result_path:
            self.args.logger.info("Results are reported in {}".format(self.args.result_path))
            res_path = Path(self.args.result_path)
            res_path.mkdir(parents=True, exist_ok=True)
            raw_res_fn = res_path / "raw_results.tsv"

            with open(raw_res_fn, "w") as f:
                header = "\t".join(
                    ["\t".join([str(i) for i in range(len(yp_probs[0]))]), "predict_label", "true_label"])
                f.write(header + "\n")
                for each in zip(yp_probs, yp_tags, yt_tags):
                    probs = "\t".join([str(e) for e in each[0]])
                    line = "\t".join([probs, str(each[1]), str(each[2])]) + "\n"
                    f.write(line)

        if do_eval:
            if self.args.loss_mode is ModelLossMode.BIN:
                # BIN use acc and ROC-AUC
                acc = self._get_acc(yt=yt_tags, yp=yp_tags)
                auc_score, auc_score_1, sensitivity, specificity, J_idx = self._get_auc(yt=yt_probs, yp=yp_probs)
                eval_res = "accuracy:{:.4f}\nauc_score:{:.4f}\nsensitivity:{:.4f}\nspecificity:{:.4f}\nJ_index:{}\n"\
                    .format(acc, auc_score, sensitivity, specificity, J_idx)
            else:
                # ModelLossMode.MUL use acc and PRF
                acc = self._get_acc(yt=yt_tags, yp=yp_tags)
                pre, rec, f1 = self._get_prf(yt=yt_tags, yp=yp_tags)
                eval_res = "accuracy:{:.4f}\nprecision:{:.4f}\nrecall:{:.4f}\nF1-micro:{:.4f}\n"\
                    .format(acc, pre, rec, f1)

                try:
                    auc_score, auc_score_1, sensitivity, specificity, J_idx = self._get_auc(yt=yt_probs, yp=yp_probs)
                    eval_res += \
                        "accuracy:{:.4f}\nauc_score:{:.4f}\nsensitivity:{:.4f}\nspecificity:{:.4f}\nJ_index:{}\n" \
                        .format(acc, auc_score, sensitivity, specificity, J_idx)
                except Exception:
                    pass

            if res_path:
                eval_metric_fn = res_path / "evaluation.txt"
                with open(eval_metric_fn, "w") as f:
                    f.write(eval_res)

            # log all the evaluations
            self.args.logger.info("evaluation results on test:\n{}".format(eval_res))

    def _save_model(self):
        root = Path(self.args.new_model_path)
        root.mkdir(parents=True, exist_ok=True)
        new_model_bin_file = root / "pytorch_model.{}.bin".format(self.args.model_type.value)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), new_model_bin_file)

        new_model_config_file = Path(self.args.new_model_path) / "config.{}.pkl".format(self.args.model_type.value)
        pkl_save(self.config, new_model_config_file)

    def _load_model(self):
        config_path = Path(self.args.new_model_path) / "config.{}.pkl".format(self.args.model_type.value)
        self.config = pkl_load(config_path)

        self.model = MixModel(config=self.config, model_type=self.args.model_type)
        model_path = Path(self.args.new_model_path) / "pytorch_model.{}.bin".format(self.args.model_type.value)
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))  # using cpu to load model
        self.model.load_state_dict(ckpt)
        self.model.to(self.args.device)

    def _init_new_model(self):
        # init model
        self.config = MixModelConfig(seq_input_dim=self.args.seq_input_dim,
                                     nonseq_input_dim=self.args.nonseq_input_dim,
                                     dropout_rate=self.args.dropout_rate,
                                     nonseq_hidden_dim=self.args.nonseq_hidden_dim,
                                     seq_hidden_dim=self.args.seq_hidden_dim,
                                     mix_hidden_dim=self.args.mix_hidden_dim,
                                     nonseq_output_dim=self.args.nonseq_representation_dim,
                                     mix_output_dim=self.args.mix_output_dim,
                                     loss_mode=self.args.loss_mode,
                                     mlp_num=self.args.mlp_num)
        self.config.sampling_weight = self.args.sampling_weight
        self.model = MixModel(config=self.config, model_type=self.args.model_type)
        self.model.to(self.args.device)

        # set up optimizer
        if self.args.optim == ModelOptimizers.ADAM.value:
            # using AdamW implementation
            # may have problem with batch=1 since it is more fit for mini-batch update
            no_decay = {'bias'}
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, amsgrad=False)
        elif self.args.optim == ModelOptimizers.SGD.value:
            # using momentum SGD implementation and default momentum is set to 0.9 and use nesterov
            # high variance of the parameter updates; less stable convergence; but work with batch=1
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, nesterov=True)
        else:
            # if optim option is not properly set, default using adam
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, amsgrad=False)
        self.args.logger.info("The optimizer detail:\n {}".format(self.optimizer))

        # set up optimizer warm up scheduler (you can set warmup_ratio=0 to deactivated this function)
        if self.args.do_warmup:
            t_total = self.args.total_step // self.args.train_epochs
            warmup_steps = np.dtype('int64').type(self.args.warmup_ratio * t_total)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        # mix precision training
        self.scaler = None
        self.autocast = None
        if self.args.fp16:
            try:
                self.autocast = torch.cuda.amp.autocast()
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception:
                raise ImportError("You need to update to PyTorch 1.6, the current PyTorch version is {}"
                                  .format(torch.__version__))

    def _eval(self, batch_iter):
        self.model.eval()
        eval_loss = 0.
        global_step = 0
        yt_probs, yp_probs, yt_tags, yp_tags, reps = None, None, None, None, None
        for step, batch in enumerate(batch_iter):
            batch = tuple(b.to(self.args.device) for b in batch)
            with torch.no_grad():
                loss, pred_probs, pred_tags, rep = self.model(batch)
                eval_loss += loss.item()
                global_step += 1

                pred_probs = pred_probs.detach().cpu().numpy()  # to cpu as np array
                pred_tags = pred_tags.detach().cpu().numpy()
                rep = rep.detach().cpu().numpy()
                if self.args.loss_mode is ModelLossMode.BIN:
                    true_probs = batch[-1].detach().cpu().numpy()
                    true_tags = np.argmax(true_probs, axis=-1)
                else:
                    true_probs = self._covert_single_label_to_ohe_label(batch[-1].detach().cpu().numpy())
                    true_tags = batch[-1].detach().cpu().numpy()

                if yt_probs is None:
                    yt_probs = true_probs
                    yp_probs = pred_probs
                    yt_tags = true_tags
                    yp_tags = pred_tags
                    reps = rep
                else:
                    yp_probs = np.vstack([yp_probs, pred_probs])
                    yp_tags = np.vstack([yp_tags, pred_tags])
                    yt_probs = np.vstack([yt_probs, true_probs])
                    yt_tags = np.vstack([yt_tags, true_tags])
                    reps = np.vstack([reps, rep])

        return yt_probs, yp_probs, yt_tags, yp_tags, eval_loss/global_step, reps

    @staticmethod
    def _covert_single_label_to_ohe_label(label):
        metrix = np.ones((len(label), 2), dtype=float)
        for idx, each in enumerate(label):
            metrix[idx][each] = 1.0
        return metrix

    @staticmethod
    def _get_auc(yt, yp):
        """
        :param yt: true labels as numpy array of [[0, 1], [1,0]]
        :param yp: predicted labels as numpy array of [[0.2, 0.8], [0.8, 0.2]]
        :return: auc score, sensitivity, specificity, J-index
        """
        assert yt.shape[-1] == yp.shape[-1] == 2, \
            "expected shape of (?,2) " \
            "but get shape of true labels as {} and predict labels shape as {}".format(yt.shape, yp.shape)
        auc_score_1 = roc_auc_score(yt, yp, average='micro')
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
    def _get_prf(yt, yp):
        p, r, f, s = precision_recall_fscore_support(y_true=yt, y_pred=yp, average='micro')
        return p, r, f

    @staticmethod
    def _get_acc(yt, yp):
        return accuracy_score(yt, yp)
