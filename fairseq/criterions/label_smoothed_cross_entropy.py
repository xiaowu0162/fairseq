# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    dual_decoder_scheme = False

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if 'dual_decoder_scheme' in model.args and model.args.dual_decoder_scheme is True:
            LabelSmoothedCrossEntropyCriterion.dual_decoder_scheme = True
            net_output = model(**sample["net_input"])
            # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            loss, loss1, loss2, nll_loss1, nll_loss2 = self.compute_loss(model, net_output, sample, reduce=reduce, lambda_1=model.args.lambda_task_1, alternate=model.args.alternate_training, decoder1_ratio=model.args.decoder1_ratio)
            sample_size = (
                sample["target1"].size(0) if self.sentence_avg else sample["ntokens1"]
            )
            sample_size2 = (
                sample["target2"].size(0) if self.sentence_avg else sample["ntokens2"]
            )
            logging_output = {
                "loss": loss.data,
                "loss1": loss1.data,
                "loss2": loss2.data,
                "nll_loss1": nll_loss1.data,
                "nll_loss2": nll_loss2.data,
                # "ntokens": sample["ntokens1"],
                "ntokens1": sample["ntokens1"],
                "ntokens2": sample["ntokens2"],
                "nsentences": sample["target1"].size(0),
                "sample_size": sample_size,
                "sample_size2": sample_size2,
            }
        else:
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, target1=None, target2=None):
        lprobs = model.get_normalized_probs(net_output, log_probs=True, target1=target1, target2=target2)
        target = model.get_targets(sample, net_output, target1=target1, target2=target2)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, lambda_1=None, alternate=False, decoder1_ratio=1):
        if LabelSmoothedCrossEntropyCriterion.dual_decoder_scheme:
            assert lambda_1 is not None and 0 <= lambda_1 <= 1
            lambda_2 = 1 - lambda_1
            x1, x2, extra1, extra2 = net_output
            loss, loss1, loss2, nll_loss1, nll_loss2 = None, None, None, None, None
            if x1 is not None:
                lprobs1, target1 = self.get_lprobs_and_target(model, (x1, extra1), sample, target1=True)
                loss1, nll_loss1 = label_smoothed_nll_loss(
                    lprobs1,
                    target1,
                    self.eps,
                    ignore_index=self.padding_idx,
                    reduce=reduce,
                )
                # loss = loss1 * lambda_1
            if x2 is not None:
                lprobs2, target2 = self.get_lprobs_and_target(model, (x2, extra2), sample, target2=True)
                loss2, nll_loss2 = label_smoothed_nll_loss(
                    lprobs2,
                    target2,
                    self.eps,
                    ignore_index=self.padding_idx,
                    reduce=reduce,
                )
                # loss = loss + lambda_2 * loss2 if loss is not None else loss2 * lambda_2
            if alternate:
                r = torch.rand(1)
                if r < decoder1_ratio:
                    loss = (loss1 if loss1 is not None else 0) * lambda_1
                else:
                    loss = (loss2 if loss2 is not None else 0) * lambda_2
            else:
                loss = (loss1 if loss1 is not None else 0) * lambda_1 + (loss2 if loss2 is not None else 0) * lambda_2
            # print(loss, loss1, loss2)
            return loss, loss1, loss2, nll_loss1, nll_loss2
        else:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        if cls.dual_decoder_scheme:
            loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
            loss_sum1 = sum(log.get("loss1", 0) for log in logging_outputs)
            loss_sum2 = sum(log.get("loss2", 0) for log in logging_outputs)
            nll_loss_sum1 = sum(log.get("nll_loss1", 0) for log in logging_outputs)
            nll_loss_sum2 = sum(log.get("nll_loss2", 0) for log in logging_outputs)
            ntokens1 = sum(log.get("ntokens1", 0) for log in logging_outputs)
            ntokens2 = sum(log.get("ntokens2", 0) for log in logging_outputs)
            sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
            sample_size2 = sum(log.get("sample_size2", 0) for log in logging_outputs)

            metrics.log_scalar(
                "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "loss1", loss_sum1 / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "loss2", loss_sum2 / sample_size2 / math.log(2), sample_size2, round=3
            )
            metrics.log_scalar(
                "nll_loss1", nll_loss_sum1 / ntokens1 / math.log(2), ntokens1, round=3
            )
            metrics.log_scalar(
                "nll_loss2", nll_loss_sum2 / ntokens2 / math.log(2), ntokens2, round=3
            )
            metrics.log_derived(
                "ppl1", lambda meters: utils.get_perplexity(meters["nll_loss1"].avg)
            )
            metrics.log_derived(
                "ppl2", lambda meters: utils.get_perplexity(meters["nll_loss2"].avg)
            )

            total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
            if total > 0:
                metrics.log_scalar("total", total)
                n_correct = utils.item(
                    sum(log.get("n_correct", 0) for log in logging_outputs)
                )
                metrics.log_scalar("n_correct", n_correct)
                metrics.log_derived(
                    "accuracy",
                    lambda meters: round(
                        meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                    )
                    if meters["total"].sum > 0
                    else float("nan"),
                )
        else:
            loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
            nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

            metrics.log_scalar(
                "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )

            total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
            if total > 0:
                metrics.log_scalar("total", total)
                n_correct = utils.item(
                    sum(log.get("n_correct", 0) for log in logging_outputs)
                )
                metrics.log_scalar("n_correct", n_correct)
                metrics.log_derived(
                    "accuracy",
                    lambda meters: round(
                        meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                    )
                    if meters["total"].sum > 0
                    else float("nan"),
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True