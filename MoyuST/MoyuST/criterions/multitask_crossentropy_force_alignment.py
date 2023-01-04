#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions.multitask_crossentropy_with_contrastive_with_extra_mt import MultiTaskCrossEntropyWithContrastiveWithExtraMT
from fairseq.criterions import register_criterion
import math
import logging

logger = logging.getLogger(__name__)

@register_criterion("multi_task_cross_entropy_force_alignment")
class MultiTaskCrossEntropyFroceAlignment(MultiTaskCrossEntropyWithContrastiveWithExtraMT):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            attn_coefficient=0.,
            contrastive_weight=0.0,
            contrastive_temperature=1.0,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            contrastive_weight,
            contrastive_temperature,
        )
        self.attn_coefficient = attn_coefficient

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        MultiTaskCrossEntropyWithContrastiveWithExtraMT.add_args(parser)
        parser.add_argument('--attn-coefficient', default=0., type=float,
                            help='the weight of force alignment loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        label_smoothed_nll_loss, nll_loss = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_asr, nll_loss_asr = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_mt, nll_loss_mt = torch.tensor(0.0), torch.tensor(0.0)
        contrastive_loss, short_audio_len = torch.tensor(0.0), None
        force_alignment_loss = torch.tensor(0.0)

        if "mode" in sample["net_input"] and sample["net_input"]["mode"] == "text_to_text":
            sample["dataset_type"] = "mt"
            sample["net_input"]["is_text_input"] = True
        else:
            sample["net_input"]["is_text_input"] = False
        _net_output = model(**sample["net_input"])  # (x, extra)
        if model.training:
            net_output, encoder_out = _net_output
            if (sample["dataset_type"] != "mt") and (self.contrastive_weight > 0):
                contrastive_loss, short_audio_len = self.compute_contrastive_loss(
                    model, sample, encoder_out,
                    reduce=reduce, return_short_audio_len=True
                )
        else:
            net_output = _net_output

        if sample["target"] is not None:
            if sample["dataset_type"] == "st" and model.training:
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                label_smoothed_nll_loss_asr, nll_loss_asr = self.compute_loss_asr(model, sample, reduce=reduce)
                label_smoothed_nll_loss_mt, nll_loss_mt = self.compute_loss_mt(model, sample, reduce=reduce)
                force_alignment_loss = self.compute_force_alignment_loss(model, encoder_out, net_output, reduce=reduce)

            else:  # mt type compute CE_mt loss
                label_smoothed_nll_loss_mt, nll_loss_mt = self.compute_loss(model, net_output, sample, reduce=reduce)

        if sample["dataset_type"] == "st":
            source_ntokens = sample["source_ntokens"]
            target_ntokens = sample["target_ntokens"]
            target_ntokens_st = target_ntokens
            target_ntokens_mt = 0
            sample_size = sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]
        else:
            source_ntokens = 0
            target_ntokens = sample["ntokens"]
            target_ntokens_mt = target_ntokens
            target_ntokens_st = 0
            sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        nsentences = sample["target"].size(0)
        if sample["dataset_type"] == "st":
            multi_ce_loss = label_smoothed_nll_loss + label_smoothed_nll_loss_asr + label_smoothed_nll_loss_mt
            loss = multi_ce_loss + self.contrastive_weight * contrastive_loss + self.attn_coefficient * force_alignment_loss
        else:
            loss = label_smoothed_nll_loss_mt

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "contrastive_loss": contrastive_loss.data,
            "source_ntokens": source_ntokens,
            "target_ntokens": target_ntokens,
            "target_ntokens_mt": target_ntokens_mt,
            "target_ntokens_st": target_ntokens_st,
            "ntokens": target_ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "nll_loss_asr": nll_loss_asr.data,
            "nll_loss_mt": nll_loss_mt.data,
            "st_nsentences": nsentences if sample["dataset_type"] != "mt" else 0,
            "mt_nsentences": nsentences if sample["dataset_type"] == "mt" else 0,
            "force_alignment_loss": force_alignment_loss.data if sample["dataset_type"] != "mt" else 0,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_force_alignment_loss(self, model, encoder_out, net_output, reduce=True):
        """Compute the force alignment loss for the given sample.
        """
        if net_output[1]["decoder_layer_attns"] is None:
            return torch.tensor(0.0)
        layer_attns = net_output[1]["decoder_layer_attns"] # [bsz, tgt_len, src_len]

        def cosine_similarity(v1, v2):
            v1_norm = v1 / torch.norm(v1)
            v2_norm = v2 / torch.norm(v2)
            return torch.sum(v1_norm * v2_norm)

        def build_standard_attn(bsz, tgt_len, src_len):
            standard_attn = torch.zeros((bsz, tgt_len, src_len))

            # local attention
            ratio = int(max(tgt_len, src_len) / min(tgt_len, src_len) - 1e-6)

            stride = ratio

            if tgt_len == src_len:
                stride = 1
                length = 1
            else:
                length = max(tgt_len, src_len) - stride * (min(tgt_len, src_len) - 1)

            if tgt_len < src_len:
                for i in range(tgt_len):
                    standard_attn[:, i, i * stride: i * stride + length] = 1
            else:
                for i in range(src_len):
                    standard_attn[:, i * stride: i * stride + length, i] = 1
            
            # global attention
            standard_attn[:, -1:, :] = 1
            standard_attn[:, :, -1:] = 1

            # randon attention
            random_ratio = 0.15
            random_number = int(random_ratio * (tgt_len * src_len))
            random_position = torch.randint(0, tgt_len * src_len, (random_number, ))
            for pos in random_position:
                pos_i = torch.floor(torch.div(pos, src_len)).int()
                pos_j = pos % src_len
                standard_attn[:, pos_i, pos_j] = 1

            # standard_attn = F.softmax(standard_attn, dim=-1)

            return standard_attn

        bsz, tgt_len, src_len = layer_attns[0].size(0), layer_attns[0].size(1), layer_attns[0].size(2)
        standard_attn = build_standard_attn(bsz, tgt_len, src_len).to(layer_attns[0].device)

        loss = 0
        for attn in layer_attns:
            loss += torch.dist(standard_attn, attn, p = 2)

        if reduce:
            loss = loss.sum()
        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        force_alignment_loss_sum = sum(log.get("force_alignment_loss", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        target_ntokens = sum(log.get("target_ntokens", 0) for log in logging_outputs)
        source_ntokens = sum(log.get("source_ntokens", 0) for log in logging_outputs)
        target_ntokens_mt = sum(log.get("target_ntokens_mt", 0) for log in logging_outputs)
        target_ntokens_st = sum(log.get("target_ntokens_st", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        mt_nsentences = sum(log.get("mt_nsentences", 0) for log in logging_outputs)
        st_nsentences = sum(log.get("st_nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        nll_loss_sum_asr = sum(log.get("nll_loss_asr", 0) for log in logging_outputs)
        nll_loss_sum_mt = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)

        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss",
                           nll_loss_sum / target_ntokens_st / math.log(2), target_ntokens_st, round=3)
        metrics.log_scalar("contrasitve_loss",
                           contrastive_loss_sum / st_nsentences / math.log(2), st_nsentences, round=3)
        metrics.log_scalar("nll_loss_asr",
                           nll_loss_sum_asr / source_ntokens / math.log(2), source_ntokens, round=3)
        metrics.log_scalar("nll_loss_mt",
                           nll_loss_sum_mt / target_ntokens / math.log(2), target_ntokens, round=3)
        metrics.log_scalar("force_alignment_loss",
                            force_alignment_loss_sum / math.log(2), target_ntokens, round=3)
        metrics.log_scalar("bsz_st", st_nsentences, priority=190, round=1)
        metrics.log_scalar("bsz_mt", mt_nsentences, priority=190, round=1)

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
