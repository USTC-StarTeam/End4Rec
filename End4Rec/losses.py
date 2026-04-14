"""Loss definitions for END4Rec."""

import mindspore.nn as nn
import mindspore.ops as ops


class ContrastiveLossCell(nn.Cell):
    """Simple contrastive objective used by END4Rec."""

    def __init__(self):
        super().__init__()
        self.sigmoid = ops.Sigmoid()
        self.reduce_mean = ops.ReduceMean()
        self.log = ops.Log()

    def construct(self, clean_repr, origin_repr, noise_repr):
        q_clean = ops.ReduceMean(keep_dims=False)(clean_repr, 1)
        q_origin = ops.ReduceMean(keep_dims=False)(origin_repr, 1)
        q_noise = ops.ReduceMean(keep_dims=False)(noise_repr, 1)

        loss1 = -self.reduce_mean(self.log(self.sigmoid(q_clean - q_origin) + 1e-9))
        loss2 = -self.reduce_mean(self.log(self.sigmoid(q_origin - q_noise) + 1e-9))
        return loss1 + loss2


class End4RecLoss(nn.Cell):
    """Joint objective: prediction + regularization + contrastive losses."""

    def __init__(self, model, reg_weight=0.01, contrast_weight=0.1):
        super().__init__()
        self.model = model
        self.reg_weight = reg_weight
        self.contrast_weight = contrast_weight
        self.pred_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.contrast_loss_fn = ContrastiveLossCell()

    def construct(self, item_ids, behavior_ids, positions, labels):
        logits, reg_loss, seq_repr, hard_pos, hard_neg, soft_pos, soft_neg = self.model(
            item_ids, behavior_ids, positions
        )
        labels = ops.Squeeze()(labels)
        pred_loss = self.pred_loss(logits, labels)

        contrast_hard = self.contrast_loss_fn(hard_pos, seq_repr, hard_neg)
        contrast_soft = self.contrast_loss_fn(soft_pos, hard_pos, soft_neg)

        return (
            pred_loss
            + self.reg_weight * reg_loss
            + self.contrast_weight * (contrast_hard + contrast_soft)
        )
