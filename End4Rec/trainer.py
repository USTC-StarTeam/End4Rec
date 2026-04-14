"""Training helper for staged END4Rec optimization."""

from __future__ import annotations

import mindspore.nn as nn

from evaluation import evaluate
from utils import TrainOneStepCell, set_requires_grad


def train_stage(model_loss, optimizer, dataset, stage_name: str, epochs: int):
    train_one_step = TrainOneStepCell(model_loss, optimizer)
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for item_ids, behavior_ids, positions, labels in dataset.create_tuple_iterator():
            loss = train_one_step(item_ids, behavior_ids, positions, labels)
            epoch_loss += float(loss.asnumpy())
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        print(f"[Stage {stage_name}] epoch={epoch + 1}/{epochs}, loss={avg_loss:.6f}")


def fit_end4rec(model, model_loss, train_dataset, config):
    """4-stage training schedule from the paper implementation."""

    stage_cfg = [
        ("1", ["embedding", "ebm"], config["learning_rate_stage1"], config["stage1_epochs"]),
        ("2", ["hard_noise"], config["learning_rate_stage2"], config["stage2_epochs"]),
        ("3", ["soft_noise"], config["learning_rate_stage3"], config["stage3_epochs"]),
        (
            "4",
            ["embedding", "ebm", "hard_noise", "soft_noise", "output_layer"],
            config["learning_rate_stage4"],
            config["stage4_epochs"],
        ),
    ]

    for stage_name, train_modules, lr, epochs in stage_cfg:
        for module_name in ["embedding", "ebm", "hard_noise", "soft_noise", "output_layer"]:
            set_requires_grad(getattr(model, module_name), module_name in train_modules)

        params = []
        for module_name in train_modules:
            params.extend(list(getattr(model, module_name).get_parameters()))

        optimizer = nn.Adam(params, learning_rate=lr)
        train_stage(model_loss, optimizer, train_dataset, stage_name, epochs)


def evaluate_if_available(model, eval_dataset, topk: int):
    if eval_dataset is None:
        return None
    hr, ndcg = evaluate(model, eval_dataset, k=topk)
    return {"hr": float(hr), "ndcg": float(ndcg)}
