"""Main training entry for END4Rec."""

from __future__ import annotations

import json
from pathlib import Path

import mindspore as ms
from mindspore.communication.management import get_group_size, get_rank, init

from dataset import get_dataset
from losses import End4RecLoss
from model import END4Rec
from trainer import evaluate_if_available, fit_end4rec


def setup_context():
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="GPU")
    try:
        init()
        group_size = get_group_size()
        rank = get_rank()
        ms.context.set_auto_parallel_context(parallel_mode="data_parallel", device_num=group_size)
    except RuntimeError:
        group_size = 1
        rank = 0
    return rank, group_size


def load_config(config_path: str | None):
    default_cfg = {
        "num_items": 10000,
        "num_behaviors": 4,
        "seq_length": 50,
        "d_model": 128,
        "num_blocks": 4,
        "epsilon": 1e-3,
        "reg_weight": 0.01,
        "contrast_weight": 0.1,
        "batch_size": 64,
        "train_data_file": "",
        "eval_data_file": "",
        "topk": 10,
        "stage1_epochs": 1,
        "stage2_epochs": 1,
        "stage3_epochs": 1,
        "stage4_epochs": 1,
        "learning_rate_stage1": 1e-3,
        "learning_rate_stage2": 1e-3,
        "learning_rate_stage3": 1e-3,
        "learning_rate_stage4": 5e-4,
        "num_dummy_samples": 512,
    }
    if not config_path:
        return default_cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    user_cfg = json.loads(path.read_text(encoding="utf-8"))
    default_cfg.update(user_cfg)
    return default_cfg


def train(config):
    rank, group_size = setup_context()

    model = END4Rec(
        num_items=config["num_items"],
        num_behaviors=config["num_behaviors"],
        seq_length=config["seq_length"],
        d_model=config["d_model"],
        num_blocks=config["num_blocks"],
        epsilon=config["epsilon"],
    )
    model_loss = End4RecLoss(
        model,
        reg_weight=config["reg_weight"],
        contrast_weight=config["contrast_weight"],
    )

    train_dataset = get_dataset(
        file_path=config.get("train_data_file"),
        batch_size=config["batch_size"],
        num_shards=group_size,
        shard_id=rank,
        num_samples=config["num_dummy_samples"],
        seq_length=config["seq_length"],
        num_items=config["num_items"],
        num_behaviors=config["num_behaviors"],
    )
    eval_dataset = None
    if config.get("eval_data_file"):
        eval_dataset = get_dataset(
            file_path=config["eval_data_file"],
            batch_size=config["batch_size"],
            num_shards=group_size,
            shard_id=rank,
            num_samples=min(256, config["num_dummy_samples"]),
            seq_length=config["seq_length"],
            num_items=config["num_items"],
            num_behaviors=config["num_behaviors"],
        )

    fit_end4rec(model, model_loss, train_dataset, config)

    metrics = evaluate_if_available(model, eval_dataset, config["topk"])
    if metrics and rank == 0:
        print(f"Evaluation@{config['topk']}: HR={metrics['hr']:.4f}, NDCG={metrics['ndcg']:.4f}")

    if rank == 0:
        ms.save_checkpoint(model, "end4rec_final.ckpt")


def main(config_path: str | None = None):
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    main()
