"""Dataset utilities for END4Rec."""

from __future__ import annotations

import csv
from pathlib import Path

import mindspore.dataset as ds
import numpy as np


COLUMNS = ["item_ids", "behavior_ids", "positions", "labels"]


def data_generator(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            item_ids = np.array([int(x) for x in row["item_ids"].split(",")], dtype=np.int32)
            behavior_ids = np.array([int(x) for x in row["behavior_ids"].split(",")], dtype=np.int32)
            positions = np.array([int(x) for x in row["positions"].split(",")], dtype=np.int32)
            label = np.array([int(row["label"])], dtype=np.int32)
            yield item_ids, behavior_ids, positions, label


def dummy_data_generator(num_samples: int, seq_length: int, num_items: int, num_behaviors: int):
    for _ in range(num_samples):
        item_ids = np.random.randint(1, num_items, size=(seq_length,), dtype=np.int32)
        behavior_ids = np.random.randint(0, num_behaviors, size=(seq_length,), dtype=np.int32)
        positions = np.arange(seq_length, dtype=np.int32)
        label = np.array([np.random.randint(0, num_items)], dtype=np.int32)
        yield item_ids, behavior_ids, positions, label


def get_dataset(
    file_path: str | None,
    batch_size: int,
    num_shards: int = 1,
    shard_id: int = 0,
    num_samples: int = 2048,
    seq_length: int = 50,
    num_items: int = 10000,
    num_behaviors: int = 4,
):
    """Build training dataset from CSV file or synthetic fallback data."""

    if file_path and Path(file_path).exists():
        source = data_generator(file_path)
    else:
        source = dummy_data_generator(num_samples, seq_length, num_items, num_behaviors)

    dataset = ds.GeneratorDataset(
        source,
        column_names=COLUMNS,
        num_shards=num_shards,
        shard_id=shard_id,
        shuffle=True,
    )
    return dataset.batch(batch_size)
