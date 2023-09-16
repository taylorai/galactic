import os
from typing import Optional, Callable
import datasets
import json
import pybloom_live
import pandas as pd
from .utils import (
    read_csv,
)
from tqdm.auto import tqdm
import pyarrow.parquet as pq

import logging

logger = logging.getLogger("galactic")


def from_csv(cls, path: str):
    df = read_csv(path)
    return cls.from_pandas(df)


def from_jsonl(cls, path, **kwargs):
    dataset = datasets.load_dataset("json", data_files=path, **kwargs)
    return cls(dataset)


def from_parquet(cls, path):
    df = pd.read_parquet(path)
    return cls.from_pandas(df)


def from_pandas(cls, df, **kwargs):
    dataset = datasets.Dataset.from_pandas(df, **kwargs)
    return cls(dataset)


def from_hugging_face(
    cls, path: str, split: str, config_name: Optional[str] = None, **kwargs
):
    dataset = datasets.load_dataset(
        path, name=config_name, split=split, **kwargs
    )
    return cls(dataset)


def from_hugging_face_stream(
    cls,
    path: str,
    split: str,
    config_name: Optional[str] = None,
    filters=list[Callable[[dict], bool]],
    dedup_fields: Optional[list[str]] = None,
    max_samples: Optional[int] = 200000,
    **kwargs,
):
    handle = datasets.load_dataset(
        path, name=config_name, split=split, streaming=True, **kwargs
    )
    # first, apply the filters, which will operate on the stream
    if filters is not None:
        for filter in filters:
            handle = handle.filter(filter)

    # then set up bloom filter if applicable
    bloom = None
    if dedup_fields is not None and len(dedup_fields) > 0:
        bloom = pybloom_live.BloomFilter(
            capacity=max_samples if max_samples is not None else int(1.0e9),
            error_rate=0.001,
        )

    samples = []
    progress = tqdm(total=max_samples, desc=f"Loading samples from {path}")
    for idx, sample in enumerate(handle):
        if bloom is not None:
            key = json.dumps({k: sample[k] for k in dedup_fields})
            if key in bloom:
                continue
            bloom.add(key)
        samples.append(sample)
        progress.update(1)
        if len(samples) >= max_samples:
            break
    progress.close()
    msg = f"Loaded {len(samples)} samples from {path}. "
    if idx > len(samples):
        msg += f"Removed {idx - len(samples)} samples that were duplicated. "
    logger.info(msg)
    return cls(datasets.Dataset.from_list(samples))


# save dataset as jsonl
def save(self, path: str, overwrite: bool = False) -> None:
    # check if exists
    if os.path.exists(path) and not overwrite:
        raise ValueError(
            f"Path {path} already exists. Use overwrite=True to overwrite."
        )
    # save
    with open(path, "wb") as f:
        for sample in self.dataset:
            f.write(json.dumps(sample).encode("utf-8"))
            f.write("\n".encode("utf-8"))
