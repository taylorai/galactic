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
    """
    Loads a dataset from a CSV file.

    .. code-block:: python

        # Example usage:
        ds = GalacticDataset.from_csv('path_to_file.csv')

    :param path: The path to the CSV file.
    :type path: str
    :return: An instance of the class populated with the dataset from the CSV file.
    :rtype: cls
    """
    df = read_csv(path)
    return cls.from_pandas(df)


def from_jsonl(cls, path):
    """
    Loads a dataset from a JSONL file.

    .. code-block:: python

        # Example usage:
        ds = GalacticDataset.from_jsonl('path_to_file.jsonl')

    :param path: The path to the JSONL file.
    :type path: str
    :return: An instance of the class populated with the dataset from the JSONL file.
    :rtype: cls
    """
    df = pd.read_json(path, lines=True, orient="records")
    return cls.from_pandas(df)


def from_parquet(cls, path):
    """
    Loads a dataset from a Parquet file.

    .. code-block:: python

        # Example usage:
        ds = GalacticDataset.from_parquet('path_to_file.parquet')

    :param path: The path to the Parquet file.
    :type path: str
    :return: An instance of the class populated with the dataset from the Parquet file.
    :rtype: cls
    """
    df = pd.read_parquet(path)
    return cls.from_pandas(df)


def from_pandas(cls, df, **kwargs):
    """
    Loads a dataset from a Pandas DataFrame.

    .. code-block:: python

        # Example usage:
        ds = GalacticDataset.from_pandas(dataframe)

    :param df: The Pandas DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param kwargs: Additional keyword arguments passed to datasets.Dataset.from_pandas().
    :return: An instance of the class populated with the dataset from the DataFrame.
    :rtype: cls
    """
    dataset = datasets.Dataset.from_pandas(df, **kwargs)
    return cls(dataset)


def from_hugging_face(
    cls, path: str, split: str, config_name: Optional[str] = None, **kwargs
):
    """
    Loads a dataset from the Hugging Face hub.

    .. code-block:: python

        # Example usage:
        ds = GalacticDataset.from_hugging_face('dataset_path', 'split', 'config_name')

    :param path: The path to the dataset on the Hugging Face hub.
    :type path: str
    :param split: The specific split of the dataset to load.
    :type split: str
    :param config_name: The name of the configuration to load, if applicable. Defaults to None.
    :type config_name: Optional[str]
    :param kwargs: Additional keyword arguments passed to datasets.load_dataset().
    :return: An instance of the class populated with the dataset from the Hugging Face hub.
    :rtype: cls
    """
    dataset = datasets.load_dataset(
        path, name=config_name, split=split, **kwargs
    )
    return cls(dataset)


def from_hugging_face_stream(
    cls,
    path: str,
    split: str,
    config_name: Optional[str] = None,
    filters: list[Callable[[dict], bool]] = None,
    dedup_fields: Optional[list[str]] = None,
    max_samples: Optional[int] = 200000,
    **kwargs,
):
    """
    Loads a dataset from the Hugging Face hub, streaming the data from disk.

    .. code-block:: python

        # Example usage:
        ds = GalacticDataset.from_hugging_face_stream('dataset_path', 'split', filters=[filter_func], dedup_fields=['field1', 'field2'])

    :param path: The path to the dataset on the Hugging Face hub.
    :type path: str
    :param split: The specific split of the dataset to load.
    :type split: str
    :param config_name: The name of the configuration to load, if applicable. Defaults to None.
    :type config_name: Optional[str]
    :param filters: A list of filter functions to apply to the stream. Each filter function should take a dictionary and return a boolean. Defaults to None.
    :type filters: Optional[list[Callable[[dict], bool]]]
    :param dedup_fields: A list of fields to use for deduplication via a Bloom filter. Defaults to None.
    :type dedup_fields: Optional[list[str]]
    :param max_samples: The maximum number of samples to load. Defaults to 200000.
    :type max_samples: Optional[int]
    :param kwargs: Additional keyword arguments passed to datasets.load_dataset().
    :return: An instance of the class populated with the streamed dataset from the Hugging Face hub.
    :rtype: cls
    """
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


# save dataset as jsonl or csv
def save(self, path: str, overwrite: bool = False) -> None:
    """
    Saves the dataset as a CSV or JSONL file.

    .. code-block:: python

        # Example usage:
        ds.save('path_to_file.csv', overwrite=True)

    :param path: The path where the dataset will be saved. The format is inferred from the file extension (either .csv or .jsonl).
    :type path: str
    :param overwrite: Whether to overwrite the file if it already exists. Defaults to False.
    :type overwrite: bool
    :raises ValueError: If the path already exists and overwrite is False, or if the file format is unsupported.
    """
    # check if exists
    if os.path.exists(path) and not overwrite:
        raise ValueError(
            f"Path {path} already exists. Use overwrite=True to overwrite."
        )
    # get format
    ext = path.split(".")[-1]
    if ext not in ["csv", "jsonl"]:
        raise ValueError(
            f"Unsupported file format {ext}. Must be either csv or jsonl."
        )

    # save
    if ext == "csv":
        self.dataset.to_csv(path)
    elif ext == "jsonl":
        df = self.dataset.to_pandas()
        df.to_json(path, orient="records", lines=True)
