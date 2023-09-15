import logging
import datasets
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Sequence
from dataclasses import dataclass

# set up logging
from .logger import setup_logger

setup_logger()
logger = logging.getLogger("galactic")


@dataclass
class GalacticDataset:
    dataset: datasets.Dataset
    model: Optional[object] = None
    emb_matrix: Optional[np.ndarray] = None
    cluster_ids: Optional[Sequence[int]] = None
    cluster_centers: Optional[dict[int, np.ndarray]] = None
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        # add unique increaing int __id field if it doesn't already exist
        if "__id" not in self.dataset.column_names:
            self.dataset = self.dataset.map(
                lambda _, i: {"__id": i},
                with_indices=True,
            )
        elif "__id" in self.dataset.column_names:
            if len(self.dataset["__id"]) != len(set(self.dataset["__id"])):
                raise ValueError("Dataset contains duplicate __id values.")

        # if we loaded something that already has clusters/embeddings, but no cluster centers, we can set them
        if (
            "__cluster" in self.dataset.column_names
            and "__embedding" in self.dataset.column_names
            and self.cluster_centers is None
        ):
            self.cluster_centers = {}
            self.cluster_ids = set(self.dataset["__cluster"])
            self.clusters = len(self.cluster_ids)
            for i in self.cluster_ids:
                cluster = self.dataset.filter(lambda x: x["__cluster"] == i)
                self.cluster_centers[i] = np.mean(
                    np.array(cluster["__embedding"]), axis=0
                )

        # if we loaded something that already has embeddings, we can set the embedding matrix
        if (
            "__embedding" in self.dataset.column_names
            and self.emb_matrix is None
        ):
            self.emb_matrix = np.array(self.dataset["__embedding"])

    def __repr__(self):
        return pd.DataFrame(self.dataset.select(range(10))).__repr__()

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    # delegate basic stuff to the underlying Dataset object
    def __getattr__(self, name):
        if name in ["column_names", "features", "info"]:
            return getattr(self.dataset, name)
        elif name in ["filter", "map", "select", "shuffle"]:

            def wrapper(*args, **kwargs):
                result = getattr(self.dataset, name)(*args, **kwargs)
                return GalacticDataset(
                    dataset=result,
                    model=self.model,
                    emb_matrix=self.emb_matrix,
                    cluster_ids=self.cluster_ids,
                    cluster_centers=self.cluster_centers,
                    openai_api_key=self.openai_api_key,
                )

            return wrapper
