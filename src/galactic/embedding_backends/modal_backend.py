import numpy as np
from modal import Cls
from typing import Union, Literal


class ModalEmbeddingModel:
    def __init__(self, model_name="bge-micro"):
        ModalEmbeddingModel = Cls.lookup(
            "modal_embeddings", "RemoteEmbeddingModel"
        )
        self.model = ModalEmbeddingModel(model_name=model_name)

    def embed(
        self,
        text: str,
        normalize: bool = False,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        return self.model.embed_batch.remote(
            [text], normalize=normalize, pad=pad, split_strategy=split_strategy
        ).reshape(-1)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 250,
        normalize: bool = False,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        chunks = [
            texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]
        promises = self.model.embed_batch.map(
            chunks,
            kwargs={
                "normalize": normalize,
                "pad": pad,
                "split_strategy": split_strategy,
            },
        )
        results = [p for p in promises]  # list of np arrays
        flattened = np.concatenate(results, axis=0)
        return flattened
