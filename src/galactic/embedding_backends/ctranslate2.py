import os
import multiprocessing
import torch
import ctranslate2
from .base import EmbeddingModelBase
import numpy as np
from transformers import AutoTokenizer
from typing import Literal
from huggingface_hub import snapshot_download
import logging

logger = logging.getLogger("galactic")

model_registry = {
    "gte-small": {
        "remote_file": "gte-small",
        "tokenizer": "Supabase/gte-small",
    },
    "gte-tiny": {
        "remote_file": "gte-tiny",
        "tokenizer": "Supabase/gte-small",
    },
    "bge-micro": {
        "remote_file": "bge-micro",
        "tokenizer": "BAAI/bge-small-en-v1.5",
    },
}


class CT2EmbeddingModel(EmbeddingModelBase):
    def __init__(
        self,
        model_name: str = "bge-micro",
        device: str = "gpu",
        max_length: int = 512,
    ):
        super().__init__()

        # select device
        if device == "gpu":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                logger.warning("GPU not available, using CPU.")
                self.device = "cpu"

        # download model
        model_path = snapshot_download(
            "TaylorAI/galactic-models",
            allow_patterns=[model_registry[model_name]["remote_file"] + "/*"],
        )
        model_path = os.path.join(
            model_path, model_registry[model_name]["remote_file"]
        )
        self.model = ctranslate2.Encoder(
            model_path,
            device=self.device,
            intra_threads=multiprocessing.cpu_count(),
        )
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_registry[model_name]["tokenizer"]
        )

        # select batch size
        if self.device == "cuda":
            self.batch_size = self._select_batch_size()
        else:
            self.batch_size = 1

    def _select_batch_size(self):
        current_batch_size = 4096
        sample_batch = [[11] * self.max_length] * current_batch_size
        while current_batch_size > 0:
            try:
                self.model.forward_batch(sample_batch)
                logger.info(f"Batch size selected: {current_batch_size}")
                return current_batch_size
            except Exception as e:
                if "out of memory" in str(e):
                    current_batch_size = current_batch_size // 2
                    sample_batch = sample_batch[:current_batch_size]
                else:
                    raise e

        logger.error(
            "Failed with batch size 1. You'll need a bigger GPU or a smaller model."
        )
        raise RuntimeError(
            "Failed with batch size 1. You'll need a bigger GPU or a smaller model."
        )

    def embed(
        self,
        text: str,
        normalize: bool = False,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        if pad:
            logger.warning(
                "Padding is not necessary for CTranslate2. Ignoring pad=True."
            )

        input = self.split_and_tokenize_single(
            text, pad=False, split_strategy=split_strategy
        )
        output = self.model.forward_batch(input["input_ids"]).pooler_output
        if self.device == "cuda":
            output = (
                torch.as_tensor(output, device="cuda")
                .cpu()
                .numpy()
                .mean(axis=0)
            )
        else:
            output = np.array(output).mean(axis=0)

        if normalize:
            output = output / np.linalg.norm(output)

        return output

    def embed_batch(
        self,
        texts: list[str],
        normalize: bool = False,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        if pad:
            logger.warning(
                "Padding is not necessary for CTranslate2. Ignoring pad=True."
            )

        tokenized = self.split_and_tokenize_batch(
            texts, pad=False, split_strategy=split_strategy
        )
        inputs = tokenized["tokens"]
        offsets = tokenized["offsets"]
        outputs = None
        for i in range(0, len(inputs["input_ids"]), self.batch_size):
            batch = inputs["input_ids"][i : i + self.batch_size]
            batch_out = self.model.forward_batch(batch).pooler_output
            if self.device == "cuda":
                batch_out = (
                    torch.as_tensor(batch_out, device="cuda")
                    .cpu()
                    .numpy()  # batch_size, hidden_size
                )
            else:
                batch_out = np.array(batch_out)  # batch_size, hidden_size
            if outputs is None:
                outputs = batch_out
            else:
                outputs = np.concatenate([outputs, batch_out], axis=0)

        # use offsets to average each text's embeddings
        embs = []
        for i in range(len(offsets) - 1):
            start, end = offsets[i], offsets[i + 1]
            chunk = outputs[start:end]
            averaged = chunk.mean(axis=0)
            if normalize:
                averaged = averaged / np.linalg.norm(averaged)
            embs.append(averaged)

        return np.array(embs)
