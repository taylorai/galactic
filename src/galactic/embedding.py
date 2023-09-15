from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from typing import Union
import openai
import tiktoken
from .async_openai import embed_texts_with_openai

import logging

logger = logging.getLogger("galactic")


class EmbeddingModel:
    def __init__(self, model_path, tokenizer_path, model_type, max_length=512):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.model_type = model_type
        if model_type == "onnx":
            import onnxruntime as ort

            self.session = ort.InferenceSession(self.model_path)
        elif model_type == "ctranslate2":
            import ctranslate2

            self.session = ctranslate2.Encoder(
                self.model_path, compute_type="int8"
            )

    def split_and_tokenize(self, text):
        # first make into tokens
        tokenized = self.tokenizer(text, return_tensors="np")

        # pad to be a multiple of max_length
        rounded_len = int(
            np.ceil(len(tokenized["input_ids"][0]) / self.max_length)
            * self.max_length
        )
        for k in tokenized:
            tokenized[k] = np.pad(
                tokenized[k],
                ((0, 0), (0, rounded_len - len(tokenized[k][0]))),
                constant_values=0,
            )

        # reshape into batch with max length
        for k in tokenized:
            tokenized[k] = tokenized[k].reshape((-1, self.max_length))

        return tokenized

    def forward_onnx(self, input):
        outs = []
        for seq in range(len(input["input_ids"])):
            out = self.session.run(
                None,
                {
                    "input_ids": input["input_ids"][seq : seq + 1],
                    "attention_mask": input["attention_mask"][seq : seq + 1],
                    "token_type_ids": input["token_type_ids"][seq : seq + 1],
                },
            )[0]
            outs.append(out)
        out = np.concatenate(outs, axis=0)  # bsz, seq_len, hidden_size
        return out

    def forward_ctranslate2(self, input):
        input_ids = input["input_ids"].tolist()
        out = self.session.forward_batch(input_ids).pooler_output
        return out

    def predict(self, input):
        input = self.split_and_tokenize(input)
        if self.model_type == "onnx":
            # print("onnx")
            out = self.forward_onnx(input)
            # print("shape", out.shape)
            out = np.mean(out, axis=(0, 1))  # mean of seq and bsz
        elif self.model_type == "ctranslate2":
            # print("ct2")
            out = self.forward_ctranslate2(input)
            out = np.mean(out, axis=0)  # mean just over bsz
        # normalize to unit vector
        return out / np.linalg.norm(out)

    def __call__(self, input):
        return self.predict(input)


def embed_with_openai(text: str, key: str):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    # if longer than 8191, split into chunks
    if len(tokens) > 8191:
        num_chunks = int(np.ceil(len(tokens) / 8191))
        chunks = np.array_split(tokens, num_chunks).tolist()
    else:
        chunks = [tokens]
    openai.api_key = key
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=chunks,
    )
    embs = np.array([res.data[i].embedding for i in range(len(res.data))])
    avg = np.mean(embs, axis=0)
    return avg / np.linalg.norm(avg)


def get_embedding_model(self, backend: str = "auto"):
    # if auto, then select intelligently
    if backend == "auto":
        if "__embedding" in self.dataset.column_names:
            embedding_dim = len(self.dataset["__embedding"][0])
            if embedding_dim == 384:
                backend = "cpu"
            elif embedding_dim == 1536:
                backend = "openai"
            logger.info(
                f"Dataset already contains __embedding field. Initializing to use {backend} backend for queries."
            )

    if backend == "cpu":
        model_path = hf_hub_download(
            "TaylorAI/galactic-models", filename="model_quantized.onnx"
        )
        tokenizer_path = "Supabase/gte-small"
        max_length = 512
        model = EmbeddingModel(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            model_type="onnx",
            max_length=max_length,
        )
        self.model = model
    elif backend == "gpu":
        raise NotImplementedError("GPU backend not yet implemented.")
    elif backend == "openai":
        if self.openai_api_key is None:
            raise ValueError(
                "You must set openai_api_key before calling get_embeddings() with openai backend."
            )
        self.model = lambda x: embed_with_openai(x, self.openai_api_key)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_embeddings(self, field: str, backend: str = "auto"):
    self.get_embedding_model(backend=backend)

    if backend == "openai":
        embs = embed_texts_with_openai(
            self.dataset[field], self.openai_api_key
        )
        self.dataset = self.dataset.map(
            lambda x, idx: {"__embedding": embs[idx]}, with_indices=True
        )
    else:
        self.dataset = self.dataset.map(
            lambda x: {f"__embedding": self.model(x[field])}
        )
    self.emb_matrix = np.array(self.dataset["__embedding"])
    logger.info(f"Created embeddings on field '{field}'")
    return self


def get_nearest_neighbors(self, query: Union[str, np.ndarray], k: int = 5):
    if "__embedding" not in self.dataset.column_names:
        raise ValueError(
            "You must call get_embeddings() before calling get_nearest_neighbors(). If your dataset already has an embeddings column, make sure it's called '__embeddings'."
        )
    if not hasattr(self, "emb_matrix"):
        self.emb_matrix = np.array(self.dataset["__embedding"])
    if not hasattr(self, "model"):
        self.get_embedding_model()
    if isinstance(query, str):
        query = self.model(query)
    scores = np.dot(self.emb_matrix, query)
    top_k = np.argsort(scores)[::-1][:k]
    df = pd.DataFrame.from_records(self.dataset[top_k])
    return df.drop(columns=["__embedding"])


# finetune embeddings as in https://github.com/openai/openai-cookbook/blob/main/examples/Customizing_embeddings.ipynb
def tune_embeddings(self):
    pass
