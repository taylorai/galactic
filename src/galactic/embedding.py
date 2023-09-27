from transformers import AutoTokenizer
from typing import Optional
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from typing import Union, Literal
import openai
import tiktoken
from .async_openai import embed_texts_with_openai
import time
import logging

logger = logging.getLogger("galactic")


class EmbeddingModel:
    """
    This class provides methods to load embedding models and generate embeddings.

    :param model_path: Path to the model.
    :type model_path: str
    :param tokenizer_path: Path to the tokenizer.
    :type tokenizer_path: str
    :param model_type: Type of model to be loaded can be 'onnx' or 'ctranslate2'.
    :type model_type: str
    :param max_length: Maximum length of tokenization.
    :type max_length: int, optional, default=512

    Example:

    .. code-block:: python

        model = EmbeddingModel(model_path='path_to_model', tokenizer_path='path_to_tokenizer', model_type='onnx', max_length=512)
    """

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
        """
        Tokenize the text and pad it to be a multiple of max_length.

        :param text: The input text to be tokenized.
        :type text: str
        :return: Returns a dictionary containing tokenized and padded 'input_ids', 'attention_mask' and 'token_type_ids'.
        :rtype: Dict[str, numpy.ndarray]

        Example:

        .. code-block:: python

            tokenized_text = model.split_and_tokenize('sample text')
        """
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
    """
    Embeds a single text with OpenAI API.

    :param text: The input text to be embedded.
    :type text: str
    :param key: The API key for OpenAI.
    :type key: str
    :return: The normalized averaged embeddings.
    :rtype: numpy.ndarray

    Example:

    .. code-block:: python

        embedding = embed_with_openai('sample text', 'api_key')
    """
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


def initialize_embedding_model(self, backend: str = "auto"):
    """
    Initializes the embedding model based on the backend.

    :param backend: The backend used for embedding. Options: 'auto', 'cpu', 'gpu', 'openai'.
    :type backend: str, default='auto'
    :raises ValueError: Raises an exception if the backend is 'openai' but openai_api_key is not set, or if the backend is unknown.

    Example:

    .. code-block:: python

        initialize_embedding_model('cpu')
    """
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
        else:
            backend = "cpu"
            logger.info(
                f"Dataset does not contain __embedding field. Initializing to use {backend} backend for queries."
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


def get_embeddings(
    self,
    input_field: str,
    embedding_field: str = "__embedding",
    backend: str = "auto",
):
    """
    Get embeddings for a field in the dataset.

    .. code-block:: python

        # Replace "field_name1"
        ds.get_embeddings(field="field_name1")

    """
    self.initialize_embedding_model(backend=backend)

    # make sure the field doesn't exist already
    if embedding_field in self.dataset.column_names:
        raise ValueError(
            f"Field {embedding_field} already exists in dataset. Please choose a different name, or drop the column."
        )

    if backend == "auto":
        backend = "cpu"
    if backend == "openai":
        requests_left = {"n": len(self.dataset)}

        def _embed_batch(batch):
            start_time = time.time()
            embs = embed_texts_with_openai(
                batch[input_field], self.openai_api_key, show_progress=False
            )
            requests_left["n"] -= len(batch[input_field])
            logger.info(f"Requests left: {requests_left['n']}")
            end_time = time.time()
            # if it took less than 1m, cool down to avoid rate limits
            if end_time - start_time < 60 and requests_left["n"] > 0:
                time.sleep(60 - (end_time - start_time))
            return {embedding_field: embs}

        self.dataset = self.dataset.map(
            _embed_batch,
            batched=True,
            batch_size=self.max_requests_per_minute,
        )
    else:
        self.dataset = self.dataset.map(
            lambda x: {embedding_field: self.model(x[input_field])}
        )
    self.emb_matrix = np.array(self.dataset[embedding_field])
    logger.info(
        f"Created embeddings on field '{input_field}', stored in '{embedding_field}'."
    )
    return self


def get_nearest_neighbors(
    self,
    query: Union[str, np.ndarray, list[float]],
    k: int = 5,
    embedding_field: str = "__embedding",
):
    """Get the nearest neighbors to a query."""
    if embedding_field not in self.dataset.column_names:
        raise ValueError(
            "You must call get_embeddings() before calling get_nearest_neighbors(). If your dataset already has an embeddings column and it's not '__embeddings', pass it as the 'embedding_field' argument."
        )
    if not hasattr(self, "emb_matrix"):
        self.emb_matrix = np.array(self.dataset["__embedding"])
    if not hasattr(self, "model"):
        self.initialize_embedding_model()
    if isinstance(query, str):
        query = self.model(query)
    scores = np.dot(self.emb_matrix, query)
    top_k = list(np.argsort(scores)[::-1][:k])
    top_k_items = self.dataset.select(top_k)
    return top_k_items.to_list()


def reduce_embedding_dim(
    self,
    new_column: Optional[str] = None,
    method: Literal[
        "pca", "incremental_pca", "kernel_pca", "svd", "umap"
    ] = "pca",
    n_dims: int = 50,
    embedding_field: str = "__embedding",
    **kwargs,
):
    n_features = len(self.dataset[embedding_field][0])
    if new_column is None:
        new_column = f"{embedding_field}_{method}_{n_dims}"
    if method == "pca":
        X = np.array(self.dataset[embedding_field])
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_dims)
        X_transformed = pca.fit_transform(X).tolist()
        self.dataset = self.dataset.add_column(new_column, X_transformed)
    elif method == "incremental_pca":
        from sklearn.decomposition import IncrementalPCA

        pca = IncrementalPCA(n_components=n_dims)
        batch_size = kwargs.get("batch_size", 5 * n_features)

        # fit the PCA on batches of the dataset
        def _fit_batch(batch):
            pca.partial_fit(batch[embedding_field])
            return None

        self.dataset.map(
            _fit_batch,
            batched=True,
            batch_size=batch_size,
        )
        # transform the dataset
        self.dataset = self.dataset.map(
            lambda x: {new_column: pca.transform(x[embedding_field]).tolist()},
            batched=True,
            batch_size=batch_size,
        )
    elif method == "kernel_pca":
        X = np.array(self.dataset[embedding_field])
        from sklearn.decomposition import KernelPCA

        pca = KernelPCA(
            n_components=n_dims, kernel=kwargs.get("kernel", "rbf")
        )
        X_transformed = pca.fit_transform(X).tolist()
        self.dataset = self.dataset.add_column(new_column, X_transformed)
    elif method == "svd":
        X = np.array(self.dataset[embedding_field])
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=n_dims)
        X_transformed = svd.fit_transform(X).tolist()
        self.dataset = self.dataset.add_column(new_column, X_transformed)

    elif method == "umap":
        # warn that it will be slow if embedding dim is large
        if len(self.dataset[embedding_field][0]) >= 100:
            logger.info(
                "Embedding dimension is large, which can slow UMAP down a lot, even for small datasets. You might consider PCA first with e.g. n_dims=50, followed by UMAP."
            )
        import umap

        reducer = umap.UMAP(n_components=n_dims, **kwargs)
        X = np.array(self.dataset[embedding_field])
        X_transformed = reducer.fit_transform(X).tolist()
        self.dataset = self.dataset.add_column(new_column, X_transformed)
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(
        f"Reduced embeddings to {n_dims} dimensions using {method}. New embeddings stored in column '{new_column}'"
    )

    return self


# finetune embeddings as in https://github.com/openai/openai-cookbook/blob/main/examples/Customizing_embeddings.ipynb
def tune_embeddings(self):
    pass
