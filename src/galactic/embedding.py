from typing import Optional
import numpy as np
from typing import Union, Literal
from .embedding_backends.replicate_backend import ReplicateEmbeddingModel
from .embedding_backends.onnx_backend import ONNXEmbeddingModel
from .embedding_backends.ctranslate2_backend import CT2EmbeddingModel
from .embedding_backends.openai_backend import OpenAIEmbeddingModel
import logging

logger = logging.getLogger("galactic")


def initialize_embedding_model(
    self,
    backend: Literal[
        "cpu", "onnx", "gpu", "openai", "replicate", "modal"
    ] = "cpu",
    model: Literal[
        "gte-small", "gte-tiny", "bge-micro", "bge-micro-v2", None
    ] = "bge-micro",
    max_requests_per_minute: Optional[int] = None,
    max_tokens_per_minute: Optional[int] = None,
):
    """
    Initializes the embedding model based on the backend.

    :param backend: The backend used for embedding. Options: 'cpu', 'gpu', 'openai'.
    :type backend: str, default='auto'
    :raises ValueError: Raises an exception if the backend is 'openai' but openai_api_key is not set, or if the backend is unknown.

    Example:

    .. code-block:: python

        initialize_embedding_model('cpu')
    """
    if backend in ["cpu", "onnx"]:
        self.model = ONNXEmbeddingModel(model_name=model)
    elif backend in ["gpu", "ctranslate2"]:
        self.model = CT2EmbeddingModel(model_name=model)

    elif backend == "openai":
        if self.openai_api_key is None:
            raise ValueError(
                "You must set openai_api_key before calling get_embeddings() with openai backend."
            )
        max_rpm = max_requests_per_minute or self.max_requests_per_minute
        max_tpm = max_tokens_per_minute or self.max_tokens_per_minute
        self.model = OpenAIEmbeddingModel(
            openai_api_key=self.openai_api_key,
            max_requests_per_minute=max_rpm,
            max_tokens_per_minute=max_tpm,
        )
    elif backend == "replicate":
        self.model = ReplicateEmbeddingModel(
            replicate_api_key=self.replicate_api_key
        )
    elif backend == "modal":
        from .embedding_backends.modal_backend import ModalEmbeddingModel

        self.model = ModalEmbeddingModel(model_name=model)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_embeddings(
    self,
    input_field: str,
    embedding_field: str = "__embedding",
    model: Literal[
        "gte-small", "gte-tiny", "bge-micro", "bge-micro-v2", None
    ] = "bge-micro",
    backend: Literal[
        "cpu", "gpu", "openai", "replicate", "modal", "onnx", "ctranslate2"
    ] = "cpu",
    normalize: bool = False,
    pad: bool = False,
    split_strategy: Literal["truncate", "greedy", "even"] = "even",
):
    """
    Get embeddings for a field in the dataset.
    :param input_field: The field to get embeddings for.
    :type input_field: str
    :param embedding_field: The field to store the embeddings in.
    :type embedding_field: str, default='__embedding'
    :param model: The model to use for embeddings, if computing locally. Options: 'gte-small', 'gte-tiny', 'bge-micro'.

    .. code-block:: python

        # Replace "field_name1"
        ds.get_embeddings(field="field_name1")

    """
    # make sure the field doesn't exist already
    if embedding_field in self.dataset.column_names:
        raise ValueError(
            f"Field {embedding_field} already exists in dataset. Please choose a different name, or drop the column."
        )

    self.initialize_embedding_model(model=model, backend=backend)

    # if onnx, don't batch, it doesn't help
    if backend in ["cpu", "onnx"]:
        self.dataset = self.dataset.map(
            lambda x: {
                embedding_field: self.model.embed(
                    x[input_field],
                    normalize=normalize,
                    pad=pad,
                    split_strategy=split_strategy,
                ).tolist()
            }
        )
    # if openai, do one big batch and let our throttling handle rate limits
    elif backend == "openai":
        if normalize:
            logger.info("OpenAI embeddings are normalized by default.")
        else:
            logger.warning(
                "OpenAI embeddings are normalized by default, so normalize=False has no effect."
            )
        embs = self.model.embed_batch(
            self.dataset[input_field], normalize=normalize
        )
        self.dataset = self.dataset.add_column(embedding_field, embs)
    else:
        self.dataset = self.dataset.map(
            lambda x: {
                embedding_field: self.model.embed_batch(x[input_field])
            },
            batched=True,
            batch_size=4096,
        )
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
    """
    Get the nearest neighbors to a query.

    """
    if embedding_field not in self.dataset.column_names:
        raise ValueError(
            "You must call get_embeddings() before calling get_nearest_neighbors(). If your dataset already has an embeddings column and it's not '__embeddings', pass it as the 'embedding_field' argument."
        )

    emb_matrix = np.array(self.dataset[embedding_field])
    if not hasattr(self, "model"):
        self.initialize_embedding_model()
    if isinstance(query, str):
        query = self.model(query)
    scores = np.dot(emb_matrix, query)
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
