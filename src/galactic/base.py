# This file exists purely to define a base class that helps a lot with IDE autocompletion/type hints. Don't worry about it.

from abc import ABC, abstractmethod, abstractclassmethod
from typing import (
    TypeVar,
    Optional,
    Union,
    Dict,
    List,
    Callable,
    Literal,
    Sequence,
    Any,
)
import pandas as pd
import numpy as np
import datasets
from .utils import (
    MessageConfig,
    OPENAI_MESSAGE_CONFIG,
)

T = TypeVar("T", bound="GalacticDatasetBase")


class GalacticDatasetBase(ABC):
    def __init__(self):
        pass

    ## loaders
    @abstractclassmethod
    def from_csv(cls: T, path: str) -> T:
        pass

    @abstractclassmethod
    def from_jsonl(cls: T, path: str) -> T:
        pass

    @abstractclassmethod
    def from_parquet(cls: T, path: str) -> T:
        pass

    @abstractclassmethod
    def from_pandas(cls: T, df: pd.DataFrame) -> T:
        pass

    @abstractclassmethod
    def from_hugging_face(
        cls: T,
        path: str,
        split: str,
        config_name: Optional[str] = None,
        **kwargs,
    ) -> T:
        pass

    @abstractclassmethod
    def from_hugging_face_stream(
        cls: T,
        path: str,
        split: str,
        config_name: Optional[str] = None,
        filters: list[Callable[[dict], bool]] = None,
        dedup_fields: Optional[list[str]] = None,
        max_samples: Optional[int] = 200000,
        **kwargs,
    ) -> T:
        pass

    @abstractmethod
    def save(self: T, path: str) -> None:
        pass

    ## conversation utils
    @abstractmethod
    def conversation_from_dicts(
        self,
        input_column: str,
        input_message_config: Union[MessageConfig, Dict],
        output_column: Optional[str] = None,
        output_message_config: MessageConfig = OPENAI_MESSAGE_CONFIG,
        conversation_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> T:
        pass

    @abstractmethod
    def conversation_from_string(
        self,
        input_column: str,
        user_delimiter: str,
        assistant_delimiter: str,
        system_delimiter: str,
        output_column: Optional[str] = None,
        output_message_config: MessageConfig = OPENAI_MESSAGE_CONFIG,
        strip_whitespace: bool = True,
        replace_values: Optional[Dict[str, str]] = None,
    ) -> T:
        pass

    @abstractmethod
    def convert_conversation_to_string(
        self,
        input_column: str,
        input_message_config: Union[MessageConfig, Dict],
        output_column: str,
        user_delimiter: str,
        assistant_delimiter: str,
        system_delimiter: str,
        strip_whitespace: bool = True,
    ) -> T:
        pass

    @abstractmethod
    def get_conversation_length(
        self, input_column: str, output_column: str
    ) -> T:
        pass

    @abstractmethod
    def get_conversation_speakers(
        self,
        input_column: str,
        input_message_config: Union[MessageConfig, Dict],
        output_column: str = "__speakers",
    ) -> T:
        pass

    @abstractmethod
    def is_alternating(
        self,
        input_column: str,
        input_message_config: Union[MessageConfig, Dict],
        output_column: str = "__alternating",
    ) -> T:
        pass

    @abstractmethod
    def get_last_speaker(
        self,
        input_column: str,
        input_message_config: Union[MessageConfig, Dict],
        output_column: str = "__last_speaker",
    ) -> T:
        pass

    @abstractmethod
    def standardize_last_turn(
        self,
        input_column: str,
        input_message_config: Union[MessageConfig, Dict],
        last_speaker_role: Literal["user", "assistant", "system"],
        output_column: Optional[str] = None,
    ) -> T:
        pass

    @abstractmethod
    def get_shared_prefix(
        self,
        input_column: str,
        input_message_config: Union[MessageConfig, Dict],
        output_column: str = "__shared_prefix",
    ) -> T:
        pass

    @abstractmethod
    def add_initial_system_message(
        self: T,
        input_column: str,
        input_message_config: Union[MessageConfig, dict],
        system_message: str,
        output_column: Optional[str] = None,
    ) -> T:
        pass

    @abstractmethod
    def take_initial_system_message(
        self,
        input_column: str,
        input_message_config: Union[MessageConfig, dict],
        output_column: str,
        output_type: Literal["dict", "string"] = "string",
        remove_from_input: bool = True,
    ) -> T:
        pass

    @abstractmethod
    def take_last_message(
        self: T,
        input_column: str,
        input_message_config: Union[MessageConfig, dict],
        output_column: str,
        output_type: Literal["dict", "string"] = "string",
        remove_from_input: bool = True,
    ) -> T:
        pass

    ## filters
    # @abstractmethod
    # def filter(
    #     self: T,
    #     fn: Callable,
    # ) -> T:
    #     pass

    # @abstractmethod
    # def filter_batches(
    #     self: T,
    #     fn: Callable[[List[Dict]], List[bool]]
    # ):
    #     pass

    @abstractmethod
    def apply_bloom_filter(
        self: T, fields: Sequence[str], inplace: bool = True
    ) -> Optional[T]:
        pass

    @abstractmethod
    def filter_string(
        self: T,
        fields: Sequence[str],
        values: Sequence[str],
        inplace: bool = True,
    ) -> Optional[T]:
        pass

    @abstractmethod
    def filter_regex(
        self: T, fields: Sequence[str], regex: str, inplace: bool = True
    ) -> Optional[T]:
        pass

    ## taggers
    # @abstractmethod
    # def tag(
    #     self: T,
    #     fn: Callable,
    #     fields: Sequence[str],
    #     inplace: bool = True
    # ) -> Optional[T]:
    #     pass

    @abstractmethod
    def tag_string(
        self, fields: Sequence[str], values: Sequence[str], tag: str
    ) -> T:
        pass

    @abstractmethod
    def tag_regex(self, fields: Sequence[str], regex: str, tag: str) -> T:
        pass

    @abstractmethod
    def detect_language(
        self,
        field: str,
        method: Literal["fasttext", "langdetect"] = "fasttext",
    ) -> T:
        pass

    @abstractmethod
    def detect_pii(
        self: T,
        fields: Sequence[str],
    ) -> T:
        pass

    @abstractmethod
    def count_tokens(
        fields: Sequence[str], tokenizer: Optional[str] = None
    ) -> T:
        pass

    @abstractmethod
    def calc_perplexity(
        self,
        field: str,
        model: str = "kenlm",  # other option is pythia
        language: Optional[str] = "en",
        dataset: Optional[str] = "wikipedia",
    ) -> T:
        pass

    @abstractmethod
    def ai_tagger(
        self: T,
        field: str,
        tags: Union[List[str], Dict[str, str]],
        prompt: Optional[str] = None,
        backend="openai",
        allow_not_sure: bool = False,
    ) -> T:
        pass

    ## transforms
    @abstractmethod
    def trim_whitespace(
        self: T, fields: Sequence[str], inplace: bool = True
    ) -> Optional[T]:
        pass

    @abstractmethod
    def unicode_normalize(
        self: T, fields: Sequence[str], inplace: bool = True
    ) -> Optional[T]:
        pass

    @abstractmethod
    def ai_column(
        self: T, fields: Sequence[str], inplace: bool = True
    ) -> Optional[T]:
        pass

    ## classifiers
    @abstractmethod
    def ai_classifier(
        self,
        new_column: str,
        field: Optional[str],
        classes: Union[list[str], dict[str, str]],
        prompt: Optional[str] = None,
        backend: Literal["openai", "huggingface", "embeddings"] = "openai",
    ) -> T:
        pass

    @abstractmethod
    def train_fasttext_classifier(
        self,
        model_name: str,
        save_dir: str,
        input_field: str,
        label_field: str,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        normalize: list[str] = ["lower", "strip"],
        split_punctuation: bool = True,
        replace_newlines_with: str = " __newline__ ",
        target_model_size: str = "2M",
        training_duration: int = 300,
        random_seed: int = 42,
    ) -> None:
        pass

    @abstractmethod
    def train_embeddings_classifier(
        self,
        model_name: str,
        save_dir: str,
        label_field: str,
        model_type: str = "svm",
        input_field: str = "__embedding",
        validation_split=0.1,
        test_split=0.1,
        random_seed: int = 42,
    ) -> None:
        pass

    @abstractmethod
    def fasttext_classifier(
        self,
        new_column: str,
        model_path: str,
        field: str,
        # these should match how the model was trained
        normalize: list[str] = ["lower", "strip"],
        split_punctuation: bool = True,
        replace_newlines_with: str = " __newline__ ",
    ) -> T:
        pass

    @abstractmethod
    def embeddings_classifier(
        self, new_column: str, model_path: str, field: str = "__embedding"
    ) -> T:
        pass

    # ## embedding
    @abstractmethod
    def initialize_embedding_model(
        self: T, backend: Literal["auto", "cpu", "gpu", "openai"] = "auto"
    ) -> T:
        pass

    @abstractmethod
    def get_embeddings(
        self,
        input_field: str,
        embedding_field: str = "__embedding",
        backend: Literal["auto", "cpu", "gpu", "openai"] = "auto",
    ) -> T:
        pass

    @abstractmethod
    def get_nearest_neighbors(
        self,
        query: Union[str, np.ndarray, list[float]],
        k: int = 5,
        embedding_field: str = "__embedding",
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def reduce_embedding_dim(
        self,
        new_column: Optional[str] = None,
        method: Literal[
            "pca", "incremental_pca", "kernel_pca", "svd", "umap"
        ] = "pca",
        n_dims: int = 50,
        embedding_field: str = "__embedding",
        **kwargs,
    ) -> T:
        pass

    ## clustering
    @abstractmethod
    def cluster(
        self: T,
        n_clusters: Optional[int] = None,
        method: Literal[
            "kmeans", "minibatch_kmeans", "bisecting_kmeans", "hdbscan"
        ] = "kmeans",
        embedding_field: str = "__embedding",
        cluster_field: str = "__cluster",
        overwrite: bool = False,
        **kwargs,
    ):
        pass

    @abstractmethod
    def recompute_cluster_centers(
        self: T,
        cluster_field: str = "__cluster",
        embedding_field: str = "__embedding",
        method: Literal["centroid", "medoid"] = "centroid",
    ) -> T:
        pass

    @abstractmethod
    def _get_clusters(
        self,
        cluster_field: str = "__cluster",
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def remove_cluster(self: T, cluster_field: str, cluster_id: int) -> T:
        pass

    @abstractmethod
    def ai_label_clusters(
        self,
        new_column: str,
        context_fields: list[str],
        cluster_field: str = "__cluster",
        embedding_field: str = "__embedding",
        n_examples: int = 10,
        selection: Literal["random", "nearest"] = "random",
        prompt: Optional[str] = None,  # jinja2 template
    ) -> T:
        pass

    @abstractmethod
    def get_cluster_info(
        self,
        n_neighbors: int = 3,
        cluster_field: str = "__cluster",
        embedding_field: str = "__embedding",
        context_fields: list[str] = [],
        truncate_fields: int = 250,
        verbose: bool = True,
    ) -> T:
        pass

    @abstractmethod
    def semdedup(
        self,
        target_retention: Optional[float] = 0.8,
        threshold: Optional[float] = None,
        embedding_field: str = "__embedding",
        cluster_field: str = "__cluster",
        num_tuning_clusters: int = 3,
        dedup_strategy: Literal["random", "nearest", "furthest"] = "random",
        inplace=True,
    ):
        pass

    ## minhash lsh

    ## visualizations
    @abstractmethod
    def plot_embeddings(
        self,
        embedding_field: str,
        color_by: Optional[str] = None,
        save_path: Optional[str] = None,
        dot_size: int = 2,
        theme: Literal[
            "darkgrid", "whitegrid", "dark", "white", "ticks"
        ] = "whitegrid",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        pass
