import re
import datasets
import pybloom_live
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Sequence
from .embedding import EmbeddingModel
import networkx as nx
from sklearn.cluster import MiniBatchKMeans, KMeans
from transformers import AutoTokenizer
import scrubadub
import fasttext
import ctranslate2
import huggingface_hub
import pathlib

# set up logging
import logging
logging.basicConfig(level=logging.INFO)

# helper: length in utf-8 bytes
def byte_len(x):
    return len(x.encode("utf-8"))

class GalacticDataset():
    def __init__(self, dataset: datasets.Dataset):
        super().__init__()
        self.dataset = dataset
        # add unique increaing int __id field
        self.dataset = self.dataset.map(
            lambda x, i: {"__id": i},
            with_indices=True,
        )


    @classmethod
    def from_hugging_face(
        cls, 
        path: str,
        split: str,
        **kwargs
    ):
        ds = datasets.load_dataset(path, split=split, **kwargs)
        return GalacticDataset(ds)
    
    @classmethod
    def from_hugging_face_stream(
        cls,
        path: str,
        split: str,
        dedup_fields: list[str] = None,
        max_samples: int = 200000,
        filter_min_lengths: dict[str, int] = None,
        filter_max_lengths: dict[str, int] = None,
        **kwargs
    ):
        handle = datasets.load_dataset(path, split=split, streaming=True, **kwargs)
        bloom = pybloom_live.BloomFilter(capacity=max_samples, error_rate=0.001)
        samples = []
        progress = tqdm(total=max_samples, desc=f"Loading samples from {path}")
        for idx, sample in enumerate(handle):
            if filter_min_lengths is not None:
                if any([byte_len(str(sample[k])) < filter_min_lengths[k] for k in filter_min_lengths]):
                    continue
            if filter_max_lengths is not None:
                if any([byte_len(str(sample[k])) > filter_max_lengths[k] for k in filter_max_lengths]):
                    continue

            if dedup_fields is not None and len(dedup_fields) > 0:
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
            msg += f"Removed {idx - len(samples)} samples that were duplicated, too long, or too short. "
        logging.info(msg)
        return GalacticDataset(datasets.Dataset.from_list(samples))
    
    @classmethod
    def from_disk(
        cls,
        path: str,
        **kwargs
    ):
        ds = datasets.Dataset.load_from_disk(path, **kwargs)
        return GalacticDataset(ds)
    
    def __repr__(self):
        return pd.DataFrame(self.dataset.select(range(5))).__repr__()
    
    def __str__(self):
        return self.__repr__()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def trim_whitespace(self, fields: Sequence[str]):
        """
        Trim Unicode-defined whitespace at the beginning and end of specified fields.
        Args:
            fields (List[str]): List of fields to trim.
        """
        def trim_(sample):
            for field in fields:
                if field in sample:
                    sample[field] = sample[field].strip()
            return sample
        self.dataset = self.dataset.map(trim_)
        logging.info(f"Trimmed whitespace for fields: {fields}")

    def tag_string(self, fields: Sequence[str], values: Sequence[str], tag: str):
        # make sure the tag hasn't already been used
        if f"__tag__{tag}" in self.dataset.column_names:
            logging.warning(f"Tag {tag} already exists. This will overwrite the existing tag.")
        
        regexp = re.compile("|".join(values))
        def tag_(sample):
            for field in fields:
                if field in sample:
                    if isinstance(sample[field], str):
                        if regexp.search(sample[field]):
                            return {f"__tag__{tag}": True}
                    else:
                        if regexp.search(str(sample[field])):
                            return {f"__tag__{tag}": True}
            return {f"__tag__{tag}": False}
        self.dataset = self.dataset.map(tag_)
        logging.info(f"Tagged dataset based on provided string values in fields: {fields}")
    
    def tag_regex(self, fields: Sequence[str], regex: str, tag: str):
        # make sure the tag hasn't already been used
        if f"__tag__{tag}" in self.dataset.column_names:
            logging.warning(f"Tag {tag} already exists. This will overwrite the existing tag.")
        
        regexp = re.compile(regex)
        def tag_(sample):
            for field in fields:
                if field in sample:
                    if isinstance(sample[field], str):
                        if regexp.search(sample[field]):
                            return {f"__tag__{tag}": True}
                    else:
                        if regexp.search(str(sample[field])):
                            return {f"__tag__{tag}": True}
            return {f"__tag__{tag}": False}
        self.dataset = self.dataset.map(tag_)
        logging.info(f"Tagged dataset based on provided regex in fields: {fields}")

    def filter_string(self, fields: Sequence[str], values: Sequence[str]):
        """
        Filter data containing particular string or list of strings in the specified field.
        Args:
            field (str): The field to check against.
            values (List[str]): List of strings to filter on.

        """
        # make a regexp that matches any of the fields
        regexp = re.compile("|".join(values))

        def filter_(sample):
            for field in fields:
                if field in sample:
                    if isinstance(sample[field], str):
                        if regexp.search(sample[field]):
                            return False
                    else:
                        if regexp.search(str(sample[field])):
                            return False
            return True
        
        self.dataset = self.dataset.filter(filter_)
        logging.info(f"Filtered dataset based on provided string values in fields: {fields}")

    def filter_regex(self, fields: Sequence[str], regex: str):
        """
        Filter data containing particular regex in the specified field.
        Args:
            fields Sequence[str]: The field to check against.
            regex (str): The regex to filter on.

        """
        # make a regexp that matches any of the fields
        regexp = re.compile(regex)

        def filter_(sample):
            for field in fields:
                if field in sample:
                    if isinstance(sample[field], str):
                        if regexp.search(sample[field]):
                            return False
                    else:
                        if regexp.search(str(sample[field])):
                            return False
            return True

        self.dataset = self.dataset.filter(filter_)
        logging.info(f"Filtered dataset based on provided regex in fields: {fields}")

    def count_tokens(self, fields: Sequence[str], tokenizer = None):
        """
        Count the number of tokens in the specified fields.
        Args:
            fields (List[str]): List of fields to count tokens in.
            tokenizer (Callable): Tokenizer function to use. Defaults to None, which uses bytes.
        """
        if tokenizer is None:
            # count bytes in string
            count_fn = lambda x: byte_len(str(x))
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            count_fn = lambda x: len(tokenizer(str(x)).input_ids)
        self.dataset = self.dataset.map(
            lambda x: {f"__token_count_{field}": count_fn(x[field]) for field in fields}
        )
        logging.info(f"Counted tokens in fields: {fields}")
    
    def detect_pii(self, fields: Sequence[str]):
        """
        Detect personally identifiable information in the specified fields.
        Args:
            fields (List[str]): List of fields to detect PII in.
            Currently only supports "email", "phone", and "credential".
        """
        def detect_(sample):
            filth = []
            for field in fields:
                if field in sample:
                    if isinstance(sample[field], str):
                        filth.extend(scrubadub.list_filth(sample[field]))
                    else:
                        filth.extend(scrubadub.list_filth(str(sample[field])))
            filth_types = [f.detector_name for f in filth]
            return {
                **{f"__pii__{f}": f in filth_types for f in ["email", "phone", "credential"]},
                "__pii__any": len(filth) > 0,
            }
        self.dataset = self.dataset.map(detect_)

    def detect_language(self, field: str):
        model_path = huggingface_hub.hf_hub_download(
            repo_id="TaylorAI/galactic-models",
            filename="lid.176.ftz"
        )
        model = fasttext.load_model(model_path)
        def detect_(sample):
            if field in sample:
                if isinstance(sample[field], str):
                    return {"__language": model.predict(sample[field].replace("\n", " "))[0][0].split("__label__")[1]}
                else:
                    return {"__language": model.predict(str(sample[field]))[0][0].split("__label__")[1]}
            else:
                return {"__language": None}
        self.dataset = self.dataset.map(detect_)

    def calc_perplexity(self, field: str):
        repo_path = huggingface_hub.snapshot_download("TaylorAI/galactic-models", allow_patterns="p70/*")
        model_path = pathlib.Path(repo_path) / "p70"
        model = ctranslate2.Generator(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        def calc_(sample):
            if field in sample:
                if isinstance(sample[field], str):
                    token_ids = tokenizer(sample[field]).input_ids
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    log_probs = model.score_batch([tokens])[0].log_probs
                    ppl = np.exp(-np.sum(log_probs) / byte_len(sample[field]))
                    return {"__perplexity": ppl}
                else:
                    return {"__perplexity": None}
            else:
                return {"__perplexity": None}  

        self.dataset = self.dataset.map(calc_) 

    def get_embeddings(self, field: str, backend: str = "onnx"):
        model_path = huggingface_hub.hf_hub_download("TaylorAI/galactic-models", filename="model_quantized.onnx")
        tokenizer_path = "Supabase/gte-small"
        max_length = 512
        model = EmbeddingModel(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            model_type="onnx",
            max_length=max_length
        )
        self.model = model
        self.dataset = self.dataset.map(
            lambda x: {f"__embedding": model(x[field])}
        )

        self.dataset.add_faiss_index(column="__embedding")
        logging.info(f"Created embeddings & FAISS index on field {field}")

    def get_nearest_neighbors(self, query: str, k: int = 5):
        if not hasattr(self, "model"):
            raise ValueError("You must call get_embeddings() before calling get_nearest_neighbors()")
        query_embedding = self.model.predict(query)
        results = self.dataset.get_nearest_examples("__embedding", query_embedding, k=k)
        return results
    
    def cluster(self, 
            n_clusters: int, 
            method: str = "kmeans",
            batch_size: int = 1024, 
            n_epochs: int = 5
        ):
        
        if method == "minibatch_kmeans":
            model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
            for epoch in range(n_epochs):
                print(f"Epoch {epoch + 1}/{n_epochs}")
                self.dataset.map(
                    lambda x: model.partial_fit(np.array(x["__embedding"])),
                )
            self.cluster_centers = model.cluster_centers_
        elif method == "kmeans":
            model = KMeans(n_clusters=n_clusters)
            arr = np.array(self.dataset["__embedding"])
            print(arr.shape)
            model.fit(arr)
            self.cluster_centers = model.cluster_centers_
            self.n_clusters = n_clusters
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # add new column with cluster labels
        self.dataset = self.dataset.map(
            lambda x: {"__cluster": model.predict(x["__embedding"])},
            batched=True,
        )
    
    def remove_cluster(self, cluster: int):
        self.dataset = self.dataset.filter(lambda x: x["__cluster"] != cluster)
    
    def semdedup(self, threshold: float = 0.95):
        for cluster in tqdm(range(self.n_clusters), desc="Semantic deduplication"):
            cluster = self.dataset.filter(lambda x: x["__cluster"] == cluster)
            # create n x n matrix of cosine similarities
            emb_matrix = np.array(cluster["__embedding"])
            similarities = np.dot(emb_matrix, emb_matrix.T)
            centroid_sims = np.mean(similarities, axis=0)
            G = nx.Graph()
            num_points = len(similarities)
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    if similarities[i][j] > threshold:
                        G.add_edge(i, j)
            connected_components = list(nx.connected_components(G))

            # keep the point closest to the centroid among duplicates
            to_remove = set()
            for component in connected_components:
                closest_to_centroid = min(component, key=lambda x: centroid_sims[x])
                component.remove(closest_to_centroid)
                to_remove.update(component)
            

    # finetune embeddings as in https://github.com/openai/openai-cookbook/blob/main/examples/Customizing_embeddings.ipynb
    def tune_embeddings(self):
        pass




        
    
