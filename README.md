# Galactic

Cleaning and curation tools for massive unstructured text datasets.

To get started, install the package (`pip install galactic-ai`) and import it:

```python
from galactic import GalacticDataset
```

## Loading and Saving Data


### `from_csv`

```python
@classmethod
def from_csv(cls, path: str) -> 'GalacticDataset':
```

**Parameters:**

- `path (str)`: The path to the CSV file.

**Returns:**

- `GalacticDataset`: A dataset instance initialized from the CSV file.

**Example:**

```python
ds = GalacticDataset.from_csv("data.csv")
```

---

### `from_jsonl`

```python
@classmethod
def from_jsonl(cls, path: str, **kwargs) -> 'GalacticDataset':
```

**Parameters:**

- `path (str)`: The path to the JSONL file.
- `**kwargs`: Additional parameters passed to `datasets.load_dataset`.

**Returns:**

- `GalacticDataset`: A dataset instance initialized from the JSONL file.

**Example:**

```python
ds = GalacticDataset.from_jsonl("data.jsonl")
```

---

### `from_parquet`

```python
@classmethod
def from_jsonl(cls, path: str) -> 'GalacticDataset':
```

**Parameters:**

- `path (str)`: The path to the Parquet file.

**Returns:**

- `GalacticDataset`: A dataset instance initialized from the Parquet file.

**Example:**

```python
ds = GalacticDataset.from_parquet("data.parquet")

```

---

### `from_pandas`

```python
@classmethod
def from_pandas(cls, df, **kwargs) -> 'GalacticDataset':
```

**Parameters:**

- `df`: A Pandas DataFrame.
- `**kwargs`: Additional parameters passed to `datasets.Dataset.from_pandas`.

**Returns:**

- `GalacticDataset`: A dataset instance initialized from the DataFrame.

**Example:**

```python
import pandas as pd
df = pd.read_csv("data.csv")
ds = GalacticDataset.from_pandas(df)
```

---

### `from_hugging_face`

```python
@classmethod
def from_hugging_face(
   cls, 
   path: str, 
   split: str,  
   config_name: Optional[str] = None,
   **kwargs
) -> 'GalacticDataset':
```

**Parameters:**

- `path (str)`: The identifier of the Hugging Face dataset.
- `split (str)`: The desired split ('train', 'validation', 'test').
- `config_name (str, optional)`: Specific dataset configuration name. (For example, for C4, a config name like `en` or `realnewslike` is required).
- `**kwargs`: Additional parameters passed to `datasets.load_dataset`.

**Returns:**

- `GalacticDataset`: A dataset instance initialized from the Hugging Face dataset.

**Example:**

```python
ds = GalacticDataset.from_hugging_face("squad", split="train")
```

---

### `from_hugging_face_stream`

```python
@classmethod
def from_hugging_face_stream(
    cls,
    path: str,
    split: str,
    config_name: Optional[str] = None,
    filters: list[Callable[[dict], bool]] = [],
    dedup_fields: Optional[list[str]] = None,
    max_samples: Optional[int] = 200000,
    **kwargs
) -> 'GalacticDataset':
```

**Parameters:**

- `path (str)`: The identifier of the Hugging Face dataset.
- `split (str)`: The desired split ('train', 'validation', 'test').
- `config_name (str, optional)`: Specific dataset configuration name. (For example, for C4, a config name like `en` or `realnewslike` is required).
- `filters (list[Callable], optional)`: List of filter functions to apply.
- `dedup_fields (list[str], optional)`: Fields to check for duplicates.
- `max_samples (int, optional)`: Maximum number of samples to load, after filtering.
- `**kwargs`: Additional parameters passed to HuggingFace `datasets.load_dataset`.

**Returns:**

- `GalacticDataset`: A dataset instance initialized from the Hugging Face dataset.

**Example:**

```python
filters = [lambda x: x['field'] > 1]
ds = GalacticDataset.from_hugging_face_stream("squad", split="train", filters=filters)
```

---

### `save`

```python
def save(self, path: str, overwrite: bool = False) -> None:
```

**Parameters:**

- `path (str)`: The path to save the dataset to.
- `overwrite (bool, optional)`: Whether to overwrite the file if it already exists.

**Returns:**

- `None`

**Example:**

```python
ds.save("data.parquet")
```

---

## Filtering Data

### `apply_bloom_filter`

A Bloom filter is a memory-efficient, probabilistic data structure for exact-deduplication, which allows you to deduplicate with a single pass over the dataset rather than comparing all possible pairs. It guarantees there will be no false negatives--all actual duplicates *will* be removed. But there is a small probability of false positives, which means a small number of non-duplicates may be removed.

```python
def apply_bloom_filter(self, fields: Sequence[str], inplace: bool = True) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to apply the Bloom filter on. Two records will be considered duplicates if they match *all* of these fields.
- `inplace (bool, default=True)`: Whether to modify the dataset in-place.

**Returns:**

- `GalacticDataset`: Modified dataset with filtered records.

**Example:**

```python
ds.apply_bloom_filter(['text_field'])
```

---

### `filter_string`

```python
def filter_string(self, fields: Sequence[str], values: Sequence[str], inplace: bool = True) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to apply the filter on.
- `values (Sequence[str])`: List of string values to filter out.
- `inplace (bool, default=True)`: Whether to modify the dataset in-place.

**Returns:**

- `GalacticDataset`: Modified dataset with filtered records.

**Example:**

```python
ds.filter_string(['text_field'], ['exclude_this', 'and_this'])
```

---

### `filter_regex`

```python
def filter_regex(self, fields: Sequence[str], regex: str, inplace: bool = True) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to apply the regex-based filter on.
- `regex (str)`: The regex pattern to filter out.
- `inplace (bool, default=True)`: Whether to modify the dataset in-place.

**Returns:**

- `GalacticDataset`: Modified dataset with filtered records.

**Example:**

```python
ds.filter_regex(['text_field'], r'\d+')
```

---

Feel free to modify the descriptions and examples as you see fit.

---

## Processing Data


### `trim_whitespace`

```python
def trim_whitespace(self, fields: Sequence[str], inplace: bool = True) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to trim whitespace for.
- `inplace (bool, default=True)`: Whether to modify the dataset in-place.

**Returns:**

- `GalacticDataset`: Modified dataset with trimmed fields.

**Example:**

```python
ds.trim_whitespace(['text_field'])
```

---

### `tag_string`

```python
def tag_string(self, fields: Sequence[str], values: Sequence[str], tag: str) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to apply the tag.
- `values (Sequence[str])`: List of values to tag.
- `tag (str)`: The tag to be applied.

**Returns:**

- `GalacticDataset`: Modified dataset with new tags.

**Example:**

```python
ds.tag_string(['text_field'], ['value1', 'value2'], 'my_tag')
```

---

### `tag_regex`

```python
def tag_regex(self, fields: Sequence[str], regex: str, tag: str) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to apply the regex-based tag.
- `regex (str)`: The regex pattern.
- `tag (str)`: The tag to be applied.

**Returns:**

- `GalacticDataset`: Modified dataset with new tags.

**Example:**

```python
ds.tag_regex(['text_field'], r'\d+', 'contains_number')
```

---

### `detect_language`

```python
def detect_language(self, field: str) -> 'GalacticDataset':
```

**Parameters:**

- `field (str)`: Field to detect the language for.

**Returns:**

- `GalacticDataset`: Modified dataset with detected languages.

**Example:**

```python
ds.detect_language('text_field')
```

---

### `calc_perplexity`

```python
def calc_perplexity(self, field: str) -> 'GalacticDataset':
```

**Parameters:**

- `field (str)`: Field to calculate the perplexity for.

**Returns:**

- `GalacticDataset`: Modified dataset with calculated perplexities.

**Example:**

```python
ds.calc_perplexity('text_field')
```

---

### `detect_pii`

```python
def detect_pii(self, fields: Sequence[str]) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to detect PII in.

**Returns:**

- `GalacticDataset`: Modified dataset with detected PII.

**Example:**

```python
ds.detect_pii(['email_field', 'phone_field'])
```

---

### `count_tokens`

Counts tokens for each of the specified fields using the provided tokenizer (which is a string path to a Hugging Face tokenizer). If no tokenizer is provided, counts bytes instead.

```python
def count_tokens(self, fields: Sequence[str], tokenizer: Optional[str] = None) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to count tokens for.
- `tokenizer (str, optional)`: Tokenizer to use for token counting.

**Returns:**

- `GalacticDataset`: Modified dataset with token (or byte) counts.

**Example:**

```python
ds.count_tokens(['text_field'], tokenizer="some_tokenizer")
```
---

## Embedding and Clustering


### `get_embeddings`

```python
def get_embeddings(self, field: str, backend: str = "auto") -> 'GalacticDataset':
```

**Parameters:**

- `field (str)`: The field to create embeddings for.
- `backend (str, default='auto')`: The backend to use for generating embeddings. Currently, options are limited to "cpu" and "openai". If "auto", will use "cpu". If using "openai", you need to first set the `openai_api_key` attribute on the dataset.

**Returns:**

- `GalacticDataset`: Modified dataset with added embeddings.

**Example:**

```python
ds.get_embeddings('text_field')
```

---

### `get_nearest_neighbors`

```python
def get_nearest_neighbors(self, query: Union[str, np.ndarray], k: int = 5) -> pd.DataFrame:
```

**Parameters:**

- `query (str or np.ndarray)`: The query to find the nearest neighbors for.
- `k (int, default=5)`: Number of nearest neighbors to return.

**Returns:**

- `pd.DataFrame`: DataFrame containing the top-k nearest neighbors.

**Example:**

```python
ds.get_nearest_neighbors('sample query')
```

---

### `cluster`

```python
def cluster(self, n_clusters: int, method: str = "kmeans", batch_size: int = 1024, n_epochs: int = 5) -> None:
```

**Parameters:**

- `n_clusters (int)`: Number of clusters to form.
- `method (str, default='kmeans')`: Clustering method to use. Options are 'kmeans' or 'minibatch_kmeans'.
- `batch_size (int, default=1024)`: Batch size for 'minibatch_kmeans'.
- `n_epochs (int, default=5)`: Number of epochs for 'minibatch_kmeans'.

**Example:**

```python
ds.cluster(10)
```

---

### `get_cluster_info`

```python
def get_cluster_info(self) -> None:
```

**Description:**

- Provides information about the clusters, such as their sizes and prototypical examples.

**Example:**

```python
ds.get_cluster_info()
```

---

### `remove_cluster`

```python
def remove_cluster(self, cluster: int) -> None:
```

**Parameters:**

- `cluster (int)`: The cluster ID to remove.

**Example:**

```python
ds.remove_cluster(1)
```

---

Great, let's document the `semdedup` method for the `GalacticDataset` class.

---

### `semdedup`

```python
def semdedup(
    self,
    target_retention: Optional[float] = 0.8,
    threshold: Optional[float] = None,
    inplace: bool = True
) -> 'GalacticDataset':
```

**Parameters:**

- `target_retention (float, optional)`: The fraction of data points to retain after deduplication. If specified, the method will automatically tune the similarity on a few clusters, targeting this level of retention. Default is 0.8.
  
- `threshold (float, optional)`: The similarity threshold for marking duplicates (cosine similarity). Ignored if `target_retention` is specified.

- `inplace (bool, default=True)`: Whether to modify the dataset in-place or return a new one.

**Returns:**

- `GalacticDataset`: The dataset with semantic duplicates removed. Returns `self` if `inplace=True`.

**Raises:**

- `ValueError`: If neither `target_retention` nor `threshold` are specified.

**Example:**

```python
ds.semdedup(target_retention=0.8)
```
