# Galactic

Cleaning and curation tools for massive unstructured text datasets.

To get started, install the package (`pip install galactic-ai`) and import it:

```python
from galactic import GalacticDataset
```

## ✨ NEW! AI data labeling & classifier distillation ✨
See the [OpenHermes](https://github.com/taylorai/galactic/blob/main/examples/hermes.ipynb) notebook for a full example, where we use OpenAI to label a few thousand examples, distill a speedy classifier, and then use that classifier to label 100k+ examples in 1 minute.

### `set_openai_key`

```python
def set_openai_key(self, key: str) -> None:
```

Set the OpenAI API key for the dataset. This is required for using the `ai_column` and `ai_tagger` methods, or OpenAI embeddings. (You can compute local embeddings on CPU without it.)

### `set_rate_limits`


```python
def set_rate_limits(self, max_requests_per_minute: int, max_tokens_per_minute: int) -> None:
```

Set the rate limits for the dataset. This is only needed if you have higher rate limits than the default (generally 3000 requests per minute and 350000 tokens per minute, but it depends on the model). It's not a huge deal because we throttle requests to try to stay under these, but if you set them too high, there will be lots of pauses, and if they're too low you aren't taking full advantage of your rate limits.

### `ai_column`

```python
def ai_column(
    self,
    new_column: str,
    prompt: str,
    depends_on=list[str],
    normalize: list[str] = ["strip"],  # must be methods of str class
    system_prompt: Optional[str] = None
) -> 'GalacticDataset':
```

**Parameters:**

- `new_column (str)`: The name of the new column to create.
- `prompt (str)`: The prompt to use for the AI model. Should be a Jinja2 template, with {{ placeholder }} for each of the fields in `depends_on`.
- `depends_on (list[str])`: The fields to use as inputs to the prompt.
- `normalize (list[str], default=['strip', 'lower'])`: The normalization methods to apply to the output of the AI model.
- `system_prompt (str, optional)`: The system prompt to use. If not specified, will not use any system prompt.

**Returns:**

- `GalacticDataset`: The dataset with the new column added.


### `ai_classifier`

```python
def ai_classifier(
    self,
    new_column: str,
    field: Optional[str],
    classes: Union[list[str], dict[str, str]],
    prompt: Optional[str] = None,
    backend="openai",
)
```

**Parameters:**

- `new_column (str)`: The name of the new column to create.
- `field (str, optional)`: The field to use as input to the AI model. Can be None if using embedding similarity as the backend.
- `classes (list[str] or dict[str, str])`: The classes to use for the classifier. Can be just a list of labels, or a dict of label: description. If provided, descriptions will be used for a) prompting API model if backend = 'openai', b) embedding if backend = "embeddings", c) zero-shot classification if backend = "huggingface".
- `prompt` (str): Prompt template (Jinja2) to use for the API request, with a template slot for the field to classify. Only used if backend = 'openai'. If None, a basic default prompt will be used.
- `backend` (str, default='openai'): The backend to use for the classifier. Options are 'openai', 'embeddings', and 'huggingface'. If "openai", will make API requests with logit bias to force the model to choose a valid category. If "embeddings", will use cosine similarity to the class embeddings. If "huggingface", will use zero-shot classification with a small HuggingFace model.

### `ai_tagger`

```python
def ai_tagger(
    self,
    field: str,
    tags: Union[list[str], dict[str, str]],
    prompt: Optional[str] = None,
    backend="openai",
    allow_not_sure: bool = False,
) -> 'GalacticDataset':
```

**Parameters:**

- `field (str)`: The field to use as input to the AI model.
- `tags (list[str] or dict[str, str])`: The tags to use for the tagger. Can be just a list of labels, or a dict of label: description. If provided, descriptions will be used for a) prompting API model if backend = 'openai', b) embedding if backend = "embeddings", c) zero-shot classification if backend = "huggingface".
- `prompt` (str): Prompt template (Jinja2) to use for the API request, with a template slot for the field to classify.
- `backend` (str, default='openai'): The backend to use for the classifier. Only "openai" is currently supported.
- `allow_not_sure` (bool, default=False): Whether to allow the model to select "not sure" when deciding whether a tag applies. Can be useful if you want to consider the model's uncertainty rather than forcing it to make a decision.

**Returns:**

- `GalacticDataset`: The dataset with the new columns added for each tag.

### `train_fasttext_classifier`

```python
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
```

**Parameters:**

- `model_name (str)`: A name for the new model (will be used for saving).
- `save_dir (str)`: The directory to save the model to.
- `input_field (str)`: The field to use as input to the classifier.
- `label_field (str)`: The field to use as labels for the classifier.
- `validation_split (float, default=0.1)`: The fraction of the dataset to use for validation (hyperparameter search).
- `test_split (float, default=0.1)`: The fraction of the dataset to use for testing.
- `normalize (list[str], default=['lower', 'strip'])`: The normalization methods to apply to the input field.
- `split_punctuation (bool, default=True)`: Whether to split punctuation into separate tokens.
- `replace_newlines_with (str, default=' __newline__ ')`: The string to replace newlines with (FastText doesn't allow newlines).
- `target_model_size (str, default='2M')`: The target model size (in bytes). You can put things like, "500K", "1M", etc.
- `training_duration (int, default=300)`: The training duration in seconds (more time means more trials with different hyperparameters).
- `random_seed (int, default=42)`: The random seed to use for reproducibility.

### `fasttext_classifier`

```python
def fasttext_classifier(
    self,
    new_column: str,
    model_path: str,
    field: str,
    normalize: list[str] = ["lower", "strip"],
    split_punctuation: bool = True,
    replace_newlines_with: str = " __newline__ ",
) -> 'GalacticDataset':
```

**Parameters:**

- `new_column (str)`: The name of the new column to create.
- `model_path (str)`: The path to the FastText model saved by `train_fasttext_classifier`. Should include the extension (.ftz).
- `field (str)`: The field to use as input to the classifier.
- `normalize (list[str], default=['lower', 'strip'])`: The normalization methods to apply to the input field. Should match the normalization methods used when training the model.
- `split_punctuation (bool, default=True)`: Whether to split punctuation into separate tokens. Should match the setting used when training the model.
- `replace_newlines_with (str, default=' __newline__ ')`: The string to replace newlines with (FastText doesn't allow newlines). Should match the setting used when training the model.

### `train_embeddings_classifier`

```python
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
```

**Parameters:**

- `model_name (str)`: A name for the new model (will be used for saving).
- `save_dir (str)`: The directory to save the model to.
- `label_field (str)`: The field to use as labels for the classifier.
- `model_type (str, default='svm')`: The type of classifier to use. Options are 'svm' and 'logistic_regression'.
- `input_field (str, default='__embedding')`: The field to use as input to the classifier. Should be the embedding (or another list of floats, if you have that around for some reason).
- `validation_split (float, default=0.1)`: The fraction of the dataset to use for validation (early stopping).
- `test_split (float, default=0.1)`: The fraction of the dataset to use for testing.
- `random_seed (int, default=42)`: The random seed to use for reproducibility.

### `embeddings_classifier`

```python
def embeddings_classifier(
    self,
    new_column: str,
    model_path: str,
    field: str = "__embedding",
):
```

**Parameters:**

- `new_column (str)`: The name of the new column to create.
- `model_path (str)`: The path to the embeddings classifier model saved by `train_embeddings_classifier`. Just use the model name with no extension, since we have to load both the model and the label encoder.
- `field (str, default='__embedding')`: The field to use as input to the classifier. Should be the embedding (or another list of floats, if you have that around for some reason). Has to match the features used to train the model.



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


---

### `from_jsonl`

```python
@classmethod
def from_jsonl(cls, path: str) -> 'GalacticDataset':
```

**Parameters:**

- `path (str)`: The path to the JSONL file.

**Returns:**

- `GalacticDataset`: A dataset instance initialized from the JSONL file.
- 
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

---

### `from_pandas`

```python
@classmethod
def from_pandas(cls, df) -> 'GalacticDataset':
```

**Parameters:**

- `df`: A Pandas DataFrame.

**Returns:**

- `GalacticDataset`: A dataset instance initialized from the DataFrame.

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


---

### `from_hugging_face_stream`

```python
@classmethod
def from_hugging_face_stream(
    cls,
    path: str,
    split: str,
    config_name: Optional[str] = None,
    filters: list[Callable[[dict], bool]] = None,
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
filters = [lambda x: len(x['text']) > 1000]
ds = GalacticDataset.from_hugging_face_stream("c4", split="train", config_name="en", filters=filters)
```

---

### `save`

Saves the dataset to JSONL or CSV.

```python
def save(self, path: str, overwrite: bool = False) -> None:
```

**Parameters:**

- `path (str)`: The path to save the dataset to (must be .jsonl or .csv).
- `overwrite (bool, optional)`: Whether to overwrite the file if it already exists.

**Returns:**

- `None`


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
By default, uses KenLM trained on English Wikipedia. To use a different KenLM model from [this repo](https://huggingface.co/edugp/kenlm/), you can set the 'language' and 'dataset' parameters. To use Pythia-70m, set the 'model' parameter to 'pythia'.

```python
def calc_perplexity(
    self, 
    field: str,
    model: str = "kenlm",  # other option is pythia
    language: Optional[str] = "en",
    dataset: Optional[str] = "wikipedia",
) -> 'GalacticDataset':
```

**Parameters:**

- `field (str)`: Field to calculate the perplexity for.

**Returns:**

- `GalacticDataset`: Modified dataset with calculated perplexities.


---

### `detect_pii`

```python
def detect_pii(self, fields: Sequence[str]) -> 'GalacticDataset':
```

**Parameters:**

- `fields (Sequence[str])`: List of fields to detect PII in.

**Returns:**

- `GalacticDataset`: Modified dataset with detected PII.

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


---

### `get_nearest_neighbors`

```python
def get_nearest_neighbors(self, query: Union[str, np.ndarray], k: int = 5) -> list[dict]:
```

**Parameters:**

- `query (str or np.ndarray)`: The query to find the nearest neighbors for.
- `k (int, default=5)`: Number of nearest neighbors to return.

**Returns:**

- `list[dict]` of nearest neighbors by cosine similarity.


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

---

### `get_cluster_info`

```python
def get_cluster_info(self) -> None:
```

**Description:**

- Provides information about the clusters: their sizes and a few prototypical examples.


---

### `remove_cluster`

Removes a cluster from the dataset. Preferred to just "filtering" because it also removes the cluster from the cluster info stored in the dataset object.

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


