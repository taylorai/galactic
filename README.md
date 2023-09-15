# Galactic

Cleaning and curation tools for massive unstructured text datasets.

To get started, install the package (`pip install galactic-ai`) and import it:

```python
from galactic import GalacticDataset
```

## Transforms

Transforms are operations that modify the data in some way. They are used in the GalacticDataset class to preprocess the data before it is used for training or inference. Here are the available transforms:

- Transform 1: This transform applies a specific operation to the data. It takes in a dataset and a function as parameters. The function is applied to each element in the dataset. The transformed dataset is then returned.
- Transform 2: This transform applies a specific operation to the data. It takes in a dataset and a function as parameters. The function is applied to each element in the dataset. The transformed dataset is then returned.

## Filters

Filters are operations that remove unwanted data based on certain criteria. They are used in the GalacticDataset class to clean the data before it is used for training or inference. Here are the available filters:

- Filter 1: This filter removes unwanted data from the dataset. It takes in a dataset and a function as parameters. The function is used to determine which elements in the dataset should be removed. The filtered dataset is then returned.
- Filter 2: This filter removes unwanted data from the dataset. It takes in a dataset and a function as parameters. The function is used to determine which elements in the dataset should be removed. The filtered dataset is then returned.

## Taggers

Taggers are operations that add tags to the data based on certain criteria. They are used in the GalacticDataset class to annotate the data before it is used for training or inference. Here are the available taggers:

- Tagger 1: This tagger adds tags to the data based on certain criteria. It takes in a dataset and a function as parameters. The function is used to determine which elements in the dataset should be tagged. The tagged dataset is then returned.
- Tagger 2: This tagger adds tags to the data based on certain criteria. It takes in a dataset and a function as parameters. The function is used to determine which elements in the dataset should be tagged. The tagged dataset is then returned.

## Loading Data

The src/galactic/loaders.py file contains methods for loading data into a GalacticDataset from various formats. Here are the available methods:

- CSV
  - `ds = GalacticDataset.from_csv("path/to/file.csv")`
  - This method loads data from a CSV file into a GalacticDataset. The `path` parameter specifies the path to the CSV file.
- JSONL
  - `ds = GalacticDataset.from_jsonl("path/to/file.jsonl")`
  - This method loads data from a JSONL file into a GalacticDataset. The `path` parameter specifies the path to the JSONL file.
- Pandas DataFrame
  - `ds = GalacticDataset.from_pandas(df)`
  - This method loads data from a Pandas DataFrame into a GalacticDataset. The `df` parameter is the DataFrame to load.
- Hugging Face Dataset
  - `ds = GalacticDataset.from_hugging_face("path/to/dataset", "split")`
  - This method loads data from a Hugging Face Dataset into a GalacticDataset. The `path` parameter specifies the path to the dataset and the `split` parameter specifies the split to load (e.g., "train", "test").
- Hugging Face Dataset Stream
  - `ds = GalacticDataset.from_hugging_face_stream("path/to/dataset", "split", filters=[filter1, filter2], dedup_fields=["field1", "field2"], max_samples=200000)`
  - This method loads data from a Hugging Face Dataset stream into a GalacticDataset. The `path` parameter specifies the path to the dataset, the `split` parameter specifies the split to load, the `filters` parameter is a list of filter functions to apply to the stream, the `dedup_fields` parameter is a list of fields to use for deduplication, and the `max_samples` parameter specifies the maximum number of samples to load.
- Disk
  - `ds = GalacticDataset.from_disk("path/to/dataset")`
  - This method loads data from a dataset saved on disk into a GalacticDataset. The `path` parameter specifies the path to the dataset.

### Preprocessing

- Trim whitespace
  - `ds.trim_whitespace(fields=["field1", "field2"])`
- Tag text on string
  - `ds.tag_string(fields=["field1"], values=["value1", "value2"], tag="desired_tag")`
- Tag text with RegEx
  - `ds.tag_regex(fields=["field1"], regex="some_regex", tag="desired_tag")`
- Filter on string
  - `ds.filter_string(fields=["field1"], values=["value1", "value2"])`
- Filter with RegEx
  - `ds.filter_regex(fields=["field1"], regex="some_regex")`

### Exploration

- Count tokens
  - `ds.count_tokens(fields=["text_field"])`
- Detect PII
  - `ds.detect_pii(fields=["name", "description"])`
- Detect the language
  - `ds.detect_language(field="text_field")`

### Manipulation

- Generate embeddings
  - `ds.get_embeddings(field="text_field")`
- Retrieve the nearest neighbors
  - `results = ds.get_nearest_neighbors(query="sample text", k=5)`
- Create clusters
  - `ds.cluster(n_clusters=5, method="kmeans")`
- Remove a cluster
  - `ds.remove_cluster(cluster=3)`
- Semantically Dedeplucate
  - `doc.semdedup(threshold=0.95)`

## Example

See `example.ipynb` for an example
