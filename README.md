# Galactic
Cleaning and curation tools for massive unstructured text datasets

## Getting Started
1. Clone the repo: `git clone https://github.com/taylorai/docent.git`
3. Create a new Jupyter notebook
4. Install the dependencies: `!pip install -r requirements.txt`
5. Import Docent: `from dataset import Docent`
6. Load your dataset: `ds = Docent.from_disk("c4-with_embs")`

## Utilities
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
