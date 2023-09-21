from .galactic import GalacticDataset
from .loaders import (
    from_csv,
    from_jsonl,
    from_pandas,
    from_parquet,
    from_hugging_face,
    from_hugging_face_stream,
    save,
)
from .filters import (
    filter_string,
    filter_regex,
)
from .taggers import (
    tag_string,
    tag_regex,
    detect_language,
    detect_pii,
    detect_seo_spam,
    count_tokens,
    calc_perplexity,
    ai_tagger,
)
from .transforms import trim_whitespace, ai_column, ai_classifier
from .embedding import (
    get_embeddings,
    get_nearest_neighbors,
    get_embedding_model,
)
from .cluster import cluster, remove_cluster, get_cluster_info
from .semdedup import semdedup

from .minhash_lsh import compute_minhashes

# attach loaders to the class
GalacticDataset.from_csv = classmethod(from_csv)
GalacticDataset.from_jsonl = classmethod(from_jsonl)
GalacticDataset.from_pandas = classmethod(from_pandas)
GalacticDataset.from_parquet = classmethod(from_parquet)
GalacticDataset.from_hugging_face = classmethod(from_hugging_face)
GalacticDataset.from_hugging_face_stream = classmethod(
    from_hugging_face_stream
)
GalacticDataset.save = save

# attach filters to the class
GalacticDataset.filter_string = filter_string
GalacticDataset.filter_regex = filter_regex

# attach taggers to the class
GalacticDataset.tag_string = tag_string
GalacticDataset.tag_regex = tag_regex
GalacticDataset.detect_language = detect_language
GalacticDataset.detect_pii = detect_pii
GalacticDataset.count_tokens = count_tokens
GalacticDataset.calc_perplexity = calc_perplexity
GalacticDataset.detect_seo_spam = detect_seo_spam
GalacticDataset.ai_tagger = ai_tagger

# attach transforms to the class
GalacticDataset.trim_whitespace = trim_whitespace
GalacticDataset.ai_column = ai_column
GalacticDataset.ai_classifier = ai_classifier

# attach embedding to the class
GalacticDataset.get_embeddings = get_embeddings
GalacticDataset.get_nearest_neighbors = get_nearest_neighbors
GalacticDataset.get_embedding_model = get_embedding_model

# attach clustering to the class
GalacticDataset.cluster = cluster
GalacticDataset.remove_cluster = remove_cluster
GalacticDataset.get_cluster_info = get_cluster_info

# attach semantic deduplication to the class
GalacticDataset.semdedup = semdedup

# attach minhash lsh to the class
GalacticDataset.compute_minhashes = compute_minhashes
