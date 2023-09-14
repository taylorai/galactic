from .galactic import GalacticDataset
from .loaders import (
    from_csv,
    from_jsonl,
    from_pandas,
    from_hugging_face,
    from_hugging_face_stream,
)
from .filters import (
    filter_string,
    filter_regex,
)
from .taggers import (
    tag_string,
    tag_regex,
)

# attach loaders to the class
GalacticDataset.from_csv = classmethod(from_csv)
GalacticDataset.from_jsonl = classmethod(from_jsonl)
GalacticDataset.from_pandas = classmethod(from_pandas)
GalacticDataset.from_hugging_face = classmethod(from_hugging_face)
GalacticDataset.from_hugging_face_stream = classmethod(from_hugging_face_stream)

# attach filters to the class
GalacticDataset.filter_string = filter_string
GalacticDataset.filter_regex = filter_regex

# attach taggers to the class
GalacticDataset.tag_string = tag_string
GalacticDataset.tag_regex = tag_regex
