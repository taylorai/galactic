import re
import pathlib
from typing import Sequence, Callable, Optional
import numpy as np
import huggingface_hub
from transformers import AutoTokenizer
import scrubadub
from .utils import byte_len

import logging

logger = logging.getLogger("galactic")


def tag_string(self, fields: Sequence[str], values: Sequence[str], tag: str):
    # make sure the tag hasn't already been used
    if f"__tag__{tag}" in self.dataset.column_names:
        logger.warning(
            f"Tag {tag} already exists. This will overwrite the existing tag."
        )

    regexp = re.compile("|".join([re.escape(val) for val in values]))

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
    logger.info(
        f"Tagged dataset in-place with exact string matching on fields: {fields}"
    )
    # return self for chaining
    return self


def tag_regex(self, fields: Sequence[str], regex: str, tag: str):
    # make sure the tag hasn't already been used
    if f"__tag__{tag}" in self.dataset.column_names:
        logger.warning(
            f"Tag {tag} already exists. This will overwrite the existing tag."
        )

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
    logger.info(
        f"Tagged dataset in-place with regex matching on fields: {fields}"
    )
    return self


def detect_language(self, field: str):
    import fasttext

    model_path = huggingface_hub.hf_hub_download(
        repo_id="TaylorAI/galactic-models", filename="lid.176.ftz"
    )
    model = fasttext.load_model(model_path)

    def detect_(sample):
        if field in sample:
            if isinstance(sample[field], str):
                return {
                    "__language": model.predict(
                        sample[field].replace("\n", " ")
                    )[0][0].split("__label__")[1]
                }
            else:
                return {
                    "__language": model.predict(str(sample[field]))[0][
                        0
                    ].split("__label__")[1]
                }
        else:
            return {"__language": None}

    self.dataset = self.dataset.map(detect_)
    return self


def calc_perplexity(self, field: str):
    import ctranslate2

    repo_path = huggingface_hub.snapshot_download(
        "TaylorAI/galactic-models", allow_patterns="p70/*"
    )
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
    return self


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
            **{
                f"__pii__{f}": f in filth_types
                for f in ["email", "phone", "credential"]
            },
            "__pii__any": len(filth) > 0,
        }

    self.dataset = self.dataset.map(detect_)

    # no option to do out-of-place as this operation is not destructive
    return self


def count_tokens(self, fields: Sequence[str], tokenizer: Optional[str] = None):
    """
    Count the number of tokens in the specified fields.
    Args:
        fields (List[str]): List of fields to count tokens in.
        tokenizer (Callable): Tokenizer function to use. Defaults to None, which uses bytes.
    """
    if tokenizer is None:
        # count bytes in string
        count_fn = lambda x: byte_len(str(x))
        field_name = "__byte_count__"
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        count_fn = lambda x: len(tokenizer(str(x)).input_ids)
        field_name = "__token_count__"
    self.dataset = self.dataset.map(
        lambda x: {
            f"{field_name}{field}": count_fn(x[field]) for field in fields
        }
    )
    logger.info(f"Counted tokens in fields: {fields}")

    # no option to do out-of-place as this operation is not destructive
    return self
