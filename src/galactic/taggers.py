import os
import re
import pathlib
from typing import Sequence, Optional, Union
import numpy as np
import huggingface_hub
from transformers import AutoTokenizer
import scrubadub
import tiktoken
import jinja2
import time

from .utils import byte_len
from .async_openai import run_chat_queries_with_openai

import logging

logger = logging.getLogger("galactic")


def tag_string(self, fields: Sequence[str], values: Sequence[str], tag: str):
    """Tag data containing a particular string or list of strings in the specified field."""
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
            f"Tag already exists. This will overwrite the existing tag."
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
        f"Tagged dataset in-place with tag '__tag__{tag}', using regex matching on fields: {fields}"
    )
    return self


def detect_language(self, field: str):
    """Detect the language of the specified field."""
    # make sure field exists
    if field not in self.dataset.features:
        raise ValueError(f"Field {field} not found in dataset.")
    import fasttext

    model_path = huggingface_hub.hf_hub_download(
        repo_id="TaylorAI/galactic-models", filename="lid.176.ftz"
    )
    model = fasttext.load_model(model_path)

    def detect_(sample):
        if isinstance(sample[field], str):
            return {
                "__language": model.predict(sample[field].replace("\n", " "))[
                    0
                ][0].split("__label__")[1]
            }
        else:
            return {
                "__language": model.predict(str(sample[field]))[0][0].split(
                    "__label__"
                )[1]
            }

    self.dataset = self.dataset.map(detect_)
    logger.info(
        f"Detected language in field {field}, added language metadata to '__language'."
    )
    return self


def calc_perplexity(
    self,
    field: str,
    model: str = "kenlm",  # other option is pythia
    language: Optional[str] = "en",
    dataset: Optional[str] = "wikipedia",
):
    """
    Calculate the perplexity-per-byte of the specified field.

    .. note::

        You must install KenLM to calculate perplexity using the KenLM model from https://github.com/kpu/kenlm.
        To install, run ``pip install https://github.com/kpu/kenlm/archive/master.zip``.

    .. code-block:: python

        # Replace "field_name1"
        ds.calc_perplexity(field="field_name1")

    """
    # make sure field exists and is a string field
    if field not in self.dataset.features:
        raise ValueError(f"Field {field} not found in dataset.")
    elif self.dataset.features[field].dtype != "string":
        raise ValueError(
            f"Field {field} is not a string field, and so can't be used to calculate perplexity."
        )
    if model == "pythia":
        import ctranslate2

        repo_path = huggingface_hub.snapshot_download(
            "TaylorAI/galactic-models", allow_patterns="p70/*"
        )
        model_path = pathlib.Path(repo_path) / "p70"
        model = ctranslate2.Generator(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

        def calc_(sample):
            token_ids = tokenizer(sample[field]).input_ids
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            log_probs = model.score_batch([tokens])[0].log_probs
            ppl = np.exp(-np.sum(log_probs) / byte_len(sample[field]))
            return {"__perplexity": ppl}

    elif model == "kenlm":
        try:
            from .kenlm import KenlmModel
        except ImportError:
            raise ImportError(
                "KenLM is not installed. Install with 'pip install https://github.com/kpu/kenlm/archive/master.zip'."
            )
        if language is None or dataset is None:
            raise ValueError(
                "Must specify language (e.g. 'en') and dataset (e.g. 'wikipedia') for KenLM. See options here: https://huggingface.co/edugp/kenlm/tree/main"
            )
        model = KenlmModel.from_pretrained(dataset, language)

        def calc_(sample):
            ppl = model.get_perplexity(sample[field])
            return {"__perplexity": ppl}

    else:
        raise ValueError(
            f"Model {model} not supported. Supported models: 'kenlm', 'pythia'."
        )

    self.dataset = self.dataset.map(calc_)
    logger.info(
        f"Calculated perplexity-per-byte in field {field}, added perplexity metadata to '__perplexity'."
    )
    return self


def detect_pii(self, fields: Sequence[str]):
    """
    Detect personally identifiable information in the specified fields.
    Args:
        fields (List[str]): List of fields to detect PII in.
        Currently only supports "email", "phone", and "credential".
    """
    # make sure all the fields exist
    for field in fields:
        if field not in self.dataset.features:
            raise ValueError(f"Field {field} not found in dataset.")

    def detect_(sample):
        text = " ".join([str(sample[field]) for field in fields])
        filth = scrubadub.list_filth(text)
        filth_types = [f.detector_name for f in filth]
        return {
            **{
                f"__pii__{f}": f in filth_types
                for f in ["email", "phone", "credential"]
            },
            "__pii__any": len(filth) > 0,
        }

    self.dataset = self.dataset.map(detect_)
    logger.info(
        f"Detected PII in fields: {fields}; added __pii__email, __pii__phone, __pii__credential, and __pii__any metadata."
    )
    # no option to do out-of-place as this operation is not destructive
    return self


def count_tokens(self, fields: Sequence[str], tokenizer: Optional[str] = None):
    """
    Count the number of tokens in the specified fields.

    .. code-block:: python

        # Replace "field_name1" and "field_name2" with your fields
        ds.count_tokens(fields=["field_name1, field_name2"])

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
    logger.info(
        f"Counted tokens in fields: {fields}, added metadata to {field_name}"
    )

    # no option to do out-of-place as this operation is not destructive
    return self


def ai_tagger(
    self,
    field: str,
    tags: Union[list[str], dict[str, str]],
    prompt: Optional[str] = None,
    backend="openai",
    allow_not_sure: bool = False,
):
    """
    Use OpenAI's API or a HF zero-shot classifier to generate a new column based on an existing column.
    Args:
        name (str): Name of the new column.
        field (str): Name of the field to use for classification. Can be None if backend is 'embeddings'.
        classes (List[str] or dict[str, str]): Classes to classify into. Can be just a list of labels, or a dict of label: description.
        If provided, descriptions will be used for a) prompting API model if backend = 'openai', b) embedding if backend = "embeddings",
        c) zero-shot classification if backend = "huggingface".
        backend (str): Which backend to use. Currently supports 'embeddings', 'openai' and 'huggingface'.
    """
    if backend != "openai":
        raise NotImplementedError(
            "Only OpenAI API is currently supported for AI tagging."
        )

    elif backend == "openai":
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", None)
            if self.openai_api_key is None:
                raise ValueError(
                    "No OpenAI API key found. Set OPENAI_API_KEY environment variable, or set the openai_api_key attribute of the GalacticDataset."
                )
        # make sure field exists
        if field not in self.dataset.column_names:
            raise ValueError(f"Field {field} not found in dataset.")

        # create logit bias (same for each tag) -- model can only output True or False
        token2cls = {}
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        classes = ["True", "False"]
        if allow_not_sure:
            classes.append("Not sure")
        for cls in classes:
            t1 = tokenizer.encode(cls)[0]
            t2 = tokenizer.encode(" " + cls)[0]
            if t1 in token2cls or t2 in token2cls:
                raise ValueError(
                    "Class names must not share a prefix for logit bias trick to work. Try changing the labels and see if that helps."
                )
            else:
                token2cls[t1] = cls
                token2cls[t2] = cls
        logit_bias = {t: 100 for t in token2cls.keys()}

        for tag in tags:
            logger.info(f"Tagging with tag {tag}...")
            # construct prompt template
            if prompt is None:
                prompt_template = (
                    f"Tag the provided text with the following tag:\n"
                )
                if isinstance(tags, list):
                    prompt_template += f"  - {tag}\n"
                else:
                    prompt_template += f"  - {tag}: {tags[tag]}\n"
                prompt_template += "Answer True if the tag applies to the text, and False if it does not. "
                if allow_not_sure:
                    prompt_template += "If you're not sure, answer Not sure."
                prompt_template += (
                    "\n\n---\n\nText: {{"
                    + field
                    + "}}\n\n---\n\nDoes the tag apply?"
                )
            else:
                prompt_template = prompt
            prompt_template = jinja2.Template(prompt_template)
            prompts = self.dataset.select_columns([field]).map(
                lambda sample: {"__prompt": prompt_template.render(**sample)}
            )["__prompt"]
            logger.info(f"Example prompt for tag {tag}: {prompts[0]}")

            responses = run_chat_queries_with_openai(
                queries=prompts,
                api_key=self.openai_api_key,
                logit_bias=logit_bias,
                max_new_tokens=1,
                max_requests_per_minute=self.max_requests_per_minute,
                max_tokens_per_minute=self.max_tokens_per_minute,
            )

            # map back to labels
            first_tokens = [
                tokenizer.encode(response) for response in responses
            ]
            labels = [token2cls[t[0]] for t in first_tokens]
            self.dataset = self.dataset.add_column(
                name=f"__ai_tag__{tag.replace(' ', '_')}", column=labels
            )
            logger.info(
                f"Tagged with tag {tag}. Pausing to cool down before issuing more requests..."
            )
            time.sleep(30)

    return self
