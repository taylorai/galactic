# a transform can either modify a column in-place, or add a new column. if the new column is just true/false, it should probably be a "tag".
import os
import unicodedata
from typing import Sequence
import logging
import jinja2
from typing import Optional
from .async_openai import run_chat_queries_with_openai

logger = logging.getLogger("galactic")


def trim_whitespace(self, fields: Sequence[str], inplace: bool = True):
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

    if inplace:
        self.dataset = self.dataset.map(trim_)
        logger.info(f"Trimmed whitespace for fields: {fields}")
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.map(trim_)
        logger.info(f"Trimmed whitespace for fields: {fields}")
        return type(self)(
            new_dataset,
            model=self.model,
            emb_matrix=self.emb_matrix,
            cluster_ids=self.cluster_ids,
            cluster_centers=self.cluster_centers,
            openai_api_key=self.openai_api_key,
        )


def unicode_normalize(
    self,
    fields: Sequence[str],
    form: str = "NFC",
    inplace: bool = True,
):
    """
    Apply Unicode normalization to specified fields. This is useful as a pre-processing step for ML training.
    Note that NFKC normalization mangles certain mathematical expressions like exponents, so you might prefer NFC for that.
    See Appendix A.4 of the Gopher paper: https://arxiv.org/pdf/2112.11446.pdf
    """
    # make sure fields exist and are text fields
    for field in fields:
        if field not in self.dataset.features:
            raise ValueError(f"Field {field} not found in dataset.")
        elif self.dataset.features[field].dtype != "string":
            raise ValueError(
                f"Field {field} is not a string field, and so can't be normalized."
            )
    if inplace:
        self.dataset = self.dataset.map(
            lambda sample: {
                field: unicodedata.normalize(form, sample[field])
                for field in fields
            }
        )
        logger.info(f"Normalized fields {fields} using form {form}")
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.map(
            lambda sample: {
                field: unicodedata.normalize(form, sample[field])
                for field in fields
            }
        )
        logger.info(f"Normalized fields {fields} using form {form}")
        return type(self)(
            new_dataset,
            model=self.model,
            emb_matrix=self.emb_matrix,
            cluster_ids=self.cluster_ids,
            cluster_centers=self.cluster_centers,
            openai_api_key=self.openai_api_key,
        )


def ai_column(
    self,
    new_column: str,
    prompt: str,
    depends_on=list[str],
    normalize: list[str] = ["strip", "lower"],  # must be methods of str class
    system_prompt: Optional[str] = None,
):
    """
    Use OpenAI's API to generate a new column based on an existing column.
    Args:
        name (str): Name of the new column.
        prompt (str): Prompt template (Jinja2) to use for the API request.
        system_prompt (str): System instructions (optional, same for every request).
        depends_on (List[str]): List of fields needed to fill in the template.
    """
    for field in depends_on:
        if field not in self.dataset.column_names:
            raise ValueError(f"Field {field} not found in dataset.")
    if new_column in self.dataset.column_names:
        raise ValueError(
            f"Column {new_column} already exists in dataset. Please choose a different name, or drop the column."
        )
    if self.openai_api_key is None:
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        if self.openai_api_key is None:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY environment variable, or set the openai_api_key attribute of the GalacticDataset."
            )
    template = jinja2.Template(prompt)
    prompts = self.dataset.select_columns(depends_on).map(
        lambda sample: {"__prompt": template.render(**sample)}
    )["__prompt"]
    logger.info(f"Example prompt: {prompts[0]}")
    responses = run_chat_queries_with_openai(
        queries=prompts,
        api_key=self.openai_api_key,
        system_prompt=system_prompt,
        max_tokens_per_minute=self.max_tokens_per_minute,
        max_requests_per_minute=self.max_requests_per_minute,
    )
    # apply normalizations
    for fn in normalize:
        try:
            responses = [getattr(str, fn)(response) for response in responses]
        except AttributeError:
            raise ValueError(f"Unknown normalization function: {fn}")

    # inplace only because this isn't a destructive operation
    self.dataset = self.dataset.add_column(name=new_column, column=responses)
    logger.info(f"Added new column {new_column} using prompt {prompt}")
    # return self for chaining
    return self
