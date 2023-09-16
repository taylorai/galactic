# a transform can either modify a column in-place, or add a new column. if the new column is just true/false, it should probably be a "tag".
import os
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


def ai_column(
    self,
    name: str,
    prompt: str,
    depends_on=list[str],
    system_prompt: Optional[str] = None,
    max_tokens_per_minute: int = 90_000,
    max_requests_per_minute: int = 2000,
    inplace: bool = True,
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
        max_tokens_per_minute=max_tokens_per_minute,
        max_requests_per_minute=max_requests_per_minute,
    )
    if inplace:
        self.dataset = self.dataset.add_column(name=name, column=responses)
        logger.info(f"Added new column {name} using prompt {prompt}")
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.add_column(name=name, column=responses)
        logger.info(f"Added new column {name} using prompt {prompt}")
        return type(self)(
            new_dataset,
            model=self.model,
            emb_matrix=self.emb_matrix,
            cluster_ids=self.cluster_ids,
            cluster_centers=self.cluster_centers,
            openai_api_key=self.openai_api_key,
        )
