# a transform can either modify a column in-place, or add a new column. if the new column is just true/false, it should probably be a "tag".
import os
import re
import numpy as np
import unicodedata
from typing import Sequence
import logging
import jinja2
from typing import Optional, Union
import tiktoken
import fasttext
import joblib
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


def normalize(
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
    name: str,
    prompt: str,
    depends_on=list[str],
    normalize: list[str] = ["strip", "lower"],  # must be methods of str class
    system_prompt: Optional[str] = None,
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
    if name in self.dataset.column_names:
        raise ValueError(
            f"Column {name} already exists in dataset. Please choose a different name, or drop the column."
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


def ai_classifier(
    self,
    name: str,
    field: Optional[str],
    classes: Union[list[str], dict[str, str]],
    prompt: Optional[str] = None,
    backend="openai",
):
    """
    Use OpenAI's API or a HF zero-shot classifier to generate a new column based on an existing column.
    Args:
        name (str): Name of the new column.
        field (str): Name of the field to use for classification. Can be None if backend is 'embeddings'.
        classes (List[str] or dict[str, str]): Classes to classify into. Can be just a list of labels, or a dict of label: description.
        If provided, descriptions will be used for a) prompting API model if backend = 'openai', b) embedding if backend = "embeddings",
        c) zero-shot classification if backend = "huggingface".
        prompt (str): Prompt template (Jinja2) to use for the API request, with a template slot for the field to classify.
        Only used if backend = 'openai'. If none provided, a basic default prompt will be used.
        default prompt will be used.
        backend (str): Which backend to use. Currently supports 'embeddings', 'openai' and 'huggingface'.
    """
    if backend == "embeddings":
        # warn if field is specified that embeddings ignores the field
        if field is not None:
            logger.warning(
                f"Field {field} specified, but we're using embeddings to classify, which will use whatever field was embedded with get_embeddings()."
            )
        # make sure that we've already embedded the dataset
        if "__embedding" not in self.dataset.column_names:
            raise RuntimeError(
                "Dataset does not have embeddings. Run get_embeddings() first."
            )
        elif self.model is None:
            raise RuntimeError(
                "Dataset does not have an embedding model. Run get_embeddings() first, or run get_embedding_model() with the backend that matches '__embeddings'."
            )
        # embed the classes
        idx2label = (
            {idx: label for idx, label in enumerate(classes)}
            if isinstance(classes, list)
            else {idx: label for idx, label in enumerate(classes.keys())}
        )
        idx2input = (
            idx2label
            if isinstance(classes, list)
            else {idx: classes[label] for idx, label in idx2label.items()}
        )
        class_embeddings = np.array(
            [self.model(idx2input[i]) for i in range(len(idx2input))]
        )

        # get the embeddings for the dataset
        self.dataset = self.dataset.map(
            lambda sample: {
                name: idx2label[
                    np.dot(class_embeddings, sample["__embedding"]).argmax()
                ]
            }
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

        # if prompt is not provided, make default prompt based on class descriptions
        if prompt is None:
            prompt = "Classify the provided text into one of the following classes:\n\n"
            if isinstance(classes, list):
                for label in classes:
                    prompt += f"- {label}\n"
            else:
                for label, description in classes.items():
                    prompt += f"- {label}: {description}\n"
            prompt += "\n---\n\nText: {{" + field + "}}\n\n---\n\nClass:"

        # create logit bias
        token2cls = {}
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        class_list = (
            list(classes)
            if isinstance(classes, list)
            else list(classes.keys())
        )
        for cls in class_list:
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

        # create queries
        template = jinja2.Template(prompt)
        prompts = self.dataset.select_columns([field]).map(
            lambda sample: {"__prompt": template.render(**sample)}
        )["__prompt"]
        logger.info(f"Example prompt: {prompts[0]}")

        # run queries
        responses = run_chat_queries_with_openai(
            queries=prompts,
            api_key=self.openai_api_key,
            logit_bias=logit_bias,
            max_new_tokens=1,
            max_tokens_per_minute=self.max_tokens_per_minute,
            max_requests_per_minute=self.max_requests_per_minute,
        )

        # map back to labels
        first_tokens = [tokenizer.encode(response) for response in responses]
        labels = [token2cls[t[0]] for t in first_tokens]
        self.dataset = self.dataset.add_column(name=name, column=labels)

    elif backend == "huggingface":
        from transformers import pipeline

        pipe = pipeline(
            "zero-shot-classification",
            model="mrm8488/deberta-v3-small-finetuned-mnli",
        )
        if isinstance(classes, list):
            candidate_answers = classes
            ans2label = {c: c for c in classes}
        else:
            candidate_answers = list(classes.values())
            ans2label = {classes[k]: k for k in classes.keys()}

        self.dataset = self.dataset.map(
            lambda sample: {
                name: ans2label[
                    pipe(sample[field], candidate_answers)["labels"][0]
                ]
            }
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return self


def fasttext_classifier(
    self,
    new_column: str,
    model_path: str,
    field: str,
    # these should match how the model was trained
    normalize: list[str] = ["lower", "strip"],
    split_punctuation: bool = True,
    replace_newlines_with: str = " __newline__ ",
):
    # make sure the input field is a string
    if self.dataset.features[field].dtype != "string":
        raise ValueError(f"Field {field} is not a string field.")

    model = fasttext.load_model(model_path)

    def _preprocess(text):
        for fn in normalize:
            text = getattr(str, fn)(text)
        text = text.replace("\n", replace_newlines_with)
        for n in normalize:
            if n == "lower":
                text = text.lower()
            elif n == "strip":
                text = text.strip()
        if split_punctuation:
            text = re.sub(r"([.!?,\'/()])", r" \1 ", text)
        return text

    def _classify(sample):
        input_text = _preprocess(sample[field])
        output = model.predict(input_text)
        return {new_column: output[0][0].split("__label__")[1]}

    self.dataset = self.dataset.map(_classify)
    logger.info(
        f"Applied classifier to field {field}, added result to {new_column}."
    )
    return self


def embeddings_classifier(
    self,
    new_column: str,
    model_path: str,
    field: str = "__embedding",
):
    model = joblib.load(model_path + ".joblib")
    le = joblib.load(model_path + ".labels.joblib")

    def _classify(sample):
        outputs = le.inverse_transform(model.predict(sample[field]))
        return {new_column: outputs}

    self.dataset = self.dataset.map(_classify, batched=True)
    logger.info(
        f"Applied classifier to field {field}, added result to {new_column}."
    )

    return self
