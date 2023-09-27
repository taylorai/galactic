import re
import os
import random
import tempfile
import fasttext
import numpy as np
import joblib
import tiktoken
import jinja2
from typing import Optional, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from .async_openai import run_chat_queries_with_openai

import logging

logger = logging.getLogger("galactic")


# Tendency is that bigger target sizes take longer to optimize, so if you only have 5 min use 2M or 1M or 500K
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
):
    """
    Trains a fasttext classifier on the dataset provided to the instance.

    Args:
        self: The instance of the class.
        model_name (str): The name of the model to be saved.
        save_dir (str): The directory where the model should be saved.
        input_field (str): The field name from the dataset to be used as input.
        label_field (str): The field name from the dataset to be used as label.
        validation_split (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.1.
        test_split (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.1.
        normalize (list[str], optional): List of normalization to apply to the text. Defaults to ["lower", "strip"].
        split_punctuation (bool, optional): Whether to split the text by punctuation. Defaults to True.
        replace_newlines_with (str, optional): String to replace newlines with in the text. Defaults to " __newline__ ".
        target_model_size (str, optional): The desired size of the fasttext model. Defaults to "2M".
        training_duration (int, optional): The duration in seconds for the fasttext autotune. Defaults to 300.
        random_seed (int, optional): Seed for reproducing results. Defaults to 42.
    """

    def _preprocess(text):
        text = str(text)
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

    def _get_label(sample):
        return "__label__" + str(sample[label_field]).lower().replace(" ", "_")

    def _get_line(sample):
        return _get_label(sample) + " " + _preprocess(sample[input_field])

    def _create_train_files(dataset):
        random.seed(random_seed)
        train_file = tempfile.NamedTemporaryFile(mode="a", delete=False)
        val_file = tempfile.NamedTemporaryFile(mode="a", delete=False)
        test_file = tempfile.NamedTemporaryFile(mode="a", delete=False)
        for sample in dataset:
            if random.random() < validation_split:
                val_file.write(_get_line(sample) + "\n")
            elif random.random() < test_split + validation_split:
                test_file.write(_get_line(sample) + "\n")
            else:
                train_file.write(_get_line(sample) + "\n")
        # return names of files used for training
        return train_file.name, val_file.name, test_file.name

    # create train and validation files
    train_file, val_file, test_file = _create_train_files(self.dataset)

    # train fasttext model
    model = fasttext.train_supervised(
        input=train_file,
        autotuneValidationFile=val_file,
        autotuneDuration=training_duration,
        autotuneModelSize=target_model_size,
    )

    # test model on test set, report individual class scores
    overall = model.test(test_file)
    scores = model.test_label(test_file)
    scores = {k: v["precision"] for k, v in scores.items()}
    logger.info(f"Test set accuracy: {overall[1]}")
    logger.info(f"Test set accuracy per-class: {scores}")

    # save model
    model.save_model(os.path.join(save_dir, model_name + ".ftz"))


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
):
    """
    Trains a classifier on the dataset using embeddings.

    Args:
        self: The instance of the class.
        model_name (str): The name of the model to be saved.
        save_dir (str): The directory where the model should be saved.
        label_field (str): The field name from the dataset to be used as label.
        model_type (str, optional): The type of model to train. Either "svm" or "logistic". Defaults to "svm".
        input_field (str, optional): The field name from the dataset to be used as input. Defaults to "__embedding".
        validation_split (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.1.
        test_split (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.1.
        random_seed (int, optional): Seed for reproducing results. Defaults to 42.
    """
    if "__embedding" not in input_field:
        logger.warning(
            "This is designed to use the embeddings (typically '__embedding') as input. I hope you know what you're doing..."
        )
    elif input_field not in self.dataset.features:
        raise ValueError(f"Input field {input_field} not found in dataset.")
    elif label_field not in self.dataset.features:
        raise ValueError(f"Label field {label_field} not found in dataset.")
    # TODO: make sure input_field is a list of numbers to avoid footgun
    X = np.array(self.dataset[input_field])

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(self.dataset[label_field])

    # split train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed
    )

    # train model
    if model_type not in ["svm", "logistic"]:
        raise ValueError(
            f"Model type {model_type} not supported. Use 'svm' or 'logistic'."
        )
    loss = "hinge" if model_type == "svm" else "log_loss"
    model = SGDClassifier(
        random_state=random_seed,
        loss=loss,
        validation_fraction=validation_split,
        early_stopping=True,
    )
    model.fit(X_train, y_train)

    # display train & test accuracy
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"Training accuracy: {train_acc}")
    logger.info(f"Test accuracy: {test_acc}")
    logger.info(
        f"Validation split was used for early stopping, not for measuring out-of-sample accuracy."
    )

    # save model & label encoder
    joblib.dump(model, os.path.join(save_dir, model_name + ".joblib"))
    joblib.dump(le, os.path.join(save_dir, model_name + ".labels.joblib"))


def ai_classifier(
    self,
    new_column: str,
    field: Optional[str],
    classes: Union[list[str], dict[str, str]],
    prompt: Optional[str] = None,
    backend="openai",
):
    """
    Classify a field using OpenAI's API or a HF zero-shot classifier to generate a new column based on an existing column.

    Args:
        self: The instance of the class.
        new_column (str): Name of the new column to be added to the dataset.
        field (Optional[str]): Name of the field to use for classification. Can be None if backend is 'embeddings'.
        classes (Union[list[str], dict[str, str]]): Classes to classify into. Can be a list of labels or a dict of label: description.
        prompt (Optional[str], optional): Prompt template (Jinja2) to use for the API request, with a template slot for the field to classify. Defaults to None.
        backend (str, optional): Which backend to use. Currently supports 'embeddings', 'openai', and 'huggingface'. Defaults to "openai".
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
                new_column: idx2label[
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
        self.dataset = self.dataset.add_column(name=new_column, column=labels)

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
                new_column: ans2label[
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
    """
    Classify a field using a trained fastText model to generate a new column based on an existing column.

    Args:
        self: The instance of the class.
        fasttext_model (str): The path to the trained fastText model file.
        new_column (str): Name of the new column to be added to the dataset.
        field (Optional[str]): Name of the field to use for classification. Can be None if the fastText model is trained on embeddings.
        k (int, optional): Number of labels to predict. Defaults to 1.
    """
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
    """Use an embeddings model to classify a field."""
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
