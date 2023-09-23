import re
import os
import random
import tempfile
import fasttext
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import logging

logger = logging.getLogger("galactic")


# Tendency is that bigger target sizes take longer to optimize, so if you only have 5 min use 2M or 1M or 500K
def distill_fasttext(
    self,
    model_name: str,
    save_dir: str,
    input_field: str,
    label_field: str,
    validation_split: float = 0.1,
    normalize: list[str] = ["lower", "strip"],
    split_punctuation: bool = True,
    replace_newlines_with: str = " __newline__ ",
    target_model_size: str = "2M",
    training_duration: int = 300,
    random_seed: int = 42,
):
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

    def _get_label(sample):
        return "__label__" + sample[label_field].lower().replace(" ", "_")

    def _get_line(sample):
        return _get_label(sample) + " " + _preprocess(sample[input_field])

    def _create_train_files(dataset):
        random.seed(random_seed)
        train_file = tempfile.NamedTemporaryFile(mode="a", delete=False)
        val_file = tempfile.NamedTemporaryFile(mode="a", delete=False)
        for sample in dataset:
            if random.random() < validation_split:
                val_file.write(_get_line(sample) + "\n")
            else:
                train_file.write(_get_line(sample) + "\n")
        # return names of files used for training
        return train_file.name, val_file.name

    # create train and validation files
    train_file, val_file = _create_train_files(self.dataset)

    # train fasttext model
    model = fasttext.train_supervised(
        input=train_file,
        autotuneValidationFile=val_file,
        autotuneDuration=training_duration,
        autotuneModelSize=target_model_size,
    )

    # save model
    model.save_model(os.path.join(save_dir, model_name + ".ftz"))


def distill_embedding(
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
