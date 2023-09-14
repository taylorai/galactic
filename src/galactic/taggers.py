import re
from typing import Sequence, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO)

def tag_string(self, fields: Sequence[str], values: Sequence[str], tag: str, inplace: bool = True):
    # make sure the tag hasn't already been used
    if f"__tag__{tag}" in self.dataset.column_names:
        logging.warning(
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
    if inplace:
        self.dataset = self.dataset.map(tag_)
        logging.info(
            f"Tagged dataset in-place with exact string matching on fields: {fields}"
        )
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.map(tag_)
        logging.info(f"Tagged dataset with exact string matching on fields: {fields}")
        return type(self)(new_dataset)


def tag_regex(self, fields: Sequence[str], regex: str, tag: str, inplace: bool = True):
    # make sure the tag hasn't already been used
    if f"__tag__{tag}" in self.dataset.column_names:
        logging.warning(
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

    if inplace:
        self.dataset = self.dataset.map(tag_)
        logging.info(
            f"Tagged dataset in-place with regex matching on fields: {fields}"
        )
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.map(tag_)
        logging.info(f"Tagged dataset with regex matching on fields: {fields}")
        return type(self)(new_dataset)
