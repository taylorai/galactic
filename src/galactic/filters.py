# there are 2 types of filter: stateless filter that can operate on only a single example,
# or a stateful filter (e.g. Bloom filter) where examples depend on each other.
import re
import logging
import pybloom_live
from typing import Sequence, Callable, Optional

logging.basicConfig(level=logging.INFO)


def apply_bloom_filter():
    pass


# these filters are methods to attach to the GalacticDataset class.
# each filter takes a list of fields which the filter will be (separately) applied to.
def filter_string(
    self, fields: Sequence[str], values: Sequence[str], inplace: bool = True
):
    """
    Filter data containing particular string or list of strings in the specified field.
    Args:
        field (str): The field to check against.
        values (List[str]): List of strings to filter on.

    """
    # make a regexp that matches any of the fields
    regexp = re.compile("|".join([re.escape(val) for val in values]))

    def filter_(sample):
        for field in fields:
            if field in sample:
                if isinstance(sample[field], str):
                    if regexp.search(sample[field]):
                        return False
                else:
                    if regexp.search(str(sample[field])):
                        return False
        return True

    if inplace:
        self.dataset = self.dataset.filter(filter_)
        logging.info(
            f"Filtered dataset in-place with exact string matching on fields: {fields}"
        )
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.filter(filter_)
        logging.info(f"Filtered dataset with exact string matching on fields: {fields}")
        return type(self)(new_dataset)


def filter_regex(self, fields: Sequence[str], regex: str, inplace: bool = True):
    """
    Filter data containing particular regex in the specified field.
    Args:
        fields Sequence[str]: The field to check against.
        regex (str): The regex to filter on.

    """
    # make a regexp that matches any of the fields
    regexp = re.compile(regex)

    def filter_(sample):
        for field in fields:
            if field in sample:
                if isinstance(sample[field], str):
                    if regexp.search(sample[field]):
                        return False
                else:
                    if regexp.search(str(sample[field])):
                        return False
        return True

    if inplace:
        self.dataset = self.dataset.filter(filter_)
        logging.info(
            f"Filtered dataset in-place with regex matching on fields: {fields}"
        )
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.filter(filter_)
        logging.info(f"Filtered dataset with regex matching on fields: {fields}")
        return type(self)(new_dataset)
