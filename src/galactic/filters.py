# there are 2 types of filter: stateless filter that can operate on only a single example,
# or a stateful filter (e.g. Bloom filter) where examples depend on each other.
import re
import json
import pybloom_live
from typing import Sequence

import logging

logger = logging.getLogger("galactic")


def apply_bloom_filter(self, fields: Sequence[str], inplace: bool = True):
    """
    Applies a Bloom filter to the dataset to filter out duplicate examples based on the specified fields.

    .. code-block:: python

        # Example usage:
        ds.apply_bloom_filter(fields=['field1', 'field2'], inplace=False)

    :param fields: Sequence of field names used to apply the Bloom filter.
    :param inplace: Whether to apply the filter in-place or return a new object. Defaults to True.
    :return: If inplace is False, returns a new object of the same type with the filtered dataset.
    :rtype: type(self) if not inplace else None
    """

    bloom = pybloom_live.BloomFilter(
        capacity=len(self.dataset)
        if len(self.dataset) < int(1.0e9)
        else int(1.0e9),
        error_rate=0.001,
    )

    def bloom_filter(sample):
        key = json.dumps({field: sample[field] for field in fields})
        if key in bloom:
            return False
        else:
            bloom.add(key)
            return True

    if inplace:
        self.dataset = self.dataset.filter(bloom_filter)
        logger.info(
            f"Filtered dataset in-place with Bloom filter on fields: {fields}"
        )
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.filter(bloom_filter)
        logger.info(f"Filtered dataset with Bloom filter on fields: {fields}")
        return type(self)(
            new_dataset,
            model=self.model,
            emb_matrix=self.emb_matrix,
            cluster_ids=self.cluster_ids,
            cluster_centers=self.cluster_centers,
            openai_api_key=self.openai_api_key,
        )


# these filters are methods to attach to the GalacticDataset class.
# each filter takes a list of fields which the filter will be (separately) applied to.
def filter_string(
    self, fields: Sequence[str], values: Sequence[str], inplace: bool = True
):
    """
    Filters the dataset by removing examples that contain any of the specified strings in the specified fields.

    .. code-block:: python

        # Example usage:
        ds.filter_string(fields=['field1', 'field2'], values=['value1', 'value2'], inplace=False)

    :param fields: Sequence of field names to apply the filter on.
    :param values: Sequence of string values to filter out from the specified fields.
    :param inplace: Whether to apply the filter in-place or return a new object. Defaults to True.
    :return: If inplace is False, returns a new object of the same type with the filtered dataset.
    :rtype: type(self) if not inplace else None
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
        logger.info(
            f"Filtered dataset in-place with exact string matching on fields: {fields}"
        )
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.filter(filter_)
        logger.info(
            f"Filtered dataset with exact string matching on fields: {fields}"
        )
        return type(self)(
            new_dataset,
            model=self.model,
            emb_matrix=self.emb_matrix,
            cluster_ids=self.cluster_ids,
            cluster_centers=self.cluster_centers,
            openai_api_key=self.openai_api_key,
        )


def filter_regex(
    self, fields: Sequence[str], regex: str, inplace: bool = True
):
    """
    Filters the dataset by removing examples that match the specified regex pattern in the specified fields.

    .. code-block:: python

        # Example usage:
        ds.filter_regex(fields=['field1', 'field2'], regex='[0-9]+', inplace=False)

    :param fields: Sequence of field names to apply the regex filter on.
    :param regex: The regex pattern used to filter out matching examples from the specified fields.
    :param inplace: Whether to apply the filter in-place or return a new object. Defaults to True.
    :return: If inplace is False, returns a new object of the same type with the filtered dataset.
    :rtype: type(self) if not inplace else None
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
        logger.info(
            f"Filtered dataset in-place with regex matching on fields: {fields}"
        )
        # return self for chaining
        return self
    else:
        new_dataset = self.dataset.filter(filter_)
        logger.info(
            f"Filtered dataset with regex matching on fields: {fields}"
        )
        return type(self)(
            new_dataset,
            model=self.model,
            emb_matrix=self.emb_matrix,
            cluster_ids=self.cluster_ids,
            cluster_centers=self.cluster_centers,
            openai_api_key=self.openai_api_key,
        )
