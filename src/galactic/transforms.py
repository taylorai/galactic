# a transform can either modify a column in-place, or add a new column. if the new column is just true/false, it should probably be a "tag".
from typing import Sequence
import logging

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
