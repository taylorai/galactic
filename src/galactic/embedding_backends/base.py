from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Literal
import numpy as np
import logging

logger = logging.getLogger("galactic")


class EmbeddingModelBase(ABC):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.max_length = 512

    def __call__(
        self,
        input: Union[str, list[str]],
    ) -> Union[list[list[float]], list[float], np.ndarray]:
        if isinstance(input, str):
            return self.embed(input)
        elif isinstance(input, list):
            return self.embed_batch(input)
        else:
            raise ValueError(
                f"Input must be str or list[str], not {type(input)}"
            )

    def split_and_tokenize_single(
        self,
        text: str,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ) -> dict[str, list[list[int]]]:
        """
        Split and tokenize a single text to prepare it for the embedding model.
        Padding is only necessary if running more than 1 sequence thru the model at once.
        Splitting happens when the model exceeds the max_length (usually 512).
        You can either truncate the text, or split into chunks. Chunking can be "greedy"
        (as many 512 chunks as possible), or "even" (split into even-ish chunks with np.array_split).
        """

        # first make into tokens
        tokenized = self.tokenizer(text)  # (seq_len, )

        # if don't have to pad and don't have to split into chunks, we're done
        if not pad and len(tokenized["input_ids"]) <= self.max_length:
            return {k: [tokenized[k]] for k in tokenized}

        # handle splitting
        if split_strategy == "truncate":
            for k in tokenized:
                tokenized[k] = [tokenized[k][: self.max_length]]

        elif split_strategy == "greedy":
            for k in tokenized:
                tokenized[k] = [
                    tokenized[k][idx : idx + self.max_length]
                    for idx in range(0, len(tokenized[k]), self.max_length)
                ]

        elif split_strategy == "even":
            for k in tokenized:
                tokenized[k] = [
                    arr.tolist()
                    for arr in np.array_split(
                        tokenized[k],
                        int(np.ceil(len(tokenized[k]) / self.max_length)),
                    )
                ]

        else:
            raise ValueError(
                f"split_strategy must be 'truncate', 'greedy', or 'even', not {split_strategy}"
            )

        # pad if applicable
        if pad:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0

            pad_len = max([len(tokenized[k][0]) for k in tokenized])
            for k in tokenized:
                tokenized[k] = [
                    np.pad(
                        tokenized[k][i],
                        (0, pad_len - len(tokenized[k][i])),
                        constant_values=pad_token_id,
                    ).tolist()
                    for i in range(len(tokenized[k]))
                ]

        return tokenized

    def split_and_tokenize_batch(
        self, texts: list[str]
    ) -> Tuple[dict[str, np.ndarray], list[int]]:
        """
        Tokenize the text and pad it to be a multiple of max_length.

        :param text: The input text to be tokenized.
        :type text: str
        :return: Returns a tuple. dictionary containing tokenized and padded 'input_ids',
        'attention_mask' and 'token_type_ids', along with a list of offsets.
        :rtype: Tuple[Dict[str, numpy.ndarray], list[int]]

        Example:

        .. code-block:: python

            tokenized_text = model.split_and_tokenize('sample text')
        """
        raise NotImplementedError
        result = {}
        offsets = []
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        if self.max_length is None:
            raise ValueError("max_length is not initialized.")

        for text in texts:
            # first make into tokens
            tokenized = self.tokenizer(text, return_tensors="np")

            # pad to be a multiple of max_length
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0

            rounded_len = int(
                np.ceil(len(tokenized["input_ids"][0]) / self.max_length)
                * self.max_length
            )
            for k in tokenized:
                tokenized[k] = np.pad(
                    tokenized[k],
                    ((0, 0), (0, rounded_len - len(tokenized[k][0]))),
                    constant_values=pad_token_id,
                )

            # reshape into batch with max length
            for k in tokenized:
                tokenized[k] = tokenized[k].reshape((-1, self.max_length))

            num_chunks = tokenized["input_ids"].shape[0]
            if len(offsets) == 0:
                offsets.append(num_chunks)
            else:
                offsets.append(num_chunks + offsets[-1])

            for k in tokenized:
                if k not in result:
                    result[k] = tokenized[k]
                else:
                    result[k] = np.concatenate(
                        (result[k], tokenized[k]), axis=0
                    )

        return tokenized

    @abstractmethod
    def embed(
        self, text: str, normalize: bool = False
    ) -> Union[list[float], np.ndarray]:
        pass

    @abstractmethod
    def embed_batch(
        self, texts: list[str], normalize: bool = False
    ) -> Union[list[list[float]], np.ndarray]:
        pass
