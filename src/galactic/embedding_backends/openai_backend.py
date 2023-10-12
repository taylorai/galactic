from .base import EmbeddingModelBase
from ..async_openai import embed_texts_with_openai
import tiktoken
import openai
import numpy as np


class OpenAIEmbeddingModel(EmbeddingModelBase):
    def __init__(
        self,
        openai_api_key: str,
        max_requests_per_minute: int,
        max_tokens_per_minute: int,
    ):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.openai_api_key = openai_api_key
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

    def embed(self, text: str, normalize: bool = False):
        """
        Embeds a single text with OpenAI API.

        :param text: The input text to be embedded.
        :type text: str
        :return: The normalized averaged embeddings.
        :rtype: numpy.ndarray
        """
        tokens = self.tokenizer.encode(text)
        # if longer than 8191, split into chunks
        if len(tokens) > 8191:
            num_chunks = int(np.ceil(len(tokens) / 8191))
            chunks = np.array_split(tokens, num_chunks).tolist()
        else:
            chunks = [tokens]
        res = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunks,
        )
        embs = np.array([res.data[i].embedding for i in range(len(res.data))])
        avg = np.mean(embs, axis=0)
        return avg / np.linalg.norm(avg)

    def embed_batch(self, texts: list[str], normalize: bool = False):
        """
        Embeds a batch of texts with OpenAI API.
        :param texts: The input texts to be embedded.
        :type texts: list[str]
        :return: The normalized averaged embeddings.

        """
        embs = embed_texts_with_openai(
            texts,
            self.openai_api_key,
            max_tokens_per_minute=self.max_tokens_per_minute,
            max_requests_per_minute=self.max_requests_per_minute,
            show_progress=True,
        )
        return embs
