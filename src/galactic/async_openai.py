### Code here adapted from openai cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
import aiohttp
import json
import numpy as np
import asyncio
import logging
import time
import tiktoken
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("galactic")

EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
CHAT_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: int = 0
    total_requests = 0


@dataclass
class APIRequest:
    task_id: int
    type: str  # either embedding or chat
    text: str
    attempts_left: int
    system_prompt: Optional[str] = None
    logit_bias: Optional[dict[str, float]] = None
    max_new_tokens: Optional[int] = None
    result: list = field(default_factory=list)

    def __post_init__(self):
        # this is the same either way
        tokens = tiktoken.get_encoding("cl100k_base").encode(self.text)
        self.num_tokens = len(tokens)

        # get the URL and request JSON
        self.url = EMBEDDING_URL if self.type == "embedding" else CHAT_URL
        self.request_json = {}

        if self.type == "embedding":
            if self.logit_bias is not None or self.max_new_tokens is not None:
                raise NotImplementedError(
                    "Logit bias / max tokens doesn't apply to embeddings."
                )
            self.request_json["model"] = "text-embedding-ada-002"
            if len(tokens) > 8191:
                num_chunks = int(np.ceil(len(tokens) / 8191))
                self.request_json["input"] = np.array_split(
                    tokens, num_chunks
                ).tolist()
            else:
                self.request_json["input"] = [tokens]
        elif self.type == "chat":
            self.request_json["model"] = (
                "gpt-3.5-turbo"
                if self.num_tokens < 4000
                else "gpt-3.5-turbo-16k"
            )
            messages = []
            if self.system_prompt is not None:
                messages.append(
                    {"content": self.system_prompt, "role": "system"}
                )
            messages.append({"content": self.text, "role": "user"})
            self.request_json["messages"] = messages
            if self.logit_bias is not None:
                self.request_json["logit_bias"] = self.logit_bias
            if self.max_new_tokens is not None:
                self.request_json["max_tokens"] = self.max_new_tokens

    async def call_api(
        self,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        pbar: Optional[tqdm] = None,
    ):
        try:
            status_tracker.total_requests += 1
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url=self.url,
                    headers=request_header,
                    json=self.request_json,
                ) as response:
                    response = await response.json()
            if "error" in response:
                logger.error(str(response))
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                if "context length" in response["error"].get("message", ""):
                    logger.error(
                        "context length exceeded, retrying won't help"
                    )
                    self.attempts_left = 0
                self.result.append(response)
                if self.attempts_left:
                    self.attempts_left -= 1
                    retry_queue.put_nowait(self)
                else:
                    logger.error("out of tries")
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1
            else:
                if pbar is not None:
                    pbar.update(1)
                self.result.append(response)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_succeeded += 1
        except Exception as e:
            self.result.append(str(e))
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1


async def process_api_requests_from_list(
    texts: list[str],
    type: str,
    api_key: str,
    max_attempts: int,
    system_prompt: Optional[str] = None,
    logit_bias: Optional[dict[str, float]] = None,
    max_new_tokens: Optional[int] = None,
    max_tokens_per_minute: int = 90_000,
    max_requests_per_minute: int = 2000,
    show_progress: bool = True,
):
    if type not in ["embedding", "chat"]:
        raise ValueError("type must be either 'embedding' or 'chat'")

    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # infer API endpoint and construct request header
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    texts_not_finished = True
    logger.debug(f"Initialization complete.")

    # turn the texts into an iterator
    if show_progress:
        pbar = tqdm(total=len(texts))
    else:
        pbar = None
    texts = iter(enumerate(texts))
    results = []
    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logger.debug(
                    f"Retrying request {next_request.task_id}: {next_request}"
                )
            elif texts_not_finished:
                try:
                    # get new request
                    idx, text = next(texts)
                    next_request = APIRequest(
                        task_id=idx,
                        type=type,
                        text=text,
                        attempts_left=max_attempts,
                        system_prompt=system_prompt,
                        logit_bias=logit_bias,
                        max_new_tokens=max_new_tokens,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    results.append(next_request)
                    logger.debug(
                        f"Reading request {next_request.task_id}: {next_request}"
                    )
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logger.debug("Read file exhausted")
                    texts_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity
            + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.num_tokens
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                        pbar=pbar,
                    )
                )
                logger.debug(
                    f"Called API for request {next_request.task_id}: {next_request}"
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )
        if (
            seconds_since_rate_limit_error
            < seconds_to_pause_after_rate_limit_error
        ):
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error
                - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logger.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )

    # after finishing, log final status
    logger.info(f"""Parallel processing complete.""")
    if status_tracker.num_tasks_failed > 0:
        logger.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logger.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )
    return results


def embed_texts_with_openai(
    texts: list[str],
    api_key: str,
    max_attempts: int = 10,
    max_tokens_per_minute: int = 350_000,
    max_requests_per_minute: int = 3500,
    show_progress: bool = True,
):
    results = asyncio.run(
        process_api_requests_from_list(
            texts=texts,
            type="embedding",
            api_key=api_key,
            max_attempts=max_attempts,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            show_progress=show_progress,
        )
    )
    # extract the embeddings
    embs = []
    for result in sorted(results, key=lambda x: x.task_id):
        if "error" in result.result[-1].keys():
            embs.append(None)
        else:
            data = result.result[-1]["data"]
            arr = np.array([r["embedding"] for r in data])
            avg = np.mean(arr, axis=0)
            embs.append((avg / np.linalg.norm(avg)).tolist())
    if len(embs) != len(texts):
        # write to json for debugging
        raise ValueError("Length of results does not match length of texts.")
    return embs


def run_chat_queries_with_openai(
    queries: list[str],
    api_key: str,
    system_prompt: Optional[str] = None,
    logit_bias: Optional[dict[str, float]] = None,
    max_new_tokens: Optional[int] = None,
    max_attempts: int = 10,
    max_tokens_per_minute: int = 90_000,
    max_requests_per_minute: int = 2000,
    show_progress: bool = True,
):
    results = asyncio.run(
        process_api_requests_from_list(
            texts=queries,
            type="chat",
            api_key=api_key,
            max_attempts=max_attempts,
            system_prompt=system_prompt,
            logit_bias=logit_bias,
            max_new_tokens=max_new_tokens,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            show_progress=show_progress,
        )
    )
    # extract the replies
    replies = [None for _ in range(len(queries))]
    for result in results:
        if "error" in result.result[-1].keys():
            replies[result.task_id] = None
        else:
            replies[result.task_id] = result.result[-1]["choices"][0][
                "message"
            ]["content"]
    return replies
