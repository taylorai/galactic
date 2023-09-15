### Code here adapted from openai cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
import aiohttp
import numpy as np
import asyncio
import json
import logging
import time
import tiktoken
from tqdm.auto import tqdm
from dataclasses import dataclass, field

logger = logging.getLogger("galactic")

API_URL = "https://api.openai.com/v1/embeddings"


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: int = 0


@dataclass
class APIRequest:
    task_id: int
    text: str
    attempts_left: int
    result: list = field(default_factory=list)

    def __post_init__(self):
        tokens = tiktoken.get_encoding("cl100k_base").encode(self.text)
        self.num_tokens = len(tokens)
        if len(tokens) > 8191:
            num_chunks = int(np.ceil(len(tokens) / 8191))
            self.input = np.array_split(tokens, num_chunks).tolist()
        else:
            self.input = [tokens]

    async def call_api(
        self,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        pbar: tqdm,
    ):
        request_json = {
            "model": "text-embedding-ada-002",
            "input": self.input,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=API_URL, headers=request_header, json=request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                self.result.append(response)
                if self.attempts_left:
                    print("adding to retry queue")
                    self.attempts_left -= 1
                    retry_queue.put_nowait(self)
                else:
                    print("out of tries")
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1
            else:
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
    api_key: str,
    max_attempts: int,
):
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
    max_requests_per_minute = 3500
    max_tokens_per_minute = 350_000
    available_request_capacity = 3500
    available_token_capacity = 350_000
    last_update_time = time.time()

    # initialize flags
    texts_not_finished = True
    logger.debug(f"Initialization complete.")

    # turn the texts into an iterator
    pbar = tqdm(total=len(texts))
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
                        text=text,
                        attempts_left=max_attempts,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
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
                results.append(next_request)
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
    texts: list[str], api_key: str, max_attempts: int = 10
):
    results = asyncio.run(
        process_api_requests_from_list(
            texts=texts,
            api_key=api_key,
            max_attempts=max_attempts,
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
    return embs
