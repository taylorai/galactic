### Utils for making lots of requests to the Replicate API & throttling to avoid hitting rate limit.
### NOT CURRENTLY USED IN PRODUCTION, here for reference & possible future use for embeddings backend.
### Code here adapted from openai cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
import aiohttp
import numpy as np
import asyncio
import json
import logging
import time
import random
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("galactic")

EMBEDDING_URL = "https://api.replicate.com/v1/predictions"
API_VERSION = (
    "4b0d9b1faa51063ba14a70f5f03766a1e82cc654b28c619782653009dc130c23"
)


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
    result: list = field(default_factory=list)

    def __post_init__(self):
        # get the URL and request JSON
        self.url = EMBEDDING_URL
        self.request_json = {
            "version": API_VERSION,
            "input": {
                "text": self.text,
            },
        }

    async def call_api(
        self,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        pbar: tqdm,
    ):
        try:
            status_tracker.total_requests += 1
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # first create the prediction
                async with session.post(
                    url=self.url,
                    headers=request_header,
                    json=self.request_json,
                ) as response:
                    res_json = await response.json()

                # if status is 201, prediction was successfully created
                if response.status == 201:
                    # poll the prediction until it's ready. exponential backoff up to 1 minute. 5 -> 10 -> 20 -> 40 -> 60
                    # don't wait for more than 5 minutes.
                    prediction_url = res_json["urls"]["get"]
                    # random jitter to avoid thundering herd
                    time_to_wait = 5
                    while time_to_wait < 300:
                        await asyncio.sleep(
                            time_to_wait + random.random() * time_to_wait / 2
                        )
                        async with session.get(
                            url=prediction_url,
                            headers=request_header,
                        ) as response:
                            res_json = await response.json()

                        if res_json["status"] == "succeeded":
                            pbar.update(1)
                            self.result.append(res_json)
                            status_tracker.num_tasks_in_progress -= 1
                            status_tracker.num_tasks_succeeded += 1
                            return
                        elif res_json["status"] in ["failed", "canceled"]:
                            logger.warning("Prediction failed or canceled.")
                            self.result.append(res_json)
                            break
                        elif res_json["status"] in ["starting", "processing"]:
                            # still processing the prediction, poll again later
                            time_to_wait *= 2
                        elif response.status == 429:
                            logger.warning(
                                "Rate limit error while fetching prediction."
                            )
                            time_to_wait *= 2

                elif response.status == 429:
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    logger.warning("Rate limit error.")
                    self.result.append(res_json)
                else:
                    logger.warning("Other error. " + str(res_json))
                    self.result.append(res_json)

                if self.attempts_left:
                    logger.info(f"Will retry request {self.task_id}.")
                    self.attempts_left -= 1
                    retry_queue.put_nowait(self)
                else:
                    print("out of tries")
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1

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
    max_requests_per_minute: int = 3000,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # infer API endpoint and construct request header
    request_header = {"Authorization": f"Token {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
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
                        type=type,
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
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            if available_request_capacity >= 1:
                # update counters
                available_request_capacity -= 1
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


def embed_texts_with_replicate(
    texts: list[str],
    api_key: str,
    max_attempts: int = 10,
    max_requests_per_minute: int = 3000,
):
    results = asyncio.run(
        process_api_requests_from_list(
            texts=texts,
            api_key=api_key,
            max_attempts=max_attempts,
            max_requests_per_minute=max_requests_per_minute,
        )
    )
    # extract the embeddings
    embs = []
    for result in sorted(results, key=lambda x: x.task_id):
        embs.append(result.result[-1])
    return embs
