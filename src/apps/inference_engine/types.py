from typing import TypedDict


class GenerationResponse(TypedDict):
    """Response from a single generation request."""

    text: str
    token_count: int
    token_ids: list[int]
    prompt_token_ids: list[int]
    completion_logprobs: list[float]


class GenerateBatchResponseItem(TypedDict):
    """Single item in the batch response from generate() method."""

    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]
    completion_logprobs: list[float]
