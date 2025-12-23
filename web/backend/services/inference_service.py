"""Inference service abstraction for local vLLM, Modal, and mock inference."""

import os
import random
from abc import ABC, abstractmethod
from typing import Any


class InferenceService(ABC):
    """Abstract inference service interface."""

    @abstractmethod
    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
    ) -> list[dict]:
        """Generate responses for the given prompts.

        Args:
            prompts: List of prompt strings.
            valid_moves: List of valid moves dicts for each prompt.
            lora_name: Optional LoRA adapter name.
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            List of response dicts with keys:
                - text: Generated text
                - token_ids: Token IDs of the completion
                - prompt_token_ids: Token IDs of the prompt
                - completion_logprobs: Log probabilities for each token
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available."""
        pass


class MockInferenceService(InferenceService):
    """Mock service for UI development - picks random valid moves."""

    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
    ) -> list[dict]:
        """Generate random valid orders for each prompt."""
        results = []
        for moves_dict in valid_moves:
            orders = []
            for _unit, options in moves_dict.items():
                if options:
                    orders.append(random.choice(options))
            results.append(
                {
                    "text": "\n".join(orders),
                    "token_ids": [],
                    "prompt_token_ids": [],
                    "completion_logprobs": [],
                }
            )
        return results

    def is_available(self) -> bool:
        return True


class ModalInferenceService(InferenceService):
    """Modal-based inference for production.

    Uses WebInferenceEngine which is optimized for web traffic with:
    - GPU memory snapshotting for fast cold starts (~3-5s vs ~30-60s)
    - enforce_eager=True to skip CUDA graph compilation
    - Lower concurrency settings for single-user traffic
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self._engine_cls = None

    def _get_engine_cls(self):
        """Lazy load Modal engine class (WebInferenceEngine for fast cold starts)."""
        if self._engine_cls is None:
            import modal

            # Use WebInferenceEngine optimized for web traffic
            self._engine_cls = modal.Cls.from_name("diplomacy-grpo", "WebInferenceEngine")
        return self._engine_cls

    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
    ) -> list[dict]:
        """Generate using Modal InferenceEngine."""
        engine_cls = self._get_engine_cls()
        engine = engine_cls(model_id=self.model_id)
        return await engine.generate.remote.aio(
            prompts=prompts,
            valid_moves=valid_moves,
            lora_name=lora_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def is_available(self) -> bool:
        try:
            import modal  # noqa: F401

            return True
        except ImportError:
            return False


class LocalVLLMService(InferenceService):
    """Local vLLM inference for debugging with local GPU."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        enable_lora: bool = True,
        max_lora_rank: int = 16,
    ):
        self.model_id = model_id
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self._engine: Any = None
        self._tokenizer: Any = None

    def _initialize_engine(self):
        """Lazy initialize the vLLM engine."""
        if self._engine is not None:
            return

        import os

        os.environ["VLLM_USE_V1"] = "1"

        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        from src.inference.logits import DiplomacyLogitsProcessor

        print(f"Initializing local vLLM engine with model: {self.model_id}")

        engine_args = AsyncEngineArgs(
            model=self.model_id,
            enable_lora=self.enable_lora,
            max_loras=8,
            max_lora_rank=self.max_lora_rank,
            gpu_memory_utilization=0.9,
            max_num_seqs=64,
            enable_prefix_caching=True,
            logits_processors=[DiplomacyLogitsProcessor],
        )

        self._engine = AsyncLLM.from_engine_args(engine_args)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print("Local vLLM engine initialized")

    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
    ) -> list[dict]:
        """Generate using local vLLM engine."""
        import asyncio
        import uuid

        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import SamplingParams

        self._initialize_engine()

        # Build LoRA request if specified
        lora_req = None
        if lora_name:
            lora_path = f"/data/models/{lora_name}"
            if os.path.exists(lora_path):
                lora_int_id = abs(hash(lora_name)) % (2**31)
                lora_req = LoRARequest(lora_name, lora_int_id, lora_path)
            else:
                print(f"Warning: LoRA adapter not found at {lora_path}")

        async def generate_single(prompt: str, moves: dict) -> dict:
            request_id = str(uuid.uuid4())
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_args={"valid_moves_dict": moves, "start_active": True},
                stop=["</orders>", "</Orders>"],
                logprobs=1,
            )

            generator = self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_req,
            )

            final_output = None
            async for output in generator:
                final_output = output

            text = ""
            token_ids: list[int] = []
            prompt_token_ids: list[int] = []
            completion_logprobs: list[float] = []

            if final_output:
                prompt_token_ids = final_output.prompt_token_ids or []
                if final_output.outputs:
                    output = final_output.outputs[0]
                    text = output.text
                    token_ids = list(output.token_ids)

                    # Extract logprobs
                    if output.logprobs:
                        for i, pos_logprobs in enumerate(output.logprobs):
                            if pos_logprobs and i < len(token_ids):
                                tid = token_ids[i]
                                if tid in pos_logprobs:
                                    completion_logprobs.append(pos_logprobs[tid].logprob)
                                else:
                                    completion_logprobs.append(0.0)
                            else:
                                completion_logprobs.append(0.0)

            return {
                "text": text,
                "token_ids": token_ids,
                "prompt_token_ids": prompt_token_ids,
                "completion_logprobs": completion_logprobs,
            }

        # Generate all responses concurrently
        results = await asyncio.gather(
            *[generate_single(p, m) for p, m in zip(prompts, valid_moves, strict=True)]
        )
        return list(results)

    def is_available(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False


# Singleton instances
_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Factory to get appropriate inference service based on environment.

    Set INFERENCE_MODE environment variable:
        - "mock": Use mock service (random valid moves)
        - "modal": Use Modal InferenceEngine
        - "local": Use local vLLM engine

    Defaults to "mock" for development.
    """
    global _inference_service

    if _inference_service is not None:
        return _inference_service

    mode = os.environ.get("INFERENCE_MODE", "mock").lower()
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

    if mode == "local":
        service = LocalVLLMService(model_id=model_id)
        if service.is_available():
            _inference_service = service
            print(f"Using local vLLM inference with model: {model_id}")
        else:
            print("Local vLLM not available (no CUDA), falling back to mock")
            _inference_service = MockInferenceService()

    elif mode == "modal":
        service = ModalInferenceService(model_id=model_id)
        if service.is_available():
            _inference_service = service
            print(f"Using Modal inference with model: {model_id}")
        else:
            print("Modal not available, falling back to mock")
            _inference_service = MockInferenceService()

    else:
        _inference_service = MockInferenceService()
        print("Using mock inference (random valid moves)")

    return _inference_service


def reset_inference_service():
    """Reset the inference service singleton (for testing)."""
    global _inference_service
    _inference_service = None
