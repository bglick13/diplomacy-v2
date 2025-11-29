from pathlib import Path

import modal

# ==============================================================================
# 1. IMAGE DEFINITIONS
# ==============================================================================

start_cmd = "pip install --upgrade pip"
requirements = [
    "diplomacy",
    "pydantic",
    "numpy",
    "tqdm",
]

# MODERN PATTERN: Use .add_local_python_source("src")
# This makes 'import src.engine.wrapper' work inside the container automatically.

cpu_image = (
    modal.Image.debian_slim()
    .run_commands(start_cmd)
    .pip_install(*requirements)
    .add_local_python_source("src")
)

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

gpu_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .run_commands(start_cmd)
    .uv_pip_install(
        *requirements,
        "vllm",
        "torch",
        "transformers",
        "peft",
        "trl",
        "wandb",
        "accelerate",
    )
    .add_local_python_source("src")
)

# ==============================================================================
# 2. STORAGE & APP SETUP
# ==============================================================================

app = modal.App("diplomacy-grpo")
volume = modal.Volume.from_name("diplomacy-data", create_if_missing=True)

VOLUME_PATH = Path("/data")
MODELS_PATH = VOLUME_PATH / "models"

# ==============================================================================
# 3. INFERENCE ENGINE (THE BRAIN)
# ==============================================================================


# TODO: Add hf cache volume
@app.cls(
    image=gpu_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    container_idle_timeout=300,
    concurrency_limit=5,
    allow_concurrent_inputs=100,
)
class InferenceEngine:
    @modal.enter()
    def setup(self):
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        print("🥶 Initializing vLLM Engine...")

        model_id = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # TODO: Look into fastboot args
        engine_args = AsyncEngineArgs(
            model=model_id,
            enable_lora=True,
            max_loras=4,
            gpu_memory_utilization=0.9,
            disable_log_stats=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Import Processor (Works because of add_local_python_source)
        from src.inference.logits import DiplomacyLogitsProcessor

        self.processor_cls = DiplomacyLogitsProcessor
        print("✅ Engine Ready.")

    @modal.method()
    async def generate(
        self, prompts: list[str], valid_moves: list[dict], lora_path: str = None
    ):
        import uuid

        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import SamplingParams

        lora_req = None
        if lora_path:
            adapter_id = str(hash(lora_path))
            lora_req = LoRARequest(adapter_id, 1, lora_path)

        request_outputs = []

        for i, prompt in enumerate(prompts):
            request_id = str(uuid.uuid4())

            # Instantiate Logic Processor per request
            processor = self.processor_cls(
                tokenizer=self.tokenizer, valid_moves_dict=valid_moves[i]
            )

            sampling_params = SamplingParams(
                temperature=0.7, max_tokens=200, logits_processors=[processor]
            )

            request_outputs.append(
                self.engine.add_request(
                    request_id, prompt, sampling_params, lora_request=lora_req
                )
            )

        final_texts = []
        for generator in request_outputs:
            final_output = None
            async for output in generator:
                final_output = output
            final_texts.append(final_output.outputs[0].text)

        return final_texts


# ==============================================================================
# 4. ROLLOUT WORKER (THE SIMULATION)
# ==============================================================================


@app.function(image=cpu_image, cpu=1.0, memory=1024, timeout=600)
def run_rollout(config_dict: dict, lora_path: str = None):
    # Correct Import Path
    from src.engine.wrapper import DiplomacyWrapper
    from src.utils.config import ExperimentConfig
    from src.utils.parsing import extract_orders

    cfg = ExperimentConfig(**config_dict)

    # Initialize with Configured Horizon
    game = DiplomacyWrapper(horizon=cfg.rollout_horizon_years)

    # Store trajectory if needed
    history = []

    while not game.is_done():
        # 1. Get Inputs (Batch of 7)
        inputs = game.get_all_inputs()

        # 2. Inference (Remote GPU Call)
        raw_responses = InferenceEngine.generate.remote(
            prompts=inputs["prompts"],
            valid_moves=inputs["valid_moves"],
            lora_path=lora_path,
        )

        # 3. Parse & Execute
        all_orders = []
        for i, response_text in enumerate(raw_responses):
            power_orders = extract_orders(response_text)
            all_orders.extend(power_orders)

        game.step(all_orders)

    return game.get_state_json()


# ==============================================================================
# 5. TRAINER (THE LEARNER)
# ==============================================================================


@app.function(image=gpu_image, gpu="H100", volumes={VOLUME_PATH: volume}, timeout=86400)
def train_grpo():
    # ... (Trainer Logic Placeholder) ...
    pass
