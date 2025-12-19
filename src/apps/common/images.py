import modal

start_cmd = "pip install --upgrade pip"
requirements = [
    "diplomacy",
    "pydantic",
    "numpy",
    "tqdm",
    "cloudpickle",
    "wandb",
    "weave",  # For trajectory tracing in rollouts
]

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
        "huggingface_hub",
        "hf_transfer",
        "pynvml",
    )
    .env(
        {
            "VLLM_USE_V1": "1",  # Enable vLLM v1 engine
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",  # Allow dynamic LoRA loading
            "VLLM_PLUGINS": "lora_filesystem_resolver",  # Use built-in resolver
            "VLLM_LORA_RESOLVER_CACHE_DIR": "/data/models",  # Where adapters are stored
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Use faster hf-transfer for downloads (if available)
        }
    )
    .add_local_python_source("src")
)
