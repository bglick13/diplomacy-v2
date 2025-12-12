from pathlib import Path

import modal

volume = modal.Volume.from_name("diplomacy-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
trace_volume = modal.Volume.from_name("diplomacy-traces", create_if_missing=True)

VOLUME_PATH = Path("/data")
MODELS_PATH = VOLUME_PATH / "models"
REPLAYS_PATH = VOLUME_PATH / "replays"
BENCHMARKS_PATH = VOLUME_PATH / "benchmarks"
HF_CACHE_PATH = Path("/hf-cache")
TRACE_PATH = Path("/traces")
EVALS_PATH = VOLUME_PATH / "evals"
