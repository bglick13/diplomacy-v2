import logging
import os
import time
from datetime import datetime

import aiohttp

# Configure logging to show up clearly in Modal logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("diplomacy")
console_logger = logging.getLogger("diplomacy")


class AxiomHandler:
    """Lightweight async logger for Axiom."""

    def __init__(self):
        self.token = os.environ.get("AXIOM_TOKEN")
        self.dataset = os.environ.get("AXIOM_DATASET")
        self.batch = []

    def log(self, event: dict):
        if not self.token:
            return

        # Enrich with metadata
        event.update(
            {"_time": datetime.utcnow().isoformat() + "Z", "service": "rollout_worker"}
        )
        self.batch.append(event)

    async def flush(self):
        if not self.token or not self.batch:
            console_logger.warning("Axiom batch is empty, skipping flush")
            return
        console_logger.info(f"Flushing {len(self.batch)} events to Axiom")

        payload = list(self.batch)
        self.batch = []  # Clear immediately

        url = f"https://api.axiom.co/v1/datasets/{self.dataset}/ingest"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        console_logger.error(f"Axiom Flush Failed: {resp.status}")
        except Exception as e:
            console_logger.error(f"Axiom Error: {e}")


# Global singleton
axiom = AxiomHandler()


class stopwatch:
    """Context manager to measure and log execution time of blocks."""

    def __init__(self, name: str, metadata: dict | None = None):
        self.name = name
        self.start_time = 0
        self.metadata = metadata or {}

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"⏳ [START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        status = "error" if exc_type else "success"

        # 1. Console Log
        if exc_type:
            console_logger.error(f"❌ [FAIL] {self.name} ({duration:.2f}s)")
        else:
            console_logger.info(f"✅ [DONE] {self.name} ({duration:.2f}s)")

        # 2. Axiom Log (Buffered)
        axiom.log(
            {
                "event": "span_duration",
                "span_name": self.name,
                "duration_ms": int(duration * 1000),
                "status": status,
                "error": str(exc_val) if exc_val else None,
                **self.metadata,
            }
        )
