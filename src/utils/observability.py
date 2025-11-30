import logging
import time

# Configure logging to show up clearly in Modal logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("diplomacy")


class stopwatch:
    """Context manager to measure and log execution time of blocks."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"⏳ [START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            logger.error(f"❌ [FAIL] {self.name} after {duration:.2f}s: {exc_val}")
        else:
            logger.info(f"✅ [DONE] {self.name} took {duration:.2f}s")
