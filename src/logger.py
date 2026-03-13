import logging
import os
from datetime import datetime
from pathlib import Path

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# Use project root (two levels up from src/logger.py) to avoid CWD-dependent paths
_PROJECT_ROOT = Path(__file__).parent.parent
logs_path = str(_PROJECT_ROOT / "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE_PATH),
    ],
)


def get_logger(name: str):
    return logging.getLogger(name)
