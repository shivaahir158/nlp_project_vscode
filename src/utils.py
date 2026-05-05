import time
import logging
import requests
from requests.exceptions import HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("vhp")


def safe_get(url, **kwargs):
    headers = {
        "User-Agent": "VHP-Transcript-Scraper/1.0 academic research"
    }

    for attempt in range(6):
        try:
            resp = requests.get(url, headers=headers, timeout=30, **kwargs)

            if resp.status_code == 429:
                wait = (2 ** attempt) * 10
                logger.warning(f"429 rate limit. Waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            time.sleep(3)
            return resp

        except HTTPError as e:
            if getattr(e.response, "status_code", 0) == 429:
                wait = (2 ** attempt) * 10
                logger.warning(f"429 rate limit. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise

        except Exception as e:
            logger.warning(f"Network error: {e}. Retrying...")
            time.sleep(2 ** attempt)

    raise Exception(f"Request failed after retries: {url}")