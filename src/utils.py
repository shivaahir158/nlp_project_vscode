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

    for attempt in range(8):
        try:
            resp = requests.get(url, headers=headers, timeout=60, **kwargs)

            if resp.status_code in [429, 500, 502, 503, 504]:
                wait = min((2 ** attempt) * 10, 120)
                logger.warning(
                    f"Server/rate-limit error {resp.status_code}. Waiting {wait}s..."
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            time.sleep(3)
            return resp

        except HTTPError as e:
            status = getattr(e.response, "status_code", None)

            if status in [429, 500, 502, 503, 504]:
                wait = min((2 ** attempt) * 10, 120)
                logger.warning(f"HTTP error {status}. Waiting {wait}s...")
                time.sleep(wait)
                continue

            raise

        except Exception as e:
            wait = min((2 ** attempt) * 5, 60)
            logger.warning(f"Network error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise Exception(f"Request failed after multiple retries: {url}")