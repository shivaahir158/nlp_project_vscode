import re
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm.auto import tqdm
from lxml import etree

from src.utils import safe_get, logger


def fetch_vhp_metadata(n_items=300, page_size=25, checkpoint_file=None):
    base = "https://www.loc.gov/collections/veterans-history-project-collection/"

    existing_ids = set()
    existing_rows = []

    if checkpoint_file and Path(checkpoint_file).exists():
        existing_df = pd.read_csv(checkpoint_file)
        existing_ids = set(existing_df["item_id"].tolist())
        existing_rows = existing_df.to_dict("records")
        logger.info(f"Loaded checkpoint with {len(existing_ids)} records")

    new_items = []
    fetched = 0
    target = n_items - len(existing_ids)

    if target <= 0:
        return pd.DataFrame(existing_rows)

    max_pages = n_items // page_size + 40

    for sp in tqdm(range(1, max_pages), desc="Searching LOC pages"):
        if fetched >= target:
            break

        resp = safe_get(
            base,
            params={
                "fo": "json",
                "c": page_size,
                "sp": sp,
                "q": "World War",
            },
        )

        data = resp.json()
        results = data.get("search_results", {}).get("results", []) or data.get("results", [])

        if not results:
            break

        for item in results:
            if fetched >= target:
                break

            transcript_url = None

            for r in item.get("resources", []):
                ft = r.get("fulltext_file", "")
                label = r.get("resource_label", "").lower()

                if ft and ("transcript" in label or ft.endswith(".xml")):
                    transcript_url = ft
                    break

            if not transcript_url:
                continue

            desc_raw = item.get("description", [])
            desc = " ".join(str(x) for x in desc_raw) if isinstance(desc_raw, list) else str(desc_raw)

            if not any(k in desc.lower() for k in ["world war, 1939", "world war ii", "wwii"]):
                continue

            dig = item.get("number_date_created") or item.get("date", "")
            rec_year = None

            if dig:
                m = re.search(r"(\d{4})", str(dig))
                if m:
                    rec_year = int(m.group(1))

            item_id = re.sub(r"[^\w]", "_", item.get("id", "").strip("/").split("/")[-1])

            if item_id in existing_ids:
                continue

            new_items.append(
                {
                    "item_id": item_id,
                    "title": item.get("title", ""),
                    "transcript_url": transcript_url,
                    "recording_year": rec_year,
                    "branch": (item.get("subject_branch") or ["unknown"])[0],
                    "download_status": "pending",
                    "local_path": None,
                }
            )

            existing_ids.add(item_id)
            fetched += 1

    all_rows = existing_rows + new_items
    meta_df = pd.DataFrame(all_rows)

    if checkpoint_file:
        meta_df.to_csv(checkpoint_file, index=False)

    return meta_df


def download_transcript(url, dest):
    dest = Path(dest)

    if dest.exists():
        return "skipped"

    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        resp = safe_get(url, stream=True)

        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)

        tmp.rename(dest)
        return "ok"

    except Exception as e:
        logger.error(f"Download failed: {url}: {e}")

        if tmp.exists():
            tmp.unlink()

        return "failed"


def extract_text_from_xml(filepath):
    raw = Path(filepath).read_bytes()

    for parser in [etree.XMLParser(recover=True), etree.HTMLParser()]:
        try:
            tree = etree.fromstring(raw, parser=parser)

            interviewer_names = set()

            for author_el in tree.xpath("//author"):
                author_text = " ".join(author_el.itertext()).strip()

                if author_text:
                    for name in re.split(r"\s+and\s+", author_text):
                        name = name.strip()
                        if name:
                            interviewer_names.add(name.lower())

            sp_elements = tree.xpath("//sp")

            if sp_elements:
                speaker_texts = {}

                for sp in sp_elements:
                    speaker_els = sp.xpath(".//speaker")
                    speaker = " ".join(speaker_els[0].itertext()).strip() if speaker_els else ""

                    paragraphs = sp.xpath(".//p")
                    text_parts = [" ".join(p.itertext()).strip() for p in paragraphs]
                    text = "\n".join(t for t in text_parts if t)

                    if text:
                        speaker_lower = speaker.lower()
                        speaker_texts.setdefault(speaker_lower, []).append(text)

                veteran_parts = []

                for spk, parts in speaker_texts.items():
                    is_interviewer = any(iv in spk or spk in iv for iv in interviewer_names)

                    if not is_interviewer:
                        veteran_parts.extend(parts)

                if not veteran_parts and len(speaker_texts) > 1:
                    longest_speaker = max(
                        speaker_texts,
                        key=lambda s: sum(len(t) for t in speaker_texts[s]),
                    )
                    veteran_parts = speaker_texts[longest_speaker]

                if not veteran_parts and len(speaker_texts) == 1:
                    veteran_parts = list(speaker_texts.values())[0]

                if veteran_parts:
                    result = "\n".join(veteran_parts)
                    if result.strip():
                        return result.strip()

            paragraphs = tree.xpath("//body//p")

            if paragraphs:
                text = "\n".join(" ".join(p.itertext()).strip() for p in paragraphs)

                if text.strip():
                    return text.strip()

        except Exception:
            continue

    raw_str = raw.decode("utf-8", errors="ignore")
    clean = re.sub(r"<[^>]+>", " ", raw_str)
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean if len(clean.split()) > 50 else ""


def download_transcripts_batch(meta_df, transcript_dir, checkpoint_file=None):
    transcript_dir = Path(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    meta_df = meta_df.copy()
    transcripts = {}

    for i, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Downloading transcripts"):
        item_id = row["item_id"]
        tr_url = row.get("transcript_url")
        status_now = row.get("download_status", "pending")

        if status_now == "ok" and row.get("local_path"):
            text = extract_text_from_xml(row["local_path"])
            if text:
                transcripts[item_id] = text
            continue

        if not tr_url or pd.isna(tr_url):
            meta_df.at[i, "download_status"] = "no_url"
            continue

        dest = transcript_dir / f"{item_id}.xml"
        dl_status = download_transcript(tr_url, dest)

        if dl_status in ("ok", "skipped"):
            text = extract_text_from_xml(dest)

            if text:
                transcripts[item_id] = text
                meta_df.at[i, "download_status"] = "ok"
                meta_df.at[i, "local_path"] = str(dest)
            else:
                meta_df.at[i, "download_status"] = "empty"
        else:
            meta_df.at[i, "download_status"] = "failed"

        if checkpoint_file:
            meta_df.to_csv(checkpoint_file, index=False)

    return meta_df, transcripts