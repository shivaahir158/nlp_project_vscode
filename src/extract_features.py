import re
from collections import Counter

import datefinder
import numpy as np
import pandas as pd
import spacy
import textstat
from tqdm.auto import tqdm

nlp = spacy.load("en_core_web_sm")

WWII_BIRTH_MIN = 1906
WWII_BIRTH_MAX = 1926

WWII_COHORT = {
    "swell", "gee", "whiz", "jeep", "foxhole", "furlough", "chow", "bunk",
    "latrine", "bivouac", "platoon", "rations", "barracks", "convoy",
    "krauts", "outfit", "bombardment", "artillery", "doughboy",
}

FILLERS = {"um", "uh", "er", "ah", "hmm"}


def extract_birth_year(text, first_n_words=1000):
    words = text.split()
    early = " ".join(words[:first_n_words])
    candidates = []

    # Method 1: datefinder
    try:
        for dt in datefinder.find_dates(early, strict=False):
            if WWII_BIRTH_MIN <= dt.year <= WWII_BIRTH_MAX:
                candidates.append(dt.year)
    except Exception:
        pass

    # Method 2: flexible direct year detection
    for m in re.finditer(r"\b(19\d{2})\b", early):
        year = int(m.group())
        if WWII_BIRTH_MIN <= year <= WWII_BIRTH_MAX:
            candidates.append(year)

    # Method 3: infer birth year from phrases like:
    # "I was 18 years old in 1942"
    age_year_patterns = [
        r"\bI was (\d{1,2}) years? old.*?\b(19\d{2})\b",
        r"\bI was (\d{1,2}).*?\b(19\d{2})\b",
        r"\bat age (\d{1,2}).*?\b(19\d{2})\b",
    ]

    for pattern in age_year_patterns:
        match = re.search(pattern, early, flags=re.IGNORECASE)
        if match:
            age = int(match.group(1))
            year = int(match.group(2))
            birth_year = year - age

            if WWII_BIRTH_MIN <= birth_year <= WWII_BIRTH_MAX:
                candidates.append(birth_year)

    return Counter(candidates).most_common(1)[0][0] if candidates else None

def build_age_labels(meta_df, transcripts):
    age_labels = []

    for _, row in meta_df[meta_df["download_status"] == "ok"].iterrows():
        item_id = row["item_id"]

        if item_id not in transcripts or not row.get("recording_year"):
            continue

        birth_year = extract_birth_year(transcripts[item_id])

        if birth_year:
            recording_year = int(row["recording_year"])
            age = recording_year - birth_year

            age_labels.append(
                {
                    "item_id": item_id,
                    "birth_year": birth_year,
                    "recording_year": recording_year,
                    "age": age,
                    "branch": row.get("branch", "unknown"),
                }
            )

    return pd.DataFrame(age_labels)


def extract_features(text):
    if not text or len(text.split()) < 50:
        return None

    doc = nlp(text)

    tokens = [t for t in doc if not t.is_space]
    alpha = [t for t in tokens if t.is_alpha]
    sents = list(doc.sents)

    if not alpha or not sents:
        return None

    wl = [t.lower_ for t in alpha]
    n = len(wl)

    ttr = len(set(wl)) / n

    win = 50
    if n >= win:
        mattr = np.mean(
            [len(set(wl[i:i + win])) / win for i in range(max(1, n - win + 1))]
        )
    else:
        mattr = ttr

    avg_word_len = np.mean([len(t.text) for t in alpha])

    sent_lengths = [len([t for t in s if t.is_alpha]) for s in sents]
    avg_sent_len = np.mean(sent_lengths)
    std_sent_len = np.std(sent_lengths)

    def dep_depth(tok):
        d = 0
        t = tok

        while t.head != t:
            t = t.head
            d += 1

        return d

    avg_dep_depth = np.mean([dep_depth(t) for t in tokens])

    subord_ratio = sum(
        1 for t in tokens if t.dep_ in {"advcl", "relcl", "ccomp", "xcomp", "acl"}
    ) / len(tokens)

    pos = Counter(t.pos_ for t in tokens)

    noun_ratio = pos.get("NOUN", 0) / len(tokens)
    verb_ratio = pos.get("VERB", 0) / len(tokens)
    adj_ratio = pos.get("ADJ", 0) / len(tokens)

    filler_rate = sum(1 for w in wl if w in FILLERS) / n
    repetition_rate = sum(1 for i in range(1, n) if wl[i] == wl[i - 1]) / n

    cohort_rate = sum(1 for w in wl if w in WWII_COHORT) / n

    fp_words = {"i", "me", "my", "myself", "mine", "we", "us", "our"}
    first_person_rate = sum(1 for w in wl if w in fp_words) / n

    pronoun_types = len(set(t.lower_ for t in tokens if t.pos_ == "PRON"))

    flesch = textstat.flesch_reading_ease(text)
    fk_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)

    return {
        "ttr": ttr,
        "mattr": mattr,
        "avg_word_len": avg_word_len,
        "total_words": n,
        "avg_sent_len": avg_sent_len,
        "std_sent_len": std_sent_len,
        "avg_dep_depth": avg_dep_depth,
        "subord_ratio": subord_ratio,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio,
        "filler_rate": filler_rate,
        "repetition_rate": repetition_rate,
        "cohort_rate": cohort_rate,
        "first_person_rate": first_person_rate,
        "pronoun_types": pronoun_types,
        "flesch_reading_ease": flesch,
        "fk_grade": fk_grade,
        "gunning_fog": gunning_fog,
    }


def build_feature_dataframe(transcripts, age_df):
    age_lookup = {r["item_id"]: r for r in age_df.to_dict("records")}
    rows = []

    for item_id, text in tqdm(transcripts.items(), desc="Extracting linguistic features"):
        if item_id not in age_lookup:
            continue

        feats = extract_features(text)

        if feats:
            rows.append({**feats, **age_lookup[item_id]})

    return pd.DataFrame(rows)