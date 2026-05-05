import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex


def build_bert_features(transcripts, age_df, n_components=10):
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    valid_ids = set(age_df["item_id"].tolist())

    item_ids = []
    embeddings = []

    print("Extracting BERT embeddings...")
    for item_id, text in tqdm(transcripts.items(), desc="BERT features"):
        if item_id not in valid_ids:
            continue

        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        sentences = sentences[:200]

        if len(sentences) == 0:
            continue

        emb = model.encode(sentences, show_progress_bar=False)
        doc_emb = emb.mean(axis=0)

        item_ids.append(item_id)
        embeddings.append(doc_emb)

    if len(embeddings) == 0:
        return pd.DataFrame()

    X = np.array(embeddings)

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    cols = [f"bert_pc{i+1}" for i in range(X_pca.shape[1])]

    bert_df = pd.DataFrame(X_pca, columns=cols)
    bert_df["item_id"] = item_ids

    print(f"BERT features created: {bert_df.shape}")

    return bert_df


def extract_dlatk_features(text):
    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(text)

    try:
        nrc = NRCLex(text)
        emotions = nrc.affect_frequencies
    except Exception:
        emotions = {}

    words = text.lower().split()
    total_words = max(len(words), 1)

    first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
    function_words = {
        "the", "a", "an", "and", "or", "but", "if", "because", "to", "of",
        "in", "on", "for", "with", "at", "by", "from"
    }

    return {
        "vader_compound": scores["compound"],
        "vader_pos": scores["pos"],
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],

        "nrc_anger": emotions.get("anger", 0.0),
        "nrc_fear": emotions.get("fear", 0.0),
        "nrc_joy": emotions.get("joy", 0.0),
        "nrc_sadness": emotions.get("sadness", 0.0),
        "nrc_trust": emotions.get("trust", 0.0),
        "nrc_anticipation": emotions.get("anticipation", 0.0),

        "first_person_word_rate": sum(w in first_person for w in words) / total_words,
        "function_word_rate": sum(w in function_words for w in words) / total_words,
    }


def build_dlatk_features(transcripts, age_df):
    valid_ids = set(age_df["item_id"].tolist())
    rows = []

    print("Extracting DLATK-style features...")
    for item_id, text in tqdm(transcripts.items(), desc="DLATK features"):
        if item_id not in valid_ids:
            continue

        feats = extract_dlatk_features(text)
        rows.append({"item_id": item_id, **feats})

    dlatk_df = pd.DataFrame(rows)

    print(f"DLATK features created: {dlatk_df.shape}")

    return dlatk_df


def add_advanced_features(df, transcripts, age_df):
    bert_df = build_bert_features(transcripts, age_df)
    dlatk_df = build_dlatk_features(transcripts, age_df)

    df_all = df.copy()

    if not bert_df.empty:
        df_all = df_all.merge(bert_df, on="item_id", how="left")

    if not dlatk_df.empty:
        df_all = df_all.merge(dlatk_df, on="item_id", how="left")

    print(f"Final dataset with advanced features: {df_all.shape}")

    return df_all