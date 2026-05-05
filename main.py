from pathlib import Path

from src.scrape_transcripts import (
    fetch_vhp_metadata,
    download_transcripts_batch,
)

from src.extract_features import (
    build_age_labels,
    build_feature_dataframe,
)

from src.train_models import train_and_evaluate
from src.advanced_features import add_advanced_features


N_ITEMS =  5000

DATA_DIR = Path("data/vhp_transcripts")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = DATA_DIR / "metadata.csv"


def main():
    print("STEP 1: Fetch metadata")
    meta_df = fetch_vhp_metadata(
        n_items=N_ITEMS,
        checkpoint_file=str(CHECKPOINT_FILE),
    )

    print("STEP 2: Download transcripts")
    meta_df, transcripts = download_transcripts_batch(
        meta_df,
        DATA_DIR,
        checkpoint_file=str(CHECKPOINT_FILE),
    )

    if len(transcripts) == 0:
        print("No transcripts found")
        return

    print("STEP 3: Age labels")
    age_df = build_age_labels(meta_df, transcripts)

    if len(age_df) == 0:
        print("No age labels found")
        return

    print("STEP 4: Feature extraction")
    df = build_feature_dataframe(transcripts, age_df)

    print("STEP 4B: Advanced BERT + DLATK features")
    df = add_advanced_features(df, transcripts, age_df)

    if len(df) == 0:
        print("No features extracted")
        return

    df.to_csv("data/final_dataset.csv", index=False)

    print("STEP 5: Training")
    results = train_and_evaluate(df)

    print("DONE")
    print(results)


if __name__ == "__main__":
    main()