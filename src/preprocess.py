import pandas as pd
import re
from src.config import RAW_DIR, PROCESSED_DIR
import os
from src.preprocess_token_level import build_token_level_dataset

def preprocess_raw():
    input_path = os.path.join(RAW_DIR, "wikiedits_english.csv")
    df = pd.read_csv(input_path)

    print("Building token-level dataset...")
    token_df = build_token_level_dataset(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DIR, "tokens_labeled.csv")
    token_df.to_csv(save_path, index=False)

    print(f"Token-level dataset saved to {save_path}")


def basic_cleaning(df: pd.DataFrame):
    df["original_text"] = df["original_text"].apply(lambda x: re.sub(r"\s+", " ", str(x).strip()))
    df["corrected_text"] = df["corrected_text"].apply(lambda x: re.sub(r"\s+", " ", str(x).strip()))
    return df
