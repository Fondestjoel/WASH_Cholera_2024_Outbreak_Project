# src/clean_news.py
"""
clean_news.py
Cleans and preprocesses the raw news data collected in Step 1.
Saves output: data/clean_news.csv
"""

import os
import pandas as pd
import re

BASE_DIR = r"C:\WASH_Cholera_2024_Outbreak_Project"
RAW_PATH = os.path.join(BASE_DIR, "data", "news_articles.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "clean_news.csv")

def clean_text(text):
    """Lowercase, remove URLs, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^a-z0-9\s.,!?]", "", text)  # keep alphanumerics + basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    if not os.path.exists(RAW_PATH):
        print(f"❌ Raw file not found: {RAW_PATH}")
        return

    df = pd.read_csv(RAW_PATH)

    print(f"✅ Loaded {len(df)} rows from {RAW_PATH}")

    # Keep only relevant columns
    keep_cols = ["title", "summary", "source", "published", "query", "link"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Drop rows missing title and summary
    df = df.dropna(subset=["title", "summary"], how="all")

    # Clean text
    df["title_clean"] = df["title"].apply(clean_text)
    df["summary_clean"] = df["summary"].apply(clean_text)

    # Merge for analysis
    df["text"] = df["title_clean"] + ". " + df["summary_clean"]

    # Drop duplicates based on cleaned text
    df = df.drop_duplicates(subset=["text"])

    # Remove empty text
    df = df[df["text"].str.strip() != ""]

    print(f"🧹 Cleaned dataset contains {len(df)} unique usable articles")

    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"💾 Saved cleaned file to: {CLEAN_PATH}")

if __name__ == "__main__":
    main()
