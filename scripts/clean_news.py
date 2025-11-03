"""
clean_news.py
Robust cleaning pipeline for:
C:\\WASH_Cholera_2024_Outbreak_Project\\data\\news_articles.csv

Output: clean_news.csv (replaces existing file)
Removes duplicates, non-relevant text, punctuation, stopwords,
and normalizes casing for consistency.
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK dependencies are available
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# === Paths ===
BASE_DIR = r"C:\WASH_Cholera_2024_Outbreak_Project"
INPUT_CSV = os.path.join(BASE_DIR, "data", "news_articles.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "clean_news.csv")

# === Load raw data ===
print("🔍 Loading data...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
print(f"✅ Loaded {len(df)} rows")

# === Basic structure checks ===
expected_cols = ["title", "link", "published", "summary", "source", "query"]
missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# === Deduplication ===
print("🧹 Removing duplicates...")
df.drop_duplicates(subset=["title", "link"], inplace=True)

# === Combine text fields for cleaning ===
print("🧩 Combining text fields...")
df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.strip()

# === Drop rows with empty text ===
df = df[df["text"].str.len() > 20]  # remove too-short or empty entries

# === Define text cleaning function ===
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()    # normalize spaces
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# === Apply cleaning ===
print("🧽 Cleaning text fields (this may take a while)...")
df["clean_text"] = df["text"].apply(clean_text)

# === Final validation ===
df = df[df["clean_text"].str.len() > 20]  # remove any residual short text
df.reset_index(drop=True, inplace=True)

# === Save cleaned data ===
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Cleaned data saved: {OUTPUT_CSV}")
print(f"📊 Final record count: {len(df)}")
