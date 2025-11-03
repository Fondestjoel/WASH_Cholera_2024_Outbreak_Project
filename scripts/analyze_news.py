# analyze_news.py
"""
Performs thematic and sentiment analysis of cholera-prevention news articles (2024 Nigeria).

Outputs:
- data/theme_distribution.png
- data/sentiment_distribution.png
- data/trend_analysis_2024.png
- data/clustered_articles.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
from datetime import datetime

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "clean_news.csv")
OUTPUT_CLUSTER_FILE = os.path.join(DATA_DIR, "clustered_articles.csv")

# === READ DATA ===
df = pd.read_csv(INPUT_FILE)

# Ensure 'published' is datetime
df["published"] = pd.to_datetime(df["published"], errors="coerce")
df = df.dropna(subset=["published"])

# Keep only 2024 articles
df = df[df["published"].dt.year == 2024]

# Combine title + summary for analysis
df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.strip()

# === TF-IDF & CLUSTERING ===
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df["text"])

# Choose number of clusters — can adjust to 5 for broader themes
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# Label clusters manually based on keyword inspection
# You can later adjust these names if reviewing cluster contents
cluster_labels = {
    0: "Water, Sanitation & Hygiene",
    1: "Government & Policy Response",
    2: "Health Education & Prevention",
    3: "Outbreak Reports & Mortality",
    4: "Community Response & Aid"
}
df["theme"] = df["cluster"].map(cluster_labels)

# === SENTIMENT ANALYSIS ===
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["text"].apply(get_sentiment)

# === SAVE CLUSTERED DATA ===
df.to_csv(OUTPUT_CLUSTER_FILE, index=False, encoding="utf-8-sig")

# === VISUALIZATION OUTPUTS ===

# 1. THEME DISTRIBUTION
plt.figure(figsize=(8, 5))
df["theme"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Theme Distribution of Cholera Prevention Coverage (2024)")
plt.xlabel("Themes")
plt.ylabel("Number of Articles")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "theme_distribution.png"), dpi=300)
plt.close()

# 2. SENTIMENT DISTRIBUTION
plt.figure(figsize=(6, 4))
df["sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=140)
plt.ylabel("")
plt.title("Sentiment Distribution (2024 News Articles)")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "sentiment_distribution.png"), dpi=300)
plt.close()

# 3. TREND ANALYSIS (2024)
df["month"] = df["published"].dt.to_period("M")
trend = df["month"].value_counts().sort_index()

plt.figure(figsize=(9, 5))
trend.plot(kind="line", marker="o", linewidth=2)
plt.title("Monthly Trend of Cholera-Related News (2024)")
plt.xlabel("Month")
plt.ylabel("Number of Articles")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "trend_analysis_2024.png"), dpi=300)
plt.close()

print(f"✅ Analysis complete. {len(df)} articles processed for 2024.")
print(f"Outputs saved in: {DATA_DIR}")
