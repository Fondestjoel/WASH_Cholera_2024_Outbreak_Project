# src/collect_news.py
"""
collect_news.py
Robust Google News RSS collector tailored for:
C:\WASH_Cholera_2024_Outbreak_Project

Saves a single CSV: data/news_articles.csv (overwrites if exists).
"""

import os
import time
import urllib.parse
import feedparser
import pandas as pd

# === User-adjustable settings ===
BASE_DIR = r"C:\WASH_Cholera_2024_Outbreak_Project"
DATA_CSV = os.path.join(BASE_DIR, "data", "news_articles.csv")
QUERIES = [
    "cholera outbreak Nigeria 2024",
    "cholera prevention Nigeria 2024",
    "hygiene Nigeria 2024",
    "sanitation Nigeria 2024",
    "water contamination Nigeria 2024",
    "WASH Nigeria 2024",
    "cholera prevention measures Nigeria 2024",
    "open defecation Nigeria 2024",
    "Bayelsa cholera 2024"
]
MAX_PER_QUERY = 200         # max RSS entries to read per query (feedparser returns many)
PAUSE_SECONDS = 1.2         # polite pause between requests

# === Helpers ===
def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def safe_get_source(entry):
    """Return a source title if available (feedparser may supply .source or .source.title)."""
    try:
        src = entry.get("source", "")
        # some feedparser versions return a dict for source
        if isinstance(src, dict):
            return src.get("title", "") or ""
        return src or ""
    except Exception:
        return ""

def fetch_rss_for_query(query, max_items=100):
    """
    Fetch Google News RSS for a given query.
    Returns list of dicts with safe keys.
    """
    results = []
    q = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-NG&gl=NG&ceid=NG:en"
    feed = feedparser.parse(url)

    entries = feed.entries[:max_items] if hasattr(feed, "entries") else []
    for e in entries:
        try:
            title = e.get("title", "") if hasattr(e, "get") else getattr(e, "title", "")
            link = e.get("link", "") if hasattr(e, "get") else getattr(e, "link", "")
            published = e.get("published", "") if hasattr(e, "get") else getattr(e, "published", "")
            summary = e.get("summary", "") if hasattr(e, "get") else getattr(e, "summary", "")
            source = safe_get_source(e)
        except Exception:
            # fallback robust extraction for unexpected feedparser objects
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            published = getattr(e, "published", "") or ""
            summary = getattr(e, "summary", "") or ""
            source = ""
        results.append({
            "title": title,
            "link": link,
            "published": published,
            "summary": summary,
            "source": source,
            "query": query
        })
    return results

# === Main ===
def main():
    ensure_dir_for_file(DATA_CSV)
    all_articles = []

    print("Starting collection...")
    for q in QUERIES:
        try:
            print(f"  -> Query: {q}")
            items = fetch_rss_for_query(q, max_items=MAX_PER_QUERY)
            print(f"     found {len(items)} items")
            all_articles.extend(items)
        except Exception as exc:
            print(f"     Warning: failed to fetch for query '{q}': {exc}")
        time.sleep(PAUSE_SECONDS)

    if not all_articles:
        print("No articles were collected. Please check network or queries.")
        return

    df = pd.DataFrame(all_articles, columns=["title", "link", "published", "summary", "source", "query"])

    # Normalize empty values
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)

    # Deduplicate: prefer link-based dedupe when links exist, otherwise by title
    if df["link"].str.strip().replace("", pd.NA).notna().any():
        df = df.drop_duplicates(subset=["link"])
    else:
        df = df.drop_duplicates(subset=["title"])

    # Save to CSV (overwrite); use utf-8-sig for Excel compatibility
    df.to_csv(DATA_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} unique articles to: {DATA_CSV}")

if __name__ == "__main__":
    main()