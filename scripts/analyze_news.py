# src/analyze_news.py
"""
Deep analysis script for C:\WASH_Cholera_2024_Outbreak_Project

Performs:
- TF-IDF + SVD
- Automatic k selection (silhouette on sample)
- KMeans clustering (final: MiniBatchKMeans)
- LDA topic modelling (sklearn)
- Top terms per cluster & per LDA topic
- Robust sentiment (TextBlob if available, lexicon fallback)
- Temporal theme trends (if 'published' exists)
- Source x theme contribution
- Saves/overwrites fixed filenames under data/
"""

import os, sys, time, json, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings("ignore")

# ----- CONFIG -----
BASE_DIR = r"C:\WASH_Cholera_2024_Outbreak_Project"
DATA_DIR = Path(BASE_DIR) / "data"
CLEAN_PATH = DATA_DIR / "clean_news.csv"

OUT_ANALYZED = DATA_DIR / "analyzed_news.csv"
OUT_CLUSTER_TERMS = DATA_DIR / "cluster_top_terms.csv"
OUT_LDA_TOPICS = DATA_DIR / "lda_topics.csv"
OUT_THEME_SUM = DATA_DIR / "theme_summary.csv"
OUT_SENT_SUM = DATA_DIR / "sentiment_summary.csv"
OUT_THEME_PLOT = DATA_DIR / "theme_distribution.png"
OUT_SENT_PLOT = DATA_DIR / "sentiment_distribution.png"
OUT_PROJ = DATA_DIR / "clusters_projection_2d.png"
OUT_TREND = DATA_DIR / "theme_trend.png"
OUT_SOURCE_MATRIX = DATA_DIR / "source_theme_matrix.csv"
OUT_RUN_INFO = DATA_DIR / "run_info.json"

# Modelling params
MAX_FEATURES = 5000
SVD_COMPONENTS = 50
MIN_K = 3
MAX_K = 10
RANDOM_STATE = 42
LDA_TOPICS = 6
TOP_TERMS_N = 12

# Label rules (editable)
LABEL_RULES = [
    ("Cholera Outbreak & Case Reporting", ["cholera","case","cases","death","deaths","reported","fatal","died"]),
    ("Hygiene & Preventive Behaviour", ["hygiene","handwash","handwashing","soap","wash","sanitize","sanitiser","sanitization","sanitise"]),
    ("Water Safety & Contamination", ["water","contaminat","drinking","borehole","tap","supply","contamination","river"]),
    ("Sanitation & Waste Management", ["sanitation","latrine","toilet","waste","drainage","sewer","defecation","open defecation","faecal"]),
    ("Government Response & Policy", ["government","minister","ncdc","policy","response","authority","state","federal"]),
    ("Aid, NGOs & Community Support", ["aid","ngo","support","donation","organisation","relief","community"]),
    ("Environmental Drivers & Flooding", ["flood","flooding","rain","drainage","environment","river","overflow"])
]

# ----- HELPERS -----
def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def exit_with(msg):
    print(msg); sys.exit(1)

def load_clean():
    if not CLEAN_PATH.exists():
        exit_with(f"ERROR: clean file missing at {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH, dtype=str)
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str)
    # Build or verify text column
    if "text" not in df.columns or df["text"].str.strip().eq("").all():
        df["title"] = df.get("title","").fillna("")
        df["summary"] = df.get("summary","").fillna("")
        df["text"] = (df["title"].astype(str) + ". " + df["summary"].astype(str)).str.strip()
        print("Info: built 'text' from title + summary.")
    df = df[df["text"].str.strip() != ""].copy()
    df = df.drop_duplicates(subset=["text"])
    return df

def compute_tfidf(texts, max_features=MAX_FEATURES):
    vec = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X

def reduce_svd(X, n_components=SVD_COMPONENTS):
    n_components = min(n_components, X.shape[1]-1, X.shape[0]-1)
    n_components = max(2, n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    Xr = svd.fit_transform(X)
    return svd, Xr

def choose_k_by_silhouette(X_scaled, min_k=MIN_K, max_k=MAX_K):
    n_docs = X_scaled.shape[0]
    max_k = min(max_k, max(2, n_docs-1))
    best_k = None; best_score = -1; scores = {}
    sample_n = min(n_docs, 400)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(n_docs, sample_n, replace=False)
    for k in range(min_k, max_k+1):
        try:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, max_iter=300)
            labs = km.fit_predict(X_scaled[sample_idx])
            if len(set(labs))>1:
                sc = silhouette_score(X_scaled[sample_idx], labs)
                scores[k]=float(sc)
                if sc > best_score:
                    best_score = sc; best_k = k
            else:
                scores[k]=None
        except Exception:
            scores[k]=None
    if best_k is None:
        best_k = min(6, max(2, int(round(math.sqrt(n_docs)))))
    return best_k, scores

def top_terms_for_clusters(X_tfidf, labels, terms, top_n=TOP_TERMS_N):
    rows=[]
    for c in sorted(np.unique(labels)):
        idx = np.where(labels==c)[0]
        if len(idx)==0:
            rows.append({"cluster":int(c),"size":0,"top_terms":""})
            continue
        sub = X_tfidf[idx]
        avg = np.asarray(sub.mean(axis=0)).ravel()
        top_idx = avg.argsort()[::-1][:top_n]
        top = [terms[i] for i in top_idx if avg[i]>0]
        rows.append({"cluster":int(c),"size":int(len(idx)),"top_terms":", ".join(top)})
    return pd.DataFrame(rows)

def label_from_terms(top_terms):
    t = top_terms.lower()
    scores=[]
    for label, kws in LABEL_RULES:
        s = sum(1 for k in kws if k in t)
        scores.append((label,s))
    scores.sort(key=lambda x: -x[1])
    if scores and scores[0][1]>0:
        return scores[0][0]
    return "Other/Uncategorized"

def sentiment_textblob(text):
    try:
        from textblob import TextBlob
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return None

def sentiment_lexicon(text):
    pos = {"good","improve","safe","clean","support","vaccine","help","recover","successful","aid","protect","treatment","effective"}
    neg = {"death","dead","disease","died","outbreak","contaminat","poor","shortage","crisis","urgent","fatal","loss","cholera","infect","sick","die"}
    words = [w.strip(".,!?:;()[]\"'").lower() for w in str(text).split()]
    if not words: return 0.0
    score = sum(1 for w in words if any(p in w for p in pos)) - sum(1 for w in words if any(n in w for n in neg))
    return float(score)/math.sqrt(len(words))

def polarity_label(p):
    if p is None: return "Neutral"
    if p > 0.12: return "Positive"
    if p < -0.12: return "Negative"
    return "Neutral"

# ----- MAIN WORKFLOW -----
def main():
    t0 = time.time()
    ensure = ensure_data_dir()
    df = load_clean()
    n_docs = len(df)
    print(f"Loaded {n_docs} cleaned articles.")

    # TF-IDF
    print("Computing TF-IDF...")
    tfidf_vec, X_tfidf = compute_tfidf(df["text"].tolist(), max_features=MAX_FEATURES)
    terms = tfidf_vec.get_feature_names_out()
    print("TF-IDF shape:", X_tfidf.shape)

    # SVD reduction
    print("Reducing dimensionality with SVD...")
    svd, X_reduced = reduce_svd(X_tfidf, n_components=SVD_COMPONENTS)
    print("Reduced shape:", X_reduced.shape)

    # Standardize
    X_scaled = StandardScaler().fit_transform(X_reduced)

    # Choose k
    print("Choosing cluster count via silhouette (sampled)...")
    k_best, silhouette_scores = choose_k_by_silhouette(X_scaled, min_k=MIN_K, max_k=MAX_K)
    print(f"Selected k = {k_best}; silhouette scores: {silhouette_scores}")

    # Final clustering
    print("Clustering with MiniBatchKMeans...")
    mb = MiniBatchKMeans(n_clusters=k_best, random_state=RANDOM_STATE, batch_size=100, max_iter=400, n_init=10)
    labels = mb.fit_predict(X_scaled)
    df["cluster"] = labels.astype(int)

    # Top terms per cluster
    print("Computing top terms per cluster...")
    cluster_terms_df = top_terms_for_clusters(X_tfidf, labels, terms, top_n=TOP_TERMS_N)
    cluster_terms_df["friendly_label"] = cluster_terms_df["top_terms"].apply(label_from_terms)
    label_map = dict(zip(cluster_terms_df["cluster"].astype(int), cluster_terms_df["friendly_label"]))
    df["cluster_label"] = df["cluster"].map(label_map).fillna("Other/Uncategorized")

    # LDA topic modeling (CountVectorizer)
    print("Running LDA topic modelling (sklearn)...")
    count_vec = CountVectorizer(stop_words="english", max_features=MAX_FEATURES)
    X_count = count_vec.fit_transform(df["text"].tolist())
    lda = LatentDirichletAllocation(n_components=min(LDA_TOPICS, max(2, n_docs//10)), random_state=RANDOM_STATE, learning_method="batch", max_iter=15)
    lda.fit(X_count)
    lda_terms = count_vec.get_feature_names_out()
    lda_rows=[]
    for topic_idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[::-1][:TOP_TERMS_N]
        top = [lda_terms[i] for i in top_idx]
        lda_rows.append({"topic": int(topic_idx), "top_terms": ", ".join(top)})
    lda_df = pd.DataFrame(lda_rows)

    # Sentiment
    print("Computing sentiment...")
    tb_test = sentiment_textblob("test")
    use_tb = tb_test is not None
    polarities=[]
    for txt in df["text"].tolist():
        p = None
        if use_tb:
            p = sentiment_textblob(txt)
        if p is None:
            p = sentiment_lexicon(txt)
        polarities.append(float(p))
    df["polarity"] = polarities
    df["sentiment"] = df["polarity"].apply(polarity_label)

    # Representative examples per cluster (highest TF-IDF norm)
    def rep_examples(X_tfidf, labels, df, n=5):
        reps=[]
        for c in sorted(np.unique(labels)):
            idx = np.where(labels==c)[0]
            if len(idx)==0: continue
            sub = X_tfidf[idx]
            norms = np.sqrt(sub.multiply(sub).sum(axis=1)).A1
            top_local = np.argsort(norms)[::-1][:n]
            for local in top_local:
                doc_idx = idx[local]
                reps.append({"cluster":int(c), "title": df.iloc[doc_idx].get("title","")[:300], "snippet": df.iloc[doc_idx].get("text","")[:400], "link": df.iloc[doc_idx].get("link","")})
        return pd.DataFrame(reps)
    reps_df = rep_examples(X_tfidf, labels, df, n=3)

    # Summaries and saves (overwrite)
    print("Saving outputs (overwriting existing files)...")
    df.to_csv(OUT_ANALYZED, index=False, encoding="utf-8-sig")
    cluster_terms_df.to_csv(OUT_CLUSTER_TERMS, index=False, encoding="utf-8-sig")
    lda_df.to_csv(OUT_LDA_TOPICS, index=False, encoding="utf-8-sig")
    reps_df.to_csv(DATA_DIR / "cluster_representative_examples.csv", index=False, encoding="utf-8-sig")

    theme_summary = df["cluster_label"].value_counts().reset_index()
    theme_summary.columns = ["theme","count"]
    theme_summary.to_csv(OUT_THEME_SUM, index=False, encoding="utf-8-sig")

    sent_summary = df["sentiment"].value_counts().reset_index()
    sent_summary.columns = ["sentiment","count"]
    sent_summary.to_csv(OUT_SENT_SUM, index=False, encoding="utf-8-sig")

    # Source x theme matrix (if source column exists)
    if "source" in df.columns:
        src_theme = pd.crosstab(df["source"].fillna("Unknown"), df["cluster_label"])
        src_theme.to_csv(OUT_SOURCE_MATRIX, encoding="utf-8-sig")
    else:
        # create empty file
        pd.DataFrame().to_csv(OUT_SOURCE_MATRIX, index=False)

    # PLOTS
    print("Generating plots...")
    # theme bar
    ts = theme_summary.sort_values("count", ascending=False)
    plt.figure(figsize=(10,5)); plt.bar(ts["theme"], ts["count"], color="tab:blue"); plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of articles"); plt.title("Distribution of News Themes"); plt.tight_layout(); plt.savefig(OUT_THEME_PLOT, dpi=300); plt.close()

    # sentiment plot
    ss = sent_summary.sort_values("count", ascending=False)
    colors = ["seagreen" if s.lower()=="positive" else ("indianred" if s.lower()=="negative" else "grey") for s in ss["sentiment"]]
    plt.figure(figsize=(6,4)); plt.bar(ss["sentiment"], ss["count"], color=colors); plt.title("Sentiment distribution"); plt.tight_layout(); plt.savefig(OUT_SENT_PLOT, dpi=300); plt.close()

    # projection 2D
    proj2 = X_reduced[:, :2]
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("tab10")
    for c in sorted(np.unique(labels)):
        pts = proj2[labels==c]
        plt.scatter(pts[:,0], pts[:,1], s=20, color=cmap(c % 10), alpha=0.7, label=f"{label_map.get(int(c),'C'+str(c))} ({(labels==c).sum()})")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small'); plt.title("2D semantic projection of clusters (SVD)"); plt.tight_layout(); plt.savefig(OUT_PROJ, dpi=300); plt.close()

    # trend over time (if published date exists)
    if "published" in df.columns and df["published"].str.strip().any():
        # try parse dates
        try:
            df["published_parsed"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
            tmp = df.dropna(subset=["published_parsed"]).copy()
            if not tmp.empty:
                tmp["date"] = tmp["published_parsed"].dt.date
                trend = tmp.groupby(["date","cluster_label"]).size().unstack(fill_value=0)
                plt.figure(figsize=(10,5))
                for col in trend.columns:
                    plt.plot(trend.index, trend[col], label=col)
                plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small'); plt.title("Daily trend of articles by theme"); plt.tight_layout(); plt.savefig(OUT_TREND, dpi=300); plt.close()
        except Exception:
            # if parsing fails, skip trend
            pass

    # run info
    run_info = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "n_documents": int(n_docs), "k_chosen": int(k_best), "silhouette_sampled": silhouette_scores, "lda_topics": int(min(LDA_TOPICS, max(2, n_docs//10)))}
    with open(OUT_RUN_INFO, "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, indent=2)

    print("Done. Outputs written to data/ (overwritten).")
    print(f"Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
