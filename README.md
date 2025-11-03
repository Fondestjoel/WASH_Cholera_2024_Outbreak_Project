# WASH_Cholera_2024_Outbreak_Project

### Framing Cholera Prevention in the News:  
*A Thematic Content Analysis of the 2024 Cholera Outbreak in Nigeria*

---

## 📘 Overview
This project examined how Nigerian online news outlets framed cholera prevention and outbreak management during the 2024 cholera outbreak.  
It employed a Python-based news mining and content analysis pipeline that collected, cleaned, analyzed, and visualized news data from major Nigerian media sources.

The pipeline is structured around three main scripts:

1. **`collect_news.py`** — fetched cholera-related news articles from Google News RSS feeds.  
2. **`clean_news.py`** — cleaned and preprocessed the collected articles for text analysis.  
3. **`analyze_news.py`** — performed thematic clustering, sentiment analysis, and trend visualization.

---

## 🧩 Workflow

### 1. Data Collection
- Source: Google News RSS feeds (2024)
- Keywords: *cholera*, *outbreak*, *sanitation*, *hygiene*, *water contamination*, *WASH*, *cholera prevention*, *open defecation*, *Bayelsa cholera 2024*
- Output: `data/news_articles.csv` (415 unique articles)

### 2. Data Cleaning
- Performed using **pandas** and **NLTK**
- Removes duplicates, noise, non-relevant text, and empty rows
- Output: `data/clean_news.csv`

### 3. Analysis
Core analyses include:
- **TF-IDF vectorization** — quantifies thematic patterns  
- **K-Means clustering** — identifies distinct news frames  
- **Sentiment polarity** — measures tone using TextBlob  
- **Trend analysis (2024 only)** — visualizes temporal spread of news coverage  
- Outputs: `.png` charts in the `figures/` folder

---

## 📊 Outputs
| Output File | Description |
|--------------|-------------|
| `data/news_articles.csv` | Raw collected articles |
| `data/clean_news.csv` | Cleaned dataset ready for analysis |
| `figures/theme_distribution.png` | Thematic distribution chart |
| `figures/trend_analysis_2024.png` | Publication trend for 2024 outbreak |
| `figures/sentiment_analysis.png` | Sentiment polarity visualization |

---

## ⚙️ Setup

### 1. Clone Repository
```bash
git clone https://github.com/Fondestjoel/WASH_Cholera_2024_Outbreak_Project.git
cd WASH_Cholera_2024_Outbreak_Project
2. Create Virtual Environment
bash
Copy code
python -m venv .venv
.\.venv\Scripts\activate     # On Windows
source .venv/bin/activate    # On Linux/Mac
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Running the Pipeline
bash
Copy code
# Step 1: Collect Articles
python scripts/collect_news.py

# Step 2: Clean Data
python scripts/clean_news.py

# Step 3: Analyze and Visualize
python scripts/analyze_news.py
All results will be saved automatically into the /data and /figures directories.
If existing files are found, they are replaced safely.

🧠 Technical Stack
Language: Python 3.13.7

Editor: Visual Studio Code

OS: Windows 11

Libraries: pandas, nltk, scikit-learn, matplotlib, seaborn, textblob, feedparser

🧾 Citation
Ikiba, O.J. (2025). Framing Cholera Prevention in the News: A Thematic Content Analysis of the 2024 Outbreak in Nigeria.

📜 License

This project is open for academic and non-commercial use with attribution.

📜 License
This project is open for academic and non-commercial use with attribution.
