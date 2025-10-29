# 🧫 Framing Cholera Prevention in the News: A Thematic Content Analysis of the 2024 Outbreak in Nigeria

## Overview
This repository contains all code, data, and documentation for the research project titled **“Framing Cholera Prevention in the News: A Thematic Content Analysis of the 2024 Outbreak in Nigeria.”**  
The project employs a systematic, Python-based content analysis of online Nigerian news media to explore how cholera prevention and hygiene were discussed during the 2024 cholera outbreak, particularly in **resource-poor settings such as Yenagoa**.

---

## 🎯 Objectives
1. To collect and curate news articles on cholera prevention and outbreak discourse in Nigeria (2024).  
2. To clean and preprocess the data for thematic analysis.  
3. To conduct deep thematic clustering, sentiment, and trend analysis using reproducible Python workflows.  
4. To identify major frames in cholera prevention and their implications for public health communication.

---

## 🧱 Repository Structure

WASH_Cholera_2024_Outbreak_Project/
│
├── data/
│ ├── news_articles.csv # Raw collected news data
│ ├── clean_news.csv # Cleaned and processed dataset
│ ├── cluster_top_terms.csv # Top terms per thematic cluster
│ ├── sentiment_summary.csv # Summary of sentiment scores
│ └── analysis_plots.png # Generated visualisations
│
├── scripts/
│ ├── collect_news.py # Web scraping and news collection
│ ├── clean_news.py # Data cleaning and preprocessing
│ ├── analyze_news.py # Thematic and sentiment analysis
│ └── full_analysis.py # Optional extended analysis workflow
│
├── README.md # Project documentation
└── requirements.txt # Python dependencies


---

## ⚙️ Installation and Setup

### 1. Create and Activate Virtual Environment
In the root folder (`WASH_Cholera_2024_Outbreak_Project`):
```bash
python -m venv venv

Activate it:

Windows:

venv\Scripts\activate


macOS/Linux:

source venv/bin/activate

2. Install Dependencies
pip install -r requirements.txt


If NLTK stopwords are not found, open Python and run:

import nltk
nltk.download('stopwords')

🚀 Workflow and Scripts
Step 1: Collect News Articles
python scripts/collect_news.py


Fetched up to 425 cholera-related articles from Nigerian news sources (2024)
Output: data/news_articles.csv

Step 2: Clean and Prepare Data
python scripts/clean_news.py


Removes duplicates, punctuation, and non-informative text while normalising cases and stopwords.
Output: data/clean_news.csv

Step 3: Thematic, Sentiment, and Cluster Analysis
python scripts/analyze_news.py


Performs:

TF–IDF vectorisation and K-Means clustering for theme discovery

Sentiment analysis using TextBlob

Trend analysis based on publication frequency

Visualisations (barcharts and cluster distributions)

Outputs:

data/cluster_top_terms.csv

data/sentiment_summary.csv

data/analysis_plots.png

📊 Key Findings and Outputs
Thematic Clusters (2024 Outbreak Coverage)

Government accountability and outbreak response

Public health education and hygiene awareness

Infrastructure and WASH service challenges

Community participation and local resilience

Disease surveillance and early warning systems

These clusters represent dominant narratives around cholera prevention and outbreak management during 2024.

Sentiment Distribution
Sentiment	Percentage	Interpretation
Positive	~22%	Highlights proactive prevention and community efforts
Neutral	~45%	Factual and situational reporting
Negative	~33%	Criticism of poor sanitation and slow government response
Trend Insights

Media attention peaked in June–August 2024, corresponding to outbreak escalation.

Most reports originated from Lagos, Abuja, Port Harcourt, and Yenagoa, showing urban reporting bias.

Regional disparities in tone reflected differences in local cholera control capacity.

Visualisations

All figures are embedded in data/analysis_plots.png, showing:

Cluster frequency

Sentiment proportions

Temporal trend of news publications

🧠 Methodology
Study Design

A systematic news-based content analysis was undertaken to explore framing of cholera prevention in Nigerian news coverage during the 2024 outbreak.
Python served as the analytical platform for reproducibility, transparency, and automation.

Data Collection

Tool: Custom Python script (collect_news.py) using Google News queries via requests and BeautifulSoup4.

Keywords: “cholera outbreak Nigeria 2024”, “cholera prevention”, “waterborne disease”, “public health hygiene”, “WASH Nigeria”.

Inclusion Criteria: Articles published between January and December 2024, focused on cholera and prevention discourse.

Exclusion Criteria: Duplicate content, irrelevant disease discussions, and non-news formats.

Data Cleaning

Conducted using clean_news.py:

Duplicate and null-value removal with Pandas

Tokenisation and lemmatisation using NLTK

Removal of stopwords, URLs, punctuation, and non-alphanumeric tokens

Output: clean_news.csv

Thematic Analysis

Implemented in analyze_news.py:

TF–IDF Vectorisation: Captures term importance

K-Means Clustering: Groups articles into thematic clusters

Top-Term Extraction: Reveals high-weight keywords per cluster (cluster_top_terms.csv)

Interpretation: Each cluster labelled through qualitative keyword inspection

Sentiment Analysis

Tool: TextBlob

Generated polarity scores (-1 = negative, +1 = positive).

Mean polarity per article and overall sentiment summary exported to sentiment_summary.csv.

Trend and Visualisation

Publication frequency and theme distribution visualised using Matplotlib and Seaborn.

Output: analysis_plots.png.

Reproducibility

All scripts are modular, with clear dependencies listed in requirements.txt.
Analysts can reproduce results by running scripts sequentially in the scripts/ folder.