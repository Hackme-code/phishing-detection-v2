# Phishing Link Checker (ICT606 Assessment 3)

A machine learning tool that screens links in a pasted URL or email. It scores each link's **host name** with a character n-gram model (logistic regression) and shows a risk band (Low, Medium, High) with reasons. It does not read the page, path, certificate or email wording.

## Data (not stored in this repo)
- PhiUSIIL Phishing URL dataset (CC BY 4.0): download from the UCI repository and place the CSV in `data/`.
- Tranco list: download from tranco-list.eu into `data/top-1m.csv`.
- `data/real_links.csv` is a small independent test set (legitimate sites, PhishTank hosts and synthetic hosts).

## Run the app
    pip install -r requirements.txt
    streamlit run src/app.py

## Contents
- `notebooks/01_explore.ipynb`: dataset exploration (found that https, www and path determined the label)
- `notebooks/02_model.ipynb`: features, grouped split, models, thresholds, Tranco experiment, real-link check
- `src/features.py`: host feature extractor used in training and the app
- `src/app.py`: Streamlit app
- `models/`: saved models and band settings
- `docs/ai_use_log.md`: AI-use log

## Key limitation
Host-name-only detection has a ceiling. On held-out data the High band flagged about 5% of legitimate hosts, but a small check of well-known sites flagged about a third, because the training data's legitimate class is mostly government, education and non-profit sites. A LOW result does not mean a link is safe.