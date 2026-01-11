---
title: mlops-sentiment-reputation
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Sentiment reputation monitoring 

Progetto MLOps per monitoraggio reputazione online tramite analisi del sentiment.
Include: model serving con FastAPI, test automatici (CI), deploy (CD) su Hugging Face Space, e monitoraggio (metriche + log) con un controllo semplice di data drift.

## Architettura (high level)

- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (Hugging Face Transformers)
- **Serving**: FastAPI (`/predict`, `/health`, `/docs`)
- **Container**: Docker 
- **CI**: GitHub Actions + pytest
- **CD**: GitHub Actions → push su Hugging Face Space
- **Monitoring**: `/metrics` + log predizioni `predictions.jsonl`
- **Data drift**: baseline + script `drift_check_simple.py`

---

## Link utili

- Repo GitHub: **https://github.com/albertoliuzzo/MLops-sentiment-reputation**
- Hugging Face Space (live): **https://AlbertoLiuzzo-mlops-sentiment-reputation.hf.space**  

---

## Struttura repository

- `app/` → API FastAPI + inferenza modello
- `tests/` → test automatici (pytest) usati in CI
- `.github/workflows/` → pipeline CI e CD
- `monitoring/` → baseline e drift check (versione semplice)
- `Dockerfile` → build container per Hugging Face Space
- `requirements.txt` → dipendenze Python

---

## Dataset pubblico (Kaggle) e inferenza batch

Questo progetto usa un dataset pubblico per eseguire predizioni in batch e validare il modello.
Dataset: Kaggle "Sentiment Analysis Dataset".

