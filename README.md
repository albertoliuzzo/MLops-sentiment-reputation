---
title: mlops-sentiment-reputation
emoji: "💬"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Sentiment Reputation Monitoring (MLOps Template)

Progetto MLOps end-to-end per analisi del sentiment e monitoraggio reputazione online.

## Struttura
- `app/`: API FastAPI + inferenza modello
- `tests/`: test automatici per CI (pytest)
- `monitoring/`: conterrà config per Prometheus/Grafana
- `scripts/`: utility (drift baseline, batch inference, ecc.)
- `airflow/`: conterrà DAG per retraining/orchestrazione
