# Sentiment Reputation Monitoring (MLOps Template)

Progetto MLOps end-to-end per analisi del sentiment e monitoraggio reputazione online.

## Struttura
- `app/`: API FastAPI + inferenza modello
- `tests/`: test automatici (CI)
- `monitoring/`: Prometheus/Grafana
- `scripts/`: utility (drift baseline, batch inference, ecc.)
- `airflow/`: DAG per retraining/orchestrazione
