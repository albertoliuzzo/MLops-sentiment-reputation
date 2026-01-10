FROM python:3.11-slim

WORKDIR /app

# Dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Codice
COPY . .

# Hugging Face Spaces usa la variabile PORT (di solito 7860)
ENV PORT=7860

EXPOSE 7860

CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
