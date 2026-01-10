# Backend Dockerfile pro Techmania Prediction API

FROM python:3.11-slim

# Nastavení pracovního adresáře
WORKDIR /app

# Instalace systémových závislostí
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Kopírování requirements
COPY ./app/requirements.txt .

# Instalace Python závislostí
RUN pip install --no-cache-dir -r requirements.txt

# Kopírování aplikace a src složky
COPY ./app /app
COPY ./src /app/src
COPY ./models /app/models
COPY ./data /app/data

# Přidání src do PYTHONPATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Exponování portu
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Spuštění aplikace
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
