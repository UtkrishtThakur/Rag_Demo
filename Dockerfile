FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SENTENCE_TRANSFORMERS_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install local RAG Inspector SDK in editable mode with development dependencies
RUN pip install --no-cache-dir -e "./rag-inspector-sdk[dev]"

CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
