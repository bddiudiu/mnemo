FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml README.md ./
COPY mnemo/ mnemo/

RUN pip install --no-cache-dir -e ".[openai]"

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8080

ENV MEMORI_CHROMA_DIR=/app/data/chroma
ENV MEMORI_DB_URL=sqlite+aiosqlite:///app/data/memori.db

CMD ["python", "-m", "uvicorn", "mnemo.api:app", "--host", "0.0.0.0", "--port", "8080"]
