FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use capital B to match your GitHub folder
COPY Backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Backend/main.py .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
