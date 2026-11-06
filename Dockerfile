FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY Backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY Backend/main.py .
COPY Backend/agent.py .

# Copy ML models from the dashboard folder where they actually live
COPY dashboard/advanced_model.pkl .
COPY dashboard/le_route.pkl .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
