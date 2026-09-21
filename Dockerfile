# Base Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files to disc & buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and schema
COPY app/ ./app/
COPY sql/ ./sql/
COPY data/ ./data/
COPY .env.example .

# Expose FastAPI application port
EXPOSE 8000

# Run FastAPI with uvicorn server
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
