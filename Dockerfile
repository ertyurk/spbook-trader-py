FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY app/ ./app/

# Install Python dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install fastapi uvicorn redis aioredis pandas scikit-learn pydantic sqlalchemy asyncpg numpy python-dateutil

# Activate virtual environment for subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]