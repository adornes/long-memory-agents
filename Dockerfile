# Use Python 3.11 alpine image as base
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install system dependencies required for uv and psycopg2
RUN apk --no-cache update && apk add --no-cache \
    curl \
    build-base \
    libpq \
    postgresql-dev \
    && rm -rf /var/cache/apk/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .

# Install dependencies using uv
RUN uv sync

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 