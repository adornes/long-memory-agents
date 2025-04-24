# Long Memory Agents

A FastAPI-based service for managing long memory agents.

## Prerequisites

- Python 3.11 or higher
- `uv` package manager
- Docker (optional, for containerized deployment)

## Local Development

1. Install dependencies using `uv`:

```bash
uv sync
```

2. Set up your environment variables in `.env`

3. Run the development server:

```bash
uv run -- uvicorn service.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

- API: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## Docker Deployment

1. Build the Docker image:

```bash
docker build -t long-memory-agents .
```

2. Run the container:

```bash
docker run -p 8000:8000 long-memory-agents
```

3. Run the application using docker-compose (this will also spin up the Vector DB):

```bash
docker-compose up
```

The API will be available at the same URLs as in local development.

## API Endpoints

- `GET /health`: Health check endpoint
  - Returns: `{"status": "healthy"}`

## Development

- The project uses FastAPI for the web framework
- Dependencies are managed through `pyproject.toml`
- Environment variables are loaded from `.env`
- CORS is enabled for all origins in development (should be restricted in production)

## Database Migrations with Alembic

To handle database migrations, we use Alembic. Follow these steps to apply migrations:

1. **Initialize Alembic** (already done):

   ```bash
   uv run -- alembic init alembic
   ```

2. **Create a new migration**:

   ```bash
   uv run -- alembic revision -m "Your migration message"
   ```

3. **Apply the migration**:

   ```bash
   uv run -- alembic upgrade head
   ```

Ensure your `DATABASE_URL` environment variable is set correctly to connect to your database.
