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

## Run it with Docker

1. Build the Docker images:

```bash
docker-compose build
```

2. Run the application using docker-compose (this will also spin up the Vector DB):

```bash
docker-compose up
```

3. Set up the Database

If running it for the first time, be sure to create a database following the name in your `.env` file.
The same information must be replicated in `alembic/.env`.
One difference between these two `.env` files is that the one under the root directory needs to use the host `pgvector`, which aligns with `docker-compose` configuration.

With the database created and the `alembic/.env` file configured, **apply the migrations**:

   ```bash
   uv run -- alembic upgrade head
   ```


The API will be available at:

- API: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

4. Health-check

- `GET /health`: Health check endpoint
  - Returns: `{"status": "healthy"}`

## Test it with the client code

In the `client` subdirectory, you'll find a Python client script (`client_with_persistent_memory.py`) that demonstrates the full workflow of the system. This script tests features for persistent memory management and real-time agent interactions.

```bash
cd client
uv sync
uv run client_with_persistent_memory.py
```

The script accepts the following command-line arguments:

- `--similarity-search-threshold`: Sets the minimum cosine similarity score (0.0 to 1.0) for retrieving relevant memories (default: 0.75)
- `--similarity-search-limit`: Maximum number of similar memories to retrieve (default: 5)
- `--verbose`: Enables detailed output showing the agent's knowledge and tool calls

Examples:

```bash
# Run with default settings
uv run client_with_persistent_memory.py

# Run with a lower similarity threshold and more results
uv run client_with_persistent_memory.py --similarity-search-threshold 0.6 --similarity-search-limit 10

# Run with verbose output to see agent's reasoning
uv run client_with_persistent_memory.py --verbose
```

## Development

- The project uses FastAPI for the REST API framework
- Dependencies are managed through `uv`
- Environment variables are loaded from `.env`
- CORS is enabled for all origins in development (should be restricted in production)
