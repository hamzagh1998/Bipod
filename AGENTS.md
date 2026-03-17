# Repository Guidelines

## Project Structure & Module Organization
- `app/` contains the FastAPI backend: `api/` routes and schemas, `services/` domain logic, `db/` models/database setup, and `core/` config/logging.
- `frontend/` contains static client assets for chat and studio pages (`index.html`, `studio.html`, `js/`, `css/`).
- `imagine/` hosts the image-generation service and model preload scripts.
- `tests/` contains pytest coverage for API contracts, router/brain behavior, and runtime config defaults.
- `docker/` contains Dockerfiles; `docker-compose.yaml` wires `bipod-app`, `ollama`, and `imagine`.
- `data/` stores runtime artifacts (memory DB, vectors, generated files, local model data).

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install backend dependencies.
- `uvicorn app.main:app --host 0.0.0.0 --port 4444 --reload`: run the API locally with reload.
- `docker compose up -d`: start the full local stack.
- `docker compose logs -f`: stream service logs for debugging.
- `pytest -q`: run the test suite.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and explicit type hints.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Prefer `async`/`await` for I/O-heavy service paths.
- Keep imports ordered: standard library, third-party, local modules.
- Use structured logging via `app/core/logger.py` (`get_logger(...)`) instead of `print`.

## Testing Guidelines
- Use `pytest`; place tests under `tests/` with filenames `test_*.py`.
- Name test functions by behavior, e.g. `test_chat_stream_requires_existing_conversation`.
- Update or add tests whenever API responses, routing behavior, or service contracts change.
- Run `pytest -q` before opening a PR.

## Commit & Pull Request Guidelines
- Follow the existing commit pattern: strongly prefer Conventional Commit prefixes (`feat:`, `fix:`, `chore:`, `docs:`).
- Keep commits focused to one logical change.
- PRs should include a concise summary, touched areas, test evidence (command + result), and screenshots for frontend/studio UI changes.

## Security & Configuration Tips
- Keep secrets in `.env` (for example `HF_TOKEN`) and never commit credentials.
- Preserve local-first behavior; avoid adding external network dependencies without explicit configuration guards.
