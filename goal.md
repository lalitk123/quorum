# Goal

Build a REST API for a todo application in this repository.

## Requirements

- Python with FastAPI and SQLite.
- CRUD endpoints for todos: create, list (with completed/pending filter),
  get by id, update, delete.
- Input validation with clear 400/404 error bodies.
- Every endpoint covered by tests (pytest); tests must pass.

## Acceptance criteria

- `pytest` runs green from the repo root.
- `uvicorn app.main:app` starts the server without errors.
- README documents how to run the server and the tests.
