# ITArcheologyAI
AI thats recognize archeological artefacts on created dataset by automatic algorithm and work with it.

---

## Docker & Requirements
- For running the **server/API** (in Docker), use `requirements.txt` (lightweight, no GUI libs):

```
docker build -t artifact-api .
```

- For local development with GUI and training tools, install `requirements-ml.txt` which includes `PySide6` and other desktop/training dependencies.
- For development tools (linters, tests) use `requirements-dev.txt`.

**CI:** A GitHub Actions workflow was added at `.github/workflows/ci.yml` that runs formatting checks, linting, tests, and builds the Docker image on push/PR to `main`.

---

If you want, I can add a small `Makefile` or `scripts/` helpers for common commands (build, run, lint, test).
