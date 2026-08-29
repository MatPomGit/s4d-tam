# Contributing

Open an issue before large changes. Keep dataset converters, algorithm adapters, metrics,
and reporting independent. Add numerical tests for every metric and an end-to-end smoke
test for every new adapter contract.

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest --cov=s4dtam_benchmark
```

Do not commit datasets, model weights without provenance, credentials, restricted logs, or
results containing personal data. Contributions must document units, coordinate frames,
alignment, aggregation level, and upstream software commit.
