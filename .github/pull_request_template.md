## Summary

Describe the user-visible change and why it is needed.

## Verification

- [ ] Tests cover determinism, zero leakage, invalid inputs, and compatibility.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest` pass.
- [ ] Wheel and sdist pass integrity and isolated-install smoke checks.
- [ ] Documentation, benchmark evidence, and changelog are updated when behavior changes.
- [ ] No private dataset records or generated build artifacts are included.

## Compatibility and risk

Describe algorithm/schema version, reproducibility, objective, performance, and migration implications.
