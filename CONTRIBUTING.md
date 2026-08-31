# Contributing

Thank you for improving SplitProof. Open an issue before changing a public API or manifest
schema. For focused fixes, a pull request with a clear rationale is welcome directly.

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest
python -m build
```

Tests must demonstrate determinism and input-order independence for changes to assignment
logic. Never change an existing algorithm under the same version: introduce a new algorithm
version so old manifests remain explainable. Add a changelog entry for user-visible changes.

By participating, you agree to follow the Code of Conduct. Contributions are accepted under
the MIT License.
