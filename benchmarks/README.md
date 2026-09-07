# SplitProof benchmarks

`benchmark_fixture.py` runs repeated group-safe k-fold and holdout stability
over the checked-in support-message fixture. It records the source digest,
environment, timing, and traced memory. Generated scale benchmarks remain
separate and are not presented as external-dataset equivalence.

```bash
python benchmarks/benchmark_fixture.py
```
