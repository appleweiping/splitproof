# SplitProof benchmarks

`benchmark_fixture.py` runs repeated group-safe k-fold and holdout stability
over the checked-in support-message fixture. It records the source digest,
environment, timing, and traced memory. Generated scale benchmarks remain
separate and are not presented as external-dataset equivalence.

This is a small authored example, recorded as `checked-in-example`. Historical
outputs labelled `fixture-real` refer to this same local fixture and are not
external real-dataset validation or evidence of model performance.

```bash
python benchmarks/benchmark_fixture.py
```
