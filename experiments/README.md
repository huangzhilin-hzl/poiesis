# Experiments

Each experiment should be independently inspectable and, when practical,
runnable. Use a descriptive lowercase directory name and begin with the
experiment template:

```text
experiments/<topic>/
├── README.md       # Question, method, evidence, and conclusion
├── src/            # Construction under study
├── tests/          # Correctness checks
└── results/        # Reproducible measurements and raw evidence
```

Keep generated evidence when it is small and essential to the conclusion.
Document how to regenerate large artifacts instead of committing them blindly.

