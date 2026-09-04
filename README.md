# Poiesis

> What I cannot create, I do not understand.

**Understanding through making.**

Poiesis is a laboratory for learning by construction. It turns questions into
small, inspectable artifacts: reproductions, experiments, implementations, and
explanations whose claims can be tested.

The name comes from the Greek *poiesis* (ποίησις): making or bringing something
into being. Here, creation is not merely the result of understanding; it is the
method by which understanding is earned.

## The family

- **Euboulia** — deliberate well: decide what is worth pursuing.
- **Poiesis** — make to understand: explore an idea by constructing it.
- **Entelechy** — actualize potential: turn possibility into demonstrated
  capability.

Together:

```text
Euboulia          Poiesis                 Entelechy
good deliberation -> understanding by making -> realized potential
```

## Method

Every investigation follows the same loop:

```text
question -> model -> build -> measure -> explain -> better question
```

1. State a precise question.
2. Build the smallest artifact that could answer it.
3. Compare its behavior with a trusted reference or explicit expectation.
4. Measure before making a claim.
5. Explain what the construction revealed, including failures and limits.

## What belongs here

- Small implementations made to understand an idea from first principles.
- Reproductions of papers, systems, algorithms, and performance claims.
- Focused experiments that distinguish competing explanations.
- Notes that connect source material to something runnable or observable.
- Failed attempts when they sharpen the question or rule out a hypothesis.

This is not a bookmark collection or a gallery of unexplained demos. An
artifact belongs here when another person can inspect how it changed what we
know.

## Layout

```text
Poiesis/
├── experiments/  # Runnable investigations and their evidence
├── notes/        # Source-grounded explanations and synthesis
└── templates/    # Reusable investigation records
```

Start an investigation from
[`templates/experiment.md`](templates/experiment.md).

## Principles

- Build before claiming understanding.
- Prefer the smallest faithful reconstruction.
- Keep observations separate from interpretations.
- Preserve enough context to reproduce a result.
- Treat failure as evidence, not as missing history.
- End with the next question.

