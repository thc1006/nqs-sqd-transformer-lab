# Plan New NQS + SQD Experiment

When invoked, help Ting-Yi design a new experiment in this repository.

1. Ask for (or infer) the following:
   - target molecule and encoding (e.g. H₂ 12-bit),
   - whether the experiment is for:
     - baseline SQD,
     - FFNN NQS sampler,
     - Transformer NQS sampler,
     - hybrid / comparison.
2. Propose a **single** experiment with:
   - clear hypothesis,
   - specific configuration (referencing an existing or new YAML in configs/),
   - approximate runtime on a single RTX 4090,
   - what metric(s) will be used to evaluate success.
3. Suggest:
   - file to create or modify under `src/experiments/`,
   - how to log results into `results/raw/` and `results/processed/`,

   - what plots to generate into `results/figures/`.

End with a concrete todo list that can be completed in 1–2 focused coding
sessions.
