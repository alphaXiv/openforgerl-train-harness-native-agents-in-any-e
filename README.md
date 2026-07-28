# OpenForgeRL harness-choice reproduction

Fresh, public reproduction workspace for arXiv:2607.21557. Formal results will be added only after successful Kubernetes runs.

## Protocol

This repository contains a bounded, executable multi-service tool-use suite with deterministic recoverable failures. It compares a direct typed-tool loop with a reconstructed planner/subagent control flow using the same public model, base policy prompt, task instructions, tools, sampling settings, tool-turn limit, and generation-token budget. Context management and verification scaffolding are crossed as ablations.

The formal evaluation command is:

```bash
python -u run_eval.py
```

The experiment configuration is committed in `config.json`; every OpenResearch experiment runs the same command.
