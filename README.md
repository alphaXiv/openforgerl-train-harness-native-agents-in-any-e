# Reproducing OpenForgeRL’s harness-choice finding

This public artifact tests the behavioral claim in [OpenForgeRL (arXiv:2607.21557)](https://arxiv.org/abs/2607.21557) that the inference harness around a model can strongly change agent reliability. The paper’s simpler direct-tool harness, ZeroClaw, reached **48.5% pass@1**, versus **20.9%** for OpenClaw and **32.5%** for Codex. In our matched fixed-model reconstruction, the direct harness reached **71.9% completion and 86.9% required-service coverage**, versus **56.3% and 72.4%** for a stateful planner/subagent harness.

**Assessment: partially reproduced.** The direction is aligned and stable across three seeds, and harness scaffolding sharply changed self-verification and recovery. This is a mechanism test, not a reproduction of the paper’s SFT/GRPO gains: we substituted Qwen2.5-7B-Instruct for Qwen3-30B-A3B-Thinking, 32 public executable tasks for ClawEval, and reconstructed harnesses for ZeroClaw/OpenClaw/Codex.

[Read the illustrated report](reports/harness-choice/report.md) · [Open the self-contained marimo notebook](notebooks/openforgerl_harness_reproduction.py)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/blob/main/notebooks/openforgerl_harness_reproduction.py)

Exact Molab URL: https://molab.marimo.io/github/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/blob/main/notebooks/openforgerl_harness_reproduction.py

## What ran

Qwen2.5-7B-Instruct solved 32 deterministic multi-service tasks per condition. Every task required three to five services and contained a recoverable first-attempt failure. Harness, full versus compact context, verification scaffold, and three sampling seeds formed a balanced 2×2×2×3 design: **24 successful Kubernetes runs and 768 episodes**. Model, base policy prompt, task instructions, tools, sampling, 12-tool-turn limit, and 3,072-token generation budget were held fixed.

Compute was Kubernetes on **NVIDIA RTX PRO 6000 Blackwell** GPUs, with **16 GPUs peak concurrent**. The queue runner measured **21m29s (0.357977 wall hours)** for the complete Kubernetes campaign, including the documented infrastructure-only startup attempts.

## Experiment log

Every formal experiment inherited the exact command shown below. Links point to the public experiment branches; each condition has seeds 0, 1, and 2.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Reader-facing publication surface | Not run as an experiment (publication surface) | Report, notebook, figures, and frozen protocol | — |
| [direct/full/verified seed 0](https://github.com/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/tree/orx/direct-full-verified-seed-0-manifest-fix) | Headline direct harness; full history; verification reminder | `python -u run_eval.py` | 71.9% completion, 86.9% coverage across 3 seeds | Kubernetes, 1 GPU/run |
| [stateful/full/verified seed 0](https://github.com/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/tree/orx/stateful-full-verify-true-seed-0) | Headline planner/subagent harness | `python -u run_eval.py` | 56.3% completion, 72.4% coverage | Kubernetes, 1 GPU/run |
| [direct/full/unverified seed 0](https://github.com/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/tree/orx/direct-full-verify-false-seed-0) | Remove verification scaffold | `python -u run_eval.py` | 80.2% completion; 6.3% self-check incidence | Kubernetes, 1 GPU/run |
| [stateful/full/unverified seed 0](https://github.com/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/tree/orx/stateful-full-verify-false-seed-0) | Remove verification scaffold | `python -u run_eval.py` | 68.8% completion; 0% self-check incidence | Kubernetes, 1 GPU/run |
| [direct/compact/verified seed 0](https://github.com/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/tree/orx/direct-compact-verify-true-seed-0) | Compact to four recent tool events | `python -u run_eval.py` | 66.7% completion; 57.3% self-check incidence | Kubernetes, 1 GPU/run |
| [stateful/compact/verified seed 0](https://github.com/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/tree/orx/stateful-compact-verify-true-seed-0) | Compact planner and executor state | `python -u run_eval.py` | 0% completion; 37.5% mean coverage | Kubernetes, 1 GPU/run |
| [direct/compact/unverified seed 0](https://github.com/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/tree/orx/direct-compact-verify-false-seed-0) | Joint context and verification ablation | `python -u run_eval.py` | 79.2% completion | Kubernetes, 1 GPU/run |
| [stateful/compact/unverified seed 0](https://github.com/alphaXiv/openforgerl-train-harness-native-agents-in-any-e/tree/orx/stateful-compact-verify-false-seed-0) | Joint context and verification ablation | `python -u run_eval.py` | 12.5% completion | Kubernetes, 1 GPU/run |

The first frozen root and one initial retry failed before evaluation because the container lacked a `python` alias; they are excluded from evidence. The committed manifest supplies the alias without changing the frozen run command.

## Reproduce the protocol

The formal command is:

```bash
python -u run_eval.py
```

Configuration lives in `config.json`; Kubernetes shape lives in `.orx/k8s.yaml`. The executable task state machines, typed tools, harnesses, verifiers, and terminal `RESULT_JSON` output are implemented in `run_eval.py`. Aggregate and per-run measurements are under `results/`.

## Artifact map

- `reports/harness-choice/report.md` — concise scientific walkthrough with four evidence figures.
- `notebooks/openforgerl_harness_reproduction.py` — self-contained tutorial and interactive condition explorer.
- `results/condition_summary.csv` — aggregate results over 96 episodes per condition.
- `results/run_summary.csv` — all 24 successful Kubernetes run summaries.
- `results/README.md` — metric definitions and shared compute provenance for every result row.
- `run_eval.py` — executable benchmark and both harness implementations.
