# Result provenance and metric definitions

Every row in this directory comes from a successful OpenResearch **Kubernetes** run on an **NVIDIA RTX PRO 6000 Blackwell** GPU. Peak concurrency was **16 GPUs**. The complete successful evidence window was **15m15s (0.2542 wall hours)**, from 2026-07-28 07:29:12Z through 07:44:27Z.

`run_summary.csv` contains 24 runs (three seeds for each of eight conditions). `condition_summary.csv` aggregates 96 episodes per condition. The fixed command was `python -u run_eval.py`.

- `success`: final executable environment state satisfied every requested change.
- `tool_coverage`: fraction of required services called at least once.
- `full_tool_coverage`: episode called every required service.
- `any_self_check`: at least one successful write was followed by the paired read tool.
- `self_check_rate`: fraction of successful writes later read back.
- `retry_action`: failed operation was later retried successfully.
- `recovered_to_success`: task succeeded after its injected failure.
- `conditional_recovery`: successful recoveries divided by episodes that actually encountered the injected failure.

The infrastructure-only runs that failed before model evaluation are excluded.
