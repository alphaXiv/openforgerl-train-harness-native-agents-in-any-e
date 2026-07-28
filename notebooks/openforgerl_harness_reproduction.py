# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14",
#   "matplotlib>=3.8",
#   "pandas>=2.2",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    return mo, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # A fixed model, two harnesses, very different behavior

    **Verdict: partially reproduced.** OpenForgeRL reports that simpler direct-tool harnesses are easier to learn and that harness-aware training changes verification, tool coverage, and recovery. In our fresh fixed-model mechanism test, a direct typed-tool loop completed **71.9%** of matched tasks versus **56.3%** for a stateful planner/subagent loop, while covering **86.9% versus 72.4%** of required services.

    This notebook is self-contained: the successful Kubernetes measurements are embedded below, so opening it in Molab does not rerun the 768 model episodes or require repository-relative files.
    """)
    return


@app.cell
def _(pd):
    rows = [
        {"harness":"direct","context":"full","verification":"on","success":71.875,"coverage":86.875,"full_coverage":46.875,"self_check":2.083,"conditional_recovery":74.194,"tool_turns":7.010,"model_calls":8.010,"prompt_tokens":5155.6,"generated_tokens":312.1,"latency":5.858},
        {"harness":"direct","context":"full","verification":"off","success":80.208,"coverage":90.000,"full_coverage":50.000,"self_check":6.250,"conditional_recovery":80.208,"tool_turns":7.646,"model_calls":8.646,"prompt_tokens":5492.4,"generated_tokens":355.6,"latency":6.135},
        {"harness":"direct","context":"compact","verification":"on","success":66.667,"coverage":81.458,"full_coverage":46.875,"self_check":57.292,"conditional_recovery":68.817,"tool_turns":9.896,"model_calls":10.229,"prompt_tokens":6136.6,"generated_tokens":399.4,"latency":6.660},
        {"harness":"direct","context":"compact","verification":"off","success":79.167,"coverage":87.708,"full_coverage":50.000,"self_check":18.750,"conditional_recovery":79.167,"tool_turns":9.208,"model_calls":9.833,"prompt_tokens":5590.5,"generated_tokens":405.4,"latency":6.742},
        {"harness":"stateful","context":"full","verification":"on","success":56.250,"coverage":72.431,"full_coverage":46.875,"self_check":75.000,"conditional_recovery":56.250,"tool_turns":9.812,"model_calls":20.938,"prompt_tokens":13584.3,"generated_tokens":455.9,"latency":7.463},
        {"harness":"stateful","context":"full","verification":"off","success":68.750,"coverage":77.708,"full_coverage":43.750,"self_check":0.000,"conditional_recovery":69.474,"tool_turns":9.281,"model_calls":19.896,"prompt_tokens":12367.8,"generated_tokens":436.5,"latency":7.116},
        {"harness":"stateful","context":"compact","verification":"on","success":0.000,"coverage":37.500,"full_coverage":0.000,"self_check":0.000,"conditional_recovery":0.000,"tool_turns":12.000,"model_calls":24.000,"prompt_tokens":10578.6,"generated_tokens":334.6,"latency":5.885},
        {"harness":"stateful","context":"compact","verification":"off","success":12.500,"coverage":49.167,"full_coverage":0.000,"self_check":0.000,"conditional_recovery":25.000,"tool_turns":11.927,"model_calls":23.896,"prompt_tokens":9903.9,"generated_tokens":378.2,"latency":6.371},
    ]
    results = pd.DataFrame(rows)
    results
    return (results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. What was held fixed?

    Qwen2.5-7B-Instruct faced 32 executable tasks per condition. Each task touched three to five services and injected one recoverable first-attempt failure. The model, base policy prompt, task text, typed tools, temperature, top-p, 12 tool turns, and 3,072 generated-token budget were fixed.

    The intervention was the control flow:

    - **Direct:** one model process sees tool results and emits the next typed call.
    - **Stateful:** a planner model call delegates an operation; an executor subagent receives a separate context and chooses the typed call.

    Context (full or last four events), a verification reminder, and three seeds form the 2×2×2×3 design. Success comes from final environment state, never from the model saying “done.”
    """)
    return


@app.cell
def _(mo):
    context = mo.ui.dropdown(["full", "compact"], value="full", label="Context")
    verification = mo.ui.dropdown(["on", "off"], value="on", label="Verification scaffold")
    metric = mo.ui.dropdown(
        ["success", "coverage", "self_check", "conditional_recovery", "prompt_tokens"],
        value="success",
        label="Metric",
    )
    mo.hstack([context, verification, metric], justify="start")
    return context, metric, verification


@app.cell
def _(context, metric, plt, results, verification):
    selected = results[
        (results["context"] == context.value)
        & (results["verification"] == verification.value)
    ].copy()
    colors = ["#2563eb" if h == "direct" else "#f59e0b" for h in selected["harness"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(selected["harness"], selected[metric.value], color=colors, width=0.58)
    ax.set_title(f"{metric.value.replace('_', ' ').title()} · {context.value} context · verification {verification.value}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("tokens" if metric.value == "prompt_tokens" else "percent")
    for i, value in enumerate(selected[metric.value]):
        ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The headline comparison

    With full history and verification on, direct completion was 68.8–75.0% across seeds; stateful completion was 50.0–62.5%. Required-service coverage showed the same ordering. The stricter binary “called every required service” rate tied at 46.9%, which is why the report distinguishes mean service coverage from full coverage.

    The result is not simply “stateful does less.” It self-checks dramatically more often—75% of tasks versus 2.1%—but spends more of the fixed interaction budget coordinating and reading state back.
    """)
    return


@app.cell
def _(results):
    headline = results[(results.context == "full") & (results.verification == "on")].set_index("harness")
    comparison = {
        "completion_gap_pp": headline.loc["direct", "success"] - headline.loc["stateful", "success"],
        "coverage_gap_pp": headline.loc["direct", "coverage"] - headline.loc["stateful", "coverage"],
        "prompt_token_ratio": headline.loc["stateful", "prompt_tokens"] / headline.loc["direct", "prompt_tokens"],
        "model_call_ratio": headline.loc["stateful", "model_calls"] / headline.loc["direct", "model_calls"],
        "latency_ratio": headline.loc["stateful", "latency"] / headline.loc["direct", "latency"],
    }
    comparison
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Context and recovery diagnostics

    Compacting history barely changes the direct harness without verification (80.2% → 79.2%). The same intervention collapses the stateful harness (68.8% → 12.5%); with verification it reaches 0%. Planner and executor exhaust the 12-turn limit after losing shared plan state.

    Among episodes that actually encountered the injected failure, matched direct recovery was 74.2% and stateful recovery 56.3%. This supports the paper’s qualitative warning that error recovery remains fragile, while the explicit retry hint explains why our absolute recovery rates exceed its reported 17–26%.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. What this does—and does not—reproduce

    The paper’s exact evidence uses Qwen3-30B-A3B-Thinking, trained OpenForge checkpoints, ClawEval, and real ZeroClaw/OpenClaw/Codex harnesses. Those unreleased assets were unavailable. We therefore call the verdict **partially reproduced**: the selected harness-choice mechanism aligns in a controlled public reconstruction, but the paper’s SFT and GRPO performance gains remain untested.

    **Compute provenance:** 24 successful OpenResearch Kubernetes runs on NVIDIA RTX PRO 6000 Blackwell GPUs, peak 16 concurrent GPUs. The queue runner measured 21m29s (0.357977 wall hours) for the complete Kubernetes campaign, including the documented PATH-only startup attempts. Those attempts occurred before model evaluation and are excluded from scientific metrics.
    """)
    return


if __name__ == "__main__":
    app.run()
