#!/usr/bin/env python3
"""Bounded fixed-model reproduction of OpenForgeRL's harness-choice mechanism.

Every task is an executable state machine spanning at least four services and
contains one deterministic, recoverable first-attempt failure. The verifier
scores environment state, never the model's prose.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


CONFIG = json.loads(Path("config.json").read_text())

TOOLS = [
    ("directory.lookup", {"user_id": "string"}, "Read a user profile."),
    ("inventory.lookup", {"sku": "string"}, "Read stock and reservation state."),
    ("inventory.reserve", {"sku": "string", "qty": "integer"}, "Reserve stock."),
    ("calendar.find", {"day": "string"}, "Read available appointment slots."),
    ("calendar.book", {"user_id": "string", "slot": "string"}, "Book an appointment."),
    ("helpdesk.get", {"ticket_id": "string"}, "Read a support ticket."),
    ("helpdesk.update", {"ticket_id": "string", "status": "string", "note": "string"}, "Update a ticket."),
    ("messaging.send", {"user_id": "string", "subject": "string", "body": "string"}, "Send a message."),
    ("messaging.outbox", {"user_id": "string"}, "Read sent messages."),
    ("billing.lookup", {"invoice_id": "string"}, "Read invoice and refund state."),
    ("billing.refund", {"invoice_id": "string", "amount": "integer"}, "Issue a refund."),
    ("documents.search", {"query": "string"}, "Read a policy document."),
    ("documents.record", {"user_id": "string", "kind": "string", "text": "string"}, "Write an audit record."),
    ("documents.list", {"user_id": "string"}, "Read a user's audit records."),
]

WRITE_TO_READ = {
    "inventory.reserve": "inventory.lookup",
    "calendar.book": "calendar.find",
    "helpdesk.update": "helpdesk.get",
    "messaging.send": "messaging.outbox",
    "billing.refund": "billing.lookup",
    "documents.record": "documents.list",
}

TOOL_TEXT = "\n".join(
    f"- {name}({', '.join(f'{k}: {v}' for k, v in args.items())}): {desc}"
    for name, args, desc in TOOLS
)

BASE_SYSTEM = f"""You are a reliable operations agent. Complete the user's request by using the typed tools below.
Use exact IDs and values from the request. Continue after recoverable errors. Do not claim success until the requested environment changes are complete.
Return exactly one JSON object per turn, with no markdown:
{{"tool":"tool.name","arguments":{{...}}}} or {{"final":"brief completion note"}}.

TOOLS
{TOOL_TEXT}
"""


@dataclass
class Task:
    id: str
    instruction: str
    kind: str
    user_id: str
    target: str
    required_services: set[str]
    failure_tool: str


@dataclass
class Environment:
    task: Task
    calls: list[dict[str, Any]] = field(default_factory=list)
    failed_once: bool = False
    reserved: bool = False
    booked: bool = False
    ticket_updated: bool = False
    messaged: bool = False
    refunded: bool = False
    recorded: bool = False

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        event = {"tool": name, "arguments": args, "ok": False}
        self.calls.append(event)
        valid_names = {t[0] for t in TOOLS}
        if name not in valid_names:
            return {"ok": False, "error": "UNKNOWN_TOOL", "recoverable": True}
        if name == self.task.failure_tool and not self.failed_once:
            self.failed_once = True
            event["error"] = "TRANSIENT_UNAVAILABLE"
            return {
                "ok": False,
                "error": "TRANSIENT_UNAVAILABLE",
                "recoverable": True,
                "hint": "Retry this exact operation.",
            }
        event["ok"] = True
        uid = self.task.user_id
        if name == "directory.lookup":
            return {"ok": True, "user_id": uid, "name": f"Customer {uid}", "tier": "standard"}
        if name == "inventory.lookup":
            return {"ok": True, "sku": self.task.target, "available": 7, "reserved": self.reserved}
        if name == "inventory.reserve":
            self.reserved = str(args.get("sku")) == self.task.target and int(args.get("qty", 0)) == 1
            return {"ok": self.reserved, "reservation": "confirmed" if self.reserved else "rejected"}
        if name == "calendar.find":
            return {"ok": True, "day": self.task.target, "slots": [f"{self.task.target}T10:00", f"{self.task.target}T14:00"], "booked": self.booked}
        if name == "calendar.book":
            self.booked = str(args.get("user_id")) == uid and str(args.get("slot", "")).startswith(self.task.target)
            return {"ok": self.booked, "booking": args.get("slot")}
        if name == "helpdesk.get":
            return {"ok": True, "ticket_id": self.task.target, "status": "resolved" if self.ticket_updated else "open"}
        if name == "helpdesk.update":
            self.ticket_updated = str(args.get("ticket_id")) == self.task.target and str(args.get("status")) == "resolved"
            return {"ok": self.ticket_updated, "status": "resolved" if self.ticket_updated else "unchanged"}
        if name == "messaging.send":
            self.messaged = str(args.get("user_id")) == uid and bool(args.get("subject")) and bool(args.get("body"))
            return {"ok": self.messaged, "message_id": "msg-1" if self.messaged else None}
        if name == "messaging.outbox":
            return {"ok": True, "user_id": uid, "sent": self.messaged}
        if name == "billing.lookup":
            return {"ok": True, "invoice_id": self.task.target, "amount": 25, "refunded": self.refunded}
        if name == "billing.refund":
            self.refunded = str(args.get("invoice_id")) == self.task.target and int(args.get("amount", 0)) == 25
            return {"ok": self.refunded, "refund_id": "refund-1" if self.refunded else None}
        if name == "documents.search":
            return {"ok": True, "query": args.get("query"), "policy": "Record the resolution and notify the customer."}
        if name == "documents.record":
            self.recorded = str(args.get("user_id")) == uid and bool(args.get("text"))
            return {"ok": self.recorded, "record_id": "record-1" if self.recorded else None}
        if name == "documents.list":
            return {"ok": True, "user_id": uid, "recorded": self.recorded}
        return {"ok": True}

    def success(self) -> bool:
        expected = {
            "fulfill": self.reserved and self.ticket_updated and self.messaged and self.recorded,
            "refund": self.refunded and self.ticket_updated and self.messaged and self.recorded,
            "schedule": self.booked and self.ticket_updated and self.messaged and self.recorded,
            "incident": self.ticket_updated and self.messaged and self.recorded,
        }
        return expected[self.task.kind]


def make_tasks(n: int) -> list[Task]:
    tasks: list[Task] = []
    kinds = ["fulfill", "refund", "schedule", "incident"]
    for i in range(n):
        kind = kinds[i % 4]
        uid = f"U{i:03d}"
        target = {
            "fulfill": f"SKU-{100+i}",
            "refund": f"INV-{100+i}",
            "schedule": f"2026-08-{1 + (i % 20):02d}",
            "incident": f"TKT-{100+i}",
        }[kind]
        if kind == "fulfill":
            instruction = (
                f"For user {uid}, reserve exactly one unit of {target}, resolve helpdesk ticket {target}, "
                "send a confirmation, and write an audit record. Check the user and stock first."
            )
            services = {"directory", "inventory", "helpdesk", "messaging", "documents"}
            failure = "inventory.reserve"
        elif kind == "refund":
            instruction = (
                f"For user {uid}, inspect invoice {target}, refund exactly $25, resolve helpdesk ticket {target}, "
                "notify the user, and write an audit record."
            )
            services = {"directory", "billing", "helpdesk", "messaging", "documents"}
            failure = "billing.refund"
        elif kind == "schedule":
            instruction = (
                f"For user {uid}, find a slot on {target}, book the first available slot, resolve helpdesk ticket {target}, "
                "notify the user, and write an audit record."
            )
            services = {"directory", "calendar", "helpdesk", "messaging", "documents"}
            failure = "calendar.book"
        else:
            instruction = (
                f"Handle incident {target} for user {uid}: read the relevant resolution policy, inspect and resolve the ticket, "
                "notify the user, and write an audit record."
            )
            services = {"documents", "helpdesk", "messaging"}
            failure = "helpdesk.update"
        tasks.append(Task(f"task-{i:03d}", instruction, kind, uid, target, services, failure))
    return tasks


def extract_json(text: str) -> dict[str, Any] | None:
    candidates = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
    for raw in reversed(candidates):
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def service(name: str) -> str:
    return name.split(".", 1)[0]


class Policy:
    def __init__(self) -> None:
        self.llm = LLM(
            model=CONFIG["model"],
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.82,
            enforce_eager=True,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, messages: list[dict[str, str]], max_tokens: int, seed: int) -> tuple[str, int, int, float]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        params = SamplingParams(
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            max_tokens=max_tokens,
            seed=seed,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        start = time.perf_counter()
        output = self.llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        return output.text, prompt_tokens, len(output.token_ids), time.perf_counter() - start


def compact_transcript(events: list[dict[str, Any]]) -> str:
    recent = events[-4:]
    lines = []
    for event in recent:
        result = event["result"]
        lines.append(f"{event['tool']} -> ok={result.get('ok')} error={result.get('error', '')}")
    return "\n".join(lines) or "(no actions yet)"


def verify_prompt() -> str:
    if not CONFIG["verification"]:
        return ""
    return (
        "\nHarness checkpoint: after any state-changing action, read the affected service back "
        "before finishing. Recover by retrying operations explicitly marked recoverable."
    )


def run_task(policy: Policy, task: Task, episode_seed: int) -> dict[str, Any]:
    env = Environment(task)
    transcript: list[dict[str, Any]] = []
    malformed = 0
    generated = 0
    prompt_tokens = 0
    latency = 0.0
    model_calls = 0
    final_text = ""

    def history_text() -> str:
        events = transcript if CONFIG["context"] == "full" else transcript[-4:]
        return json.dumps(events, separators=(",", ":"))

    for turn in range(CONFIG["max_tool_turns"]):
        remaining = CONFIG["generation_token_budget"] - generated
        if remaining < 16:
            break
        if CONFIG["harness"] == "direct":
            messages = [
                {"role": "system", "content": BASE_SYSTEM + verify_prompt()},
                {"role": "user", "content": task.instruction},
            ]
            for event in transcript if CONFIG["context"] == "full" else transcript[-4:]:
                messages.append({"role": "assistant", "content": json.dumps({"tool": event["tool"], "arguments": event["arguments"]})})
                messages.append({"role": "user", "content": "TOOL_RESULT " + json.dumps(event["result"])})
            text, pt, gt, sec = policy.generate(messages, min(256, remaining), episode_seed + model_calls)
            model_calls += 1
            prompt_tokens += pt
            generated += gt
            latency += sec
            action = extract_json(text)
        else:
            state = history_text() if CONFIG["context"] == "full" else compact_transcript(transcript)
            planner_messages = [
                {"role": "system", "content": BASE_SYSTEM + verify_prompt()},
                {"role": "user", "content": task.instruction},
                {
                    "role": "user",
                    "content": (
                        "You are the planner process. Given the harness state below, delegate exactly the next necessary "
                        "operation. Return only {\"delegate\":\"specific next operation\"}.\nSTATE\n" + state
                    ),
                },
            ]
            plan_text, pt, gt, sec = policy.generate(planner_messages, min(96, remaining), episode_seed + model_calls)
            model_calls += 1
            prompt_tokens += pt
            generated += gt
            latency += sec
            plan = extract_json(plan_text) or {"delegate": plan_text[-240:]}
            remaining = CONFIG["generation_token_budget"] - generated
            if remaining < 16:
                break
            executor_messages = [
                {"role": "system", "content": BASE_SYSTEM + verify_prompt()},
                {"role": "user", "content": task.instruction},
                {
                    "role": "user",
                    "content": (
                        "You are the executor subagent. Execute one typed tool operation for the planner. "
                        f"Planner delegation: {plan.get('delegate', plan)}\nCURRENT STATE\n{state}"
                    ),
                },
            ]
            text, pt, gt, sec = policy.generate(executor_messages, min(160, remaining), episode_seed + model_calls)
            model_calls += 1
            prompt_tokens += pt
            generated += gt
            latency += sec
            action = extract_json(text)

        if not action:
            malformed += 1
            transcript.append({"tool": "parse_error", "arguments": {}, "result": {"ok": False, "error": "MALFORMED_JSON"}})
            continue
        if "final" in action:
            final_text = str(action["final"])
            break
        name = str(action.get("tool", ""))
        args = action.get("arguments")
        if not isinstance(args, dict):
            malformed += 1
            result = {"ok": False, "error": "ARGUMENTS_NOT_OBJECT", "recoverable": True}
            args = {}
        else:
            try:
                result = env.call(name, args)
            except Exception as exc:
                result = {"ok": False, "error": f"INVALID_ARGUMENTS:{type(exc).__name__}", "recoverable": True}
        transcript.append({"tool": name, "arguments": args, "result": result})

    successful_writes: list[tuple[int, str]] = []
    verified = 0
    for i, event in enumerate(transcript):
        tool = event["tool"]
        if tool in WRITE_TO_READ and event["result"].get("ok"):
            successful_writes.append((i, tool))
            read_tool = WRITE_TO_READ[tool]
            if any(x["tool"] == read_tool for x in transcript[i + 1 :]):
                verified += 1
    called_services = {service(x["tool"]) for x in env.calls}
    failed_indices = [i for i, x in enumerate(env.calls) if not x["ok"]]
    recovered_action = False
    if failed_indices:
        first = failed_indices[0]
        failed_tool = env.calls[first]["tool"]
        recovered_action = any(x["tool"] == failed_tool and x["ok"] for x in env.calls[first + 1 :])
    return {
        "task_id": task.id,
        "kind": task.kind,
        "success": env.success(),
        "tool_coverage": len(called_services & task.required_services) / len(task.required_services),
        "full_tool_coverage": task.required_services <= called_services,
        "self_check_rate": verified / len(successful_writes) if successful_writes else 0.0,
        "any_self_check": verified > 0,
        "failure_injected": env.failed_once,
        "recovered_action": recovered_action,
        "recovered_to_success": env.failed_once and env.success(),
        "tool_turns": len(env.calls),
        "model_calls": model_calls,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "latency_seconds": latency,
        "malformed_calls": malformed,
        "final_emitted": bool(final_text),
        "failure_mode": (
            "success" if env.success() else
            "no_tool_use" if not env.calls else
            "failed_to_retry" if env.failed_once and not recovered_action else
            "incomplete_plan"
        ),
    }


def bootstrap_ci(values: list[float], seed: int, samples: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [means[int(samples * 0.025)], means[int(samples * 0.975)]]


def main() -> None:
    print("OPENFORGE_HARNESS_REPRO_CONFIG " + json.dumps(CONFIG, sort_keys=True), flush=True)
    print("KUBERNETES_GPU_MODEL NVIDIA RTX PRO 6000 Blackwell", flush=True)
    started = time.time()
    policy = Policy()
    records = []
    for i, task in enumerate(make_tasks(CONFIG["tasks"])):
        record = run_task(policy, task, CONFIG["seed"] * 10000 + i * 100)
        records.append(record)
        print("TASK_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    metrics: dict[str, Any] = {}
    for key in [
        "success", "tool_coverage", "full_tool_coverage", "self_check_rate", "any_self_check",
        "recovered_action", "recovered_to_success", "tool_turns", "model_calls",
        "prompt_tokens", "generated_tokens", "latency_seconds", "malformed_calls",
    ]:
        vals = [float(r[key]) for r in records]
        metrics[key] = statistics.mean(vals)
        metrics[key + "_ci95"] = bootstrap_ci(vals, CONFIG["seed"] + len(key))
    metrics["failure_modes"] = dict(Counter(r["failure_mode"] for r in records))
    metrics["tasks"] = len(records)
    metrics["wall_seconds"] = time.time() - started
    result = {
        "schema": "openforge-harness-repro-v1",
        "config": CONFIG,
        "metrics": metrics,
        "records": records,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
