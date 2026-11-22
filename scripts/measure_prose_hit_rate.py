"""Measure how often the prose protocol actually produces a tool call.

The web app does not reliably accept a schema-shaped definition of the tools, so the prose
protocol hands the model a plain-language action list instead and parses an action block out of
the reply. That trades a guarantee for a probability, and this script measures the probability:
it sends a fixed set of agent-style instructions through the bridge and counts how many replies
came back with a usable call.

Usage, with the bridge running and `tool_protocol: "prose"` in its config:

    python scripts/measure_prose_hit_rate.py --trials 5 --out prose-bench.json
    python scripts/measure_prose_hit_rate.py --language zh --trials 5 --out prose-bench-zh.json

Each trial is one completion request, so the run costs `tasks x trials` turns of quota. The
result records the configuration it ran against: with `tool_retry_attempts` above 0 the bridge
re-asks a forced choice that came back without a call, so the hit rate measured here is
end-to-end, not first-shot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "config.yaml"


KIND_SCHEMAS: dict[str, dict[str, Any]] = {
    "text": {"type": "string"},
    "number": {"type": "number"},
    "list of text": {"type": "array", "items": {"type": "string"}},
}


@dataclass(frozen=True)
class Tool:
    """One offered action: a name, what it is for, and its arguments."""

    name: str
    description: str
    arguments: dict[str, tuple[str, bool]] = field(default_factory=dict)
    # argument name -> (kind, required); kind is a key of `KIND_SCHEMAS`

    def payload(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {**KIND_SCHEMAS[kind], "description": name}
                        for name, (kind, _required) in self.arguments.items()
                    },
                    "required": [
                        name for name, (_kind, required) in self.arguments.items() if required
                    ],
                },
            },
        }


TOOLS = {
    tool.name: tool
    for tool in [
        Tool("read_file", "Read a text file", {"path": ("text", True)}),
        Tool("list_dir", "List the entries of a directory", {"path": ("text", True)}),
        Tool(
            "find_files",
            "Find files whose path matches a pattern",
            {"pattern": ("text", True), "path": ("text", False)},
        ),
        Tool(
            "search_text",
            "Search the files of a directory for a string",
            {"query": ("text", True), "path": ("text", False)},
        ),
        Tool(
            "run_command",
            "Run a shell command and return its output",
            {"command": ("text", True), "cwd": ("text", False)},
        ),
        Tool(
            "write_file",
            "Write text to a file, replacing what it held",
            {"path": ("text", True), "content": ("text", True)},
        ),
        Tool("fetch_url", "Fetch a URL and return its text", {"url": ("text", True)}),
        Tool(
            "todo_write",
            "Replace the task list of the current session",
            {"todos": ("list of text", True)},
        ),
    ]
}


@dataclass(frozen=True)
class Task:
    """One instruction, and the actions that could serve it."""

    name: str
    instruction: str
    offered: tuple[str, ...]
    expected: frozenset[str]


TASKS = [
    Task(
        "read a file",
        "Read app/main.py and tell me what the first function does.",
        ("read_file", "find_files", "search_text"),
        frozenset({"read_file"}),
    ),
    Task(
        "list a directory",
        "List what is inside the tests directory.",
        ("list_dir", "find_files", "read_file"),
        frozenset({"list_dir"}),
    ),
    Task(
        "find files by pattern",
        "Find every Python file under the app directory.",
        ("find_files", "list_dir", "search_text"),
        frozenset({"find_files"}),
    ),
    Task(
        "search text",
        "Search the repository for the string TODO and tell me which files mention it.",
        ("search_text", "find_files", "read_file"),
        frozenset({"search_text"}),
    ),
    Task(
        "run a command",
        "Run the test suite of this project and show me the result.",
        ("run_command", "read_file", "list_dir"),
        frozenset({"run_command"}),
    ),
    Task(
        "write a file",
        "Create notes.md containing the single line hello.",
        ("write_file", "read_file", "list_dir"),
        frozenset({"write_file"}),
    ),
    Task(
        "fetch a page",
        "Fetch https://example.com and summarize what it says.",
        ("fetch_url", "write_file", "run_command"),
        frozenset({"fetch_url"}),
    ),
    Task(
        "record todos",
        "Write down three todos for releasing this project.",
        ("todo_write", "write_file", "read_file"),
        frozenset({"todo_write"}),
    ),
    Task(
        "git status",
        "Show me the git status of this repository.",
        ("run_command", "read_file", "list_dir"),
        frozenset({"run_command"}),
    ),
    Task(
        "two steps ahead",
        "Check whether CHANGELOG.md exists, then tell me its first heading.",
        ("read_file", "list_dir", "find_files"),
        frozenset({"read_file", "list_dir", "find_files"}),
    ),
    Task(
        "argument-heavy edit",
        "Append the line 'shipped' to the end of CHANGELOG.md.",
        ("read_file", "write_file", "run_command"),
        frozenset({"read_file", "write_file", "run_command"}),
    ),
    Task(
        "outside the tools",
        "What is the capital of France?",
        ("read_file", "list_dir", "run_command"),
        frozenset({"read_file", "list_dir", "run_command"}),
    ),
]


def bridge_settings(config_path: Path) -> tuple[str, str | None, str, str]:
    """Base url, api key, default model and the tool settings the bridge is running with."""
    raw = yaml.safe_load(config_path.read_bytes().decode("utf-8"))
    server = raw.get("server") or {}
    gemini = raw.get("gemini") or {}
    port = int(server.get("port") or 8000)
    base = f"http://127.0.0.1:{port}"
    protocol = str(gemini.get("tool_protocol", "pascal"))
    language = str(gemini.get("tool_protocol_language", "en"))
    retries = int(gemini.get("tool_retry_attempts", 0))
    return base, server.get("api_key"), protocol, f"{language}/retries={retries}"


def post(base: str, api_key: str | None, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one completion request and return the parsed body, or the error surface."""
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        return {"_error": f"HTTP {error.code}: {detail}"}
    except Exception as error:  # a socket timeout, a refused connection, a broken body
        return {"_error": f"{type(error).__name__}: {error}"}


def run_trial(
    base: str, api_key: str | None, model: str, task: Task, timeout: float
) -> dict[str, Any]:
    """One request for one task, and what came back."""
    started = time.monotonic()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": task.instruction}],
        "tools": [TOOLS[name].payload() for name in task.offered],
        "tool_choice": "required",
        "stream": False,
    }
    reply = post(base, api_key, body, timeout)
    record: dict[str, Any] = {
        "task": task.name,
        "seconds": round(time.monotonic() - started, 1),
    }
    if "_error" in reply:
        record |= {"ok": False, "why": "error", "detail": reply["_error"]}
        return record
    message = (reply.get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    names = [call.get("function", {}).get("name") for call in calls]
    record |= {"called": names, "content": (message.get("content") or "")[:200]}
    if not names:
        record |= {"ok": False, "why": "no call"}
    elif names[0] not in task.offered:
        record |= {"ok": False, "why": "called an action that was not offered"}
    else:
        record |= {"ok": True, "why": "call", "expected": names[0] in task.expected}
    return record


def summarize(records: list[dict[str, Any]], trials: int, settings: str) -> str:
    """A markdown table of the run, with the miss reasons that matter."""
    total = len(records)
    hits = sum(1 for record in records if record["ok"])
    errors = sum(1 for record in records if record["why"] == "error")
    silent = sum(1 for record in records if record["why"] == "no call")
    lines = [
        f"# Prose protocol hit rate ({settings})",
        "",
        f"- requests: {total} ({len(TASKS)} tasks x {trials} trials)",
        f"- a usable call: {hits}/{total} ({hits / total:.0%})",
        f"- answered without a call: {silent}",
        f"- refused or failed: {errors}",
        "",
        "| task | calls | misses | median seconds |",
        "| --- | --- | --- | --- |",
    ]
    for task in TASKS:
        rows = [record for record in records if record["task"] == task.name]
        if not rows:
            continue
        misses = len(rows) - sum(1 for row in rows if row["ok"])
        seconds = sorted(row["seconds"] for row in rows)[len(rows) // 2]
        lines.append(f"| {task.name} | {len(rows) - misses}/{len(rows)} | {misses} | {seconds} |")
    known = [record for record in records if record.get("ok") and not record.get("expected")]
    if known:
        lines += ["", "## Calls that were offered but not the expected one", ""]
        lines += [f"- {record['task']}: {record['called']}" for record in known]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--base-url", default=None, help="overrides the port in the config")
    parser.add_argument("--model", default="gemini-auto")
    parser.add_argument("--trials", type=int, default=5, help="requests per task")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=REPO / "prose-bench.json")
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    base, api_key, protocol, settings = bridge_settings(args.config)
    base = args.base_url or base
    if protocol != "prose":
        print(f"tool_protocol is {protocol!r}; this measures the prose protocol", file=sys.stderr)
        return 2
    print(f"{base}  model={args.model}  {settings}  {len(TASKS)} tasks x {args.trials} trials")

    records: list[dict[str, Any]] = []
    for task in TASKS:
        for trial in range(1, args.trials + 1):
            record = run_trial(base, api_key, args.model, task, args.timeout)
            records.append(record | {"trial": trial})
            mark = "ok" if record["ok"] else f"MISS ({record['why']})"
            print(f"  {task.name:<22} #{trial}  {mark}")
        if not records:
            break

    summary = summarize(records, args.trials, settings)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"raw records: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
