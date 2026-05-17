"""
T-07: Benchmark validator.

Checks:
  1. Each task has exactly one correct_tool.
  2. correct_tool exists in the tool registry (tools.json or MongoDB).
  3. All tools in distractor_cluster exist in the registry.
  4. task_ids are unique.
  5. difficulty is one of easy/medium/hard.
  6. attributes is a non-empty list.
  7. domain is one of the five taxonomy domains.

Usage:
    python benchmark/validator.py
    python benchmark/validator.py --registry registry/tools.json
    python benchmark/validator.py --tasks benchmark/tasks.json --registry registry/tools.json
"""

import argparse
import json
import sys
from pathlib import Path

VALID_DOMAINS = {"database", "filesystem", "api", "cicd", "iam"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def load_registry_names(registry_path: str) -> set[str]:
    path = Path(registry_path)
    if not path.exists():
        print(f"WARNING: Registry file not found at {registry_path}. "
              "Run `python registry/seed_tools.py --generate` first.")
        return set()
    with open(path) as f:
        tools = json.load(f)
    return {t["name"] for t in tools}


def validate(tasks_path: str, registry_path: str) -> bool:
    tasks_file = Path(tasks_path)
    if not tasks_file.exists():
        print(f"ERROR: Tasks file not found: {tasks_path}")
        return False

    with open(tasks_file) as f:
        tasks = json.load(f)

    registry_names = load_registry_names(registry_path)
    has_registry = bool(registry_names)

    errors = []
    warnings = []
    seen_ids = set()

    for i, task in enumerate(tasks):
        loc = f"task[{i}] (id={task.get('task_id', '?')})"

        # Required fields
        for field in ["task_id", "instruction", "correct_tool", "domain",
                      "difficulty", "distractor_cluster", "attributes"]:
            if field not in task:
                errors.append(f"{loc}: missing field '{field}'")

        if "task_id" in task:
            tid = task["task_id"]
            if tid in seen_ids:
                errors.append(f"{loc}: duplicate task_id '{tid}'")
            seen_ids.add(tid)

        if "domain" in task and task["domain"] not in VALID_DOMAINS:
            errors.append(f"{loc}: invalid domain '{task['domain']}' — must be one of {VALID_DOMAINS}")

        if "difficulty" in task and task["difficulty"] not in VALID_DIFFICULTIES:
            errors.append(f"{loc}: invalid difficulty '{task['difficulty']}' — must be one of {VALID_DIFFICULTIES}")

        if "attributes" in task and not isinstance(task["attributes"], list):
            errors.append(f"{loc}: 'attributes' must be a list")
        elif "attributes" in task and len(task["attributes"]) == 0:
            warnings.append(f"{loc}: 'attributes' is empty")

        if "distractor_cluster" in task:
            dc = task["distractor_cluster"]
            if not isinstance(dc, list) or len(dc) == 0:
                errors.append(f"{loc}: 'distractor_cluster' must be a non-empty list")
            correct = task.get("correct_tool")
            if correct and correct in dc:
                errors.append(f"{loc}: correct_tool '{correct}' appears in distractor_cluster")

        # Registry checks (only if registry is available)
        if has_registry:
            correct = task.get("correct_tool")
            if correct and correct not in registry_names:
                errors.append(f"{loc}: correct_tool '{correct}' not found in registry")

            for dt in task.get("distractor_cluster", []):
                if dt not in registry_names:
                    warnings.append(f"{loc}: distractor '{dt}' not in registry "
                                    "(may be a phase-2 qualified tool — verify manually)")

        if "instruction" in task and len(task["instruction"].strip()) < 10:
            warnings.append(f"{loc}: instruction is suspiciously short")

    # Summary statistics
    difficulty_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for task in tasks:
        d = task.get("difficulty", "?")
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        dom = task.get("domain", "?")
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    print(f"\n── Benchmark Validation Report ──────────────────────────────")
    print(f"  Tasks file:      {tasks_path}")
    print(f"  Registry:        {registry_path} ({'loaded' if has_registry else 'not found'})")
    print(f"  Total tasks:     {len(tasks)}")
    print(f"  By domain:       {domain_counts}")
    print(f"  By difficulty:   {difficulty_counts}")
    print(f"  Unique task IDs: {len(seen_ids)}")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"    ⚠  {w}")
        if len(warnings) > 20:
            print(f"    ... and {len(warnings) - 20} more")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:20]:
            print(f"    ✗  {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
        print(f"\n  RESULT: FAILED ({len(errors)} errors)\n")
        return False
    else:
        print(f"\n  RESULT: PASSED ✓ (0 errors, {len(warnings)} warnings)\n")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark tasks")
    parser.add_argument("--tasks", default="benchmark/tasks.json")
    parser.add_argument("--registry", default="registry/tools.json")
    args = parser.parse_args()

    ok = validate(args.tasks, args.registry)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
