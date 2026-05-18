#!/usr/bin/env python3
"""Standalone full-trajectory rubric judging with minimal plumbing.

Example:
    GEMINI_API_KEY=your-key python run_full_trajectory_per_rubric.py \
        --model gemini-2.5-flash \
        --runs-dir path/to/runs_dir \
        --task-source-json path/to/tasks.json \
        --output path/to/eval_results_full_traj_per_rubric.json \
        --num-workers 16
"""

import argparse
import asyncio
import base64
import json
import math
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types
from openai import AsyncOpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_MAX_IMAGES = 0  # Keep all images
FINAL_JUDGMENT_MAX_COMPLETION_TOKENS = 8192

FULL_TRAJ_JUDGMENT_SYSTEM = """You are an expert evaluator of web-navigation agent trajectories.

You will receive:
- The user task (for context).
- ONE specific rubric item with a requirement and a verification description.
- The agent's full action history (one line per step).
- Every screenshot from the trajectory, in chronological order.

Your goal is to decide whether this single rubric item is satisfied by the trajectory.

Evaluation rules:
- Judge ONLY the one rubric item you are given; ignore all other implicit requirements.
- Ground your judgment in what the screenshots and actions actually show. Do not invent state.
- Filtering / sorting / form requirements must be applied and confirmed to count as satisfied.
- If the agent was blocked (captcha, access denied, etc.) and therefore could not satisfy the rubric, report failure.

Respond in exactly this format:

Thoughts: <your reasoning, citing specific steps/screenshots>
Status: "success" or "failure"
"""

def append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def make_client(args: argparse.Namespace) -> tuple[str, Any]:
    if args.model.lower().startswith("gemini"):
        api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Gemini API key required. Set GEMINI_API_KEY / GOOGLE_API_KEY or pass --gemini-api-key.")
        return "gemini", genai.Client(api_key=api_key)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OpenAI API key required. Set OPENAI_API_KEY or pass --api-key.")
    kwargs = {"api_key": api_key}
    if args.api_base:
        kwargs["base_url"] = args.api_base
    return "openai", AsyncOpenAI(**kwargs)


def empty_result(run_dir: Path, task_id: str, task: str, rubrics: list[dict[str, Any]], error: str) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "task_id": task_id,
        "task": task,
        "rubric_scores": {str(r.get("id", f"R{i+1}")): 0 for i, r in enumerate(rubrics)} if rubrics else {},
        "rubric_results": [],
        "average_rubric_score": 0.0,
        "perfect": False,
        "error": error,
    }


def infer_task_id(run_dir: Path) -> str:
    name = run_dir.name
    if re.fullmatch(r"[0-9a-f]{32}(?:_\S+)?", name, re.IGNORECASE):
        return name
    match = re.search(r"(?:^|[_-])([0-9a-f]{40}|[0-9a-f]{32})(?:_[0-9]+)?$", name, re.IGNORECASE)
    return match.group(1) if match else name


async def evaluate_run(run_dir: Path, client: Any, backend: str, model: str, task_index: dict[str, dict[str, Any]], max_images: int, max_steps: Optional[int] = None) -> dict[str, Any]:
    task_id = infer_task_id(run_dir)
    meta = task_index[task_id]
    task = meta["task"].strip()
    rubrics = meta["rubrics"]

    try:
        # Extract actions from the OSWorld runner outputs
        if (run_dir / "steps.jsonl").exists():
            grouped: dict[int, dict[str, Any]] = {}
            with open(run_dir / "steps.jsonl", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            for i, row in enumerate(rows):
                step = int(row.get("step_num", i) or i)
                if max_steps is not None and max_steps > 0 and step > max_steps:
                    continue
                merged = grouped.setdefault(step, {"step": step, "screenshot": "", "response": "", "events": [], "_actions": [], "_responses": [], "final": False})
                merged["events"].append(row)
                if row.get("screenshot") or row.get("screenshot_file"):
                    merged["screenshot"] = str(row.get("screenshot") or row.get("screenshot_file"))
                response = str(row.get("response") or "").strip()
                if response:
                    append_unique(merged["_responses"], response)
                    merged["response"] = response
                action = str(row.get("action") or "").strip()
                if action and action.lower() != "screenshot":
                    arguments = row.get("arguments")
                    if isinstance(arguments, dict):
                        arguments = {k: v for k, v in arguments.items() if k != "action"}
                        action_text = f"{action} {json.dumps(arguments, ensure_ascii=True, sort_keys=True)}" if arguments else action
                    else:
                        action_text = action
                else:
                    action_text = str(row.get("action_line") or "").strip()
                    if action_text in ('{"type": "screenshot"}', '{"type":"screenshot"}'):
                        action_text = ""
                append_unique(merged["_actions"], action_text)
                merged["final"] = merged["final"] or row.get("final") is True
            steps = [{**merged, "responses": merged.pop("_responses"), "action_line": " -> ".join(merged.pop("_actions"))} for _, merged in sorted(grouped.items())]
        elif (run_dir / "traj.jsonl").exists():
            grouped = {}
            with open(run_dir / "traj.jsonl", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            for i, row in enumerate(rows, start=1):
                step = int(row.get("step_num", i) or i)
                if max_steps is not None and max_steps > 0 and step > max_steps:
                    continue
                merged = grouped.setdefault(step, {"step": step, "screenshot": "", "response": "", "events": [], "_actions": [], "_fallback": []})
                merged["events"].append(row)
                if (row.get("screenshot") or row.get("screenshot_file")) and not merged["screenshot"]:
                    merged["screenshot"] = str(row.get("screenshot") or row.get("screenshot_file"))
                response = str(row.get("response") or "").strip()
                if response and not merged["response"]:
                    merged["response"] = response
                action = row.get("action")
                if isinstance(action, dict):
                    action_text = str(action.get("command") or action.get("action") or "").strip()
                    is_screenshot = isinstance(action.get("input"), dict) and str(action["input"].get("action", "")).lower() == "screenshot"
                else:
                    action_text, is_screenshot = str(action or "").strip(), False
                append_unique(merged["_fallback"], action_text)
                if not is_screenshot:
                    append_unique(merged["_actions"], action_text)
            steps = [{**merged, "action_line": " -> ".join(merged.pop("_actions") or merged.pop("_fallback"))} for _, merged in sorted(grouped.items())]
        else:
            raise FileNotFoundError(f"No trajectory file found in {run_dir} (expected steps.jsonl or traj.jsonl)")
    except Exception as exc:
        return empty_result(run_dir, task_id, task, rubrics, str(exc))

    if not rubrics:
        return empty_result(run_dir, task_id, task, rubrics, f"No rubric found for task_id={task_id}")
    if not steps:
        return empty_result(run_dir, task_id, task, rubrics, "Empty trajectory - no steps recorded.")
    if not task:
        return empty_result(run_dir, task_id, task, rubrics, "Missing task text. Provide config.json with task, or pass --task-source-json.")

    history_lines = []
    for i, step in enumerate(steps, start=1):
        parts = []
        if str(step.get("response") or "").strip():
            parts.append(f"Response: {str(step['response']).strip()}")
        if str(step.get("action_line") or "").strip():
            parts.append(f"Action: {str(step['action_line']).strip()}")
        if parts:
            history_lines.append(f"{i}. " + "\n".join(parts))
    action_history = "\n".join(history_lines) if history_lines else "No actions recorded."

    resolved_paths = []
    for step in steps:
        if not step.get("screenshot"):
            continue
        raw = Path(str(step["screenshot"]))
        candidates = [raw] if raw.is_absolute() else [run_dir / raw.name, run_dir / raw, raw]
        match = next((path for path in candidates if path.exists()), None)
        if match:
            resolved_paths.append(match)
    screenshot_assets = []
    for path in (resolved_paths[-max_images:] if max_images > 0 else resolved_paths):
        try:
            data = path.read_bytes()
        except Exception:
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        screenshot_assets.append({"bytes": data, "mime": mime, "data_url": f"data:{mime};base64,{base64.b64encode(data).decode()}"})

    rubric_results = []
    for rubric in rubrics:
        rubric_lines = [f"Rubric ID: {rubric.get('id', '?')}", f"Requirement: {str(rubric.get('requirement', '')).strip()}"]
        verification = str(rubric.get("verification", "")).strip()
        if verification:
            rubric_lines.append(f"Verification: {verification}")
        user_text = (
            f"User Task (context only): {task}\n\nEvaluate ONLY this rubric item:\n" + "\n".join(rubric_lines)
            + f"\n\nFull Action History:\n{action_history}\n\nScreenshots attached below: {len(screenshot_assets)} "
            f"(trajectory had {len(steps)} total step(s)).\n\nDecide whether the rubric ({rubric.get('id', '?')}) is satisfied. "
            "Use the required 'Thoughts:' / 'Status:' format."
        )
        try:
            if backend == "gemini":
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)] + [types.Part.from_bytes(data=s["bytes"], mime_type=s["mime"]) for s in screenshot_assets])],
                    config=types.GenerateContentConfig(system_instruction=FULL_TRAJ_JUDGMENT_SYSTEM, max_output_tokens=FINAL_JUDGMENT_MAX_COMPLETION_TOKENS),
                )
                result_text = str(response.text or "").strip()
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": FULL_TRAJ_JUDGMENT_SYSTEM}, {"role": "user", "content": [{"type": "text", "text": user_text}] + [{"type": "image_url", "image_url": {"url": s["data_url"], "detail": "high"}} for s in screenshot_assets]}],
                    max_completion_tokens=FINAL_JUDGMENT_MAX_COMPLETION_TOKENS,
                )
                result_text = str(response.choices[0].message.content or "").strip()
            status_match = re.search(r'Status:\s*["\']?(success|failure)["\']?', result_text, re.IGNORECASE)
            thoughts_match = re.search(r"Thoughts:\s*(.+?)(?:Status:|$)", result_text, re.DOTALL)
            success = bool(status_match and status_match.group(1).lower() == "success")
            reasoning = thoughts_match.group(1).strip() if thoughts_match else result_text.strip() or "Empty judge response."
        except Exception as exc:
            success, reasoning = False, f"Error judging rubric {rubric.get('id', '?')}: {exc}"
        rubric_results.append({
            "rubric_id": rubric.get("id", "?"),
            "requirement": rubric.get("requirement", ""),
            "verification": rubric.get("verification", ""),
            "score": 1 if success else 0,
            "success": success,
            "final_reasoning": reasoning,
        })

    rubric_scores = {item["rubric_id"]: item["score"] for item in rubric_results}
    average = sum(rubric_scores.values()) / len(rubric_scores) if rubric_scores else 0.0
    return {
        "run_dir": str(run_dir),
        "task_id": task_id,
        "task": task,
        "num_steps": len(steps),
        "num_screenshots_sent": len(screenshot_assets),
        "rubric_scores": rubric_scores,
        "rubric_results": rubric_results,
        "average_rubric_score": round(average, 4),
        "perfect": bool(rubric_scores) and all(score == 1 for score in rubric_scores.values()),
    }


async def main_async(args: argparse.Namespace) -> None:
    if not args.runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {args.runs_dir}")

    if (args.runs_dir / "steps.jsonl").exists() or (args.runs_dir / "traj.jsonl").exists():
        discovered = [args.runs_dir]
    else:
        discovered = sorted({path.parent for name in ("steps.jsonl", "traj.jsonl") for path in args.runs_dir.rglob(name)}, key=str)
    if not discovered:
        raise SystemExit(f"No run directories found in: {args.runs_dir}")

    run_dirs = []
    for run_dir in discovered:
        if args.include_incomplete:
            run_dirs.append(run_dir)
            continue
        result_path = run_dir / "result.txt"
        try:
            first_line = result_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[0]
            if math.isfinite(float(first_line.strip())):
                run_dirs.append(run_dir)
        except Exception:
            pass
    print(f"Found {len(discovered)} run(s) in total")
    print(f"Using {len(run_dirs)} run(s) (including incomplete runs)." if args.include_incomplete else f"Using {len(run_dirs)} completed run(s) with valid result.txt; skipping {len(discovered) - len(run_dirs)} incomplete run(s).")
    if not run_dirs:
        raise SystemExit(f"No completed runs found in: {args.runs_dir}")

    task_source = args.task_source_json
    if task_source is None:
        current = args.runs_dir.resolve()
        for _ in range(5):
            args_path = current / "args.json"
            if args_path.is_file():
                try:
                    with open(args_path, encoding="utf-8") as f:
                        task_source_raw = json.load(f).get("mind2web_tasks_path")
                    if isinstance(task_source_raw, str) and task_source_raw.strip():
                        task_source = Path(task_source_raw).expanduser()
                        if not task_source.is_absolute():
                            task_source = (args_path.parent / task_source).resolve()
                        break
                except Exception:
                    break
            if current.parent == current:
                break
            current = current.parent

    task_index = {}
    if task_source and task_source.exists():
        with open(task_source, encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            rubric = item["rubrics"]
            rubrics = [{"id": str(rid), **value} for rid, value in rubric.items()]
            task_index[item["task_id"]] = {"task": item["confirmed_task"], "level": item["level"], "rubrics": rubrics}
    print(f"Loaded {len(task_index)} task(s) from {task_source}" if task_source and task_source.exists() else "Warning: no task source provided; rubric-based evaluation usually needs --task-source-json.")

    output_path = args.output or (args.runs_dir / "eval_results_full_traj_per_rubric.json")
    try:
        existing = {
            str(Path(item["run_dir"])): item
            for item in json.loads(output_path.read_text(encoding="utf-8")).get("tasks", [])
            if isinstance(item, dict) and not item.get("error") and isinstance(item.get("run_dir"), str) and isinstance(item.get("rubric_scores"), dict) and item["rubric_scores"]
        }
    except Exception:
        existing = {}
    pending = [run_dir for run_dir in run_dirs if str(run_dir) not in existing]
    if existing:
        print(f"Loaded {len(existing)} scored run(s) from existing output.")
    print(f"Skipping {len(run_dirs) - len(pending)} already scored run(s); evaluating {len(pending)} run(s).")

    new_results = []
    if pending:
        concurrency = min(max(1, args.num_workers), len(pending))
        print(f"Backend: {'gemini' if args.model.lower().startswith('gemini') else 'openai'} ({args.model}), concurrency={concurrency}")
        backend, client = make_client(args)
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(run_dir: Path) -> dict[str, Any]:
            async with semaphore:
                return await evaluate_run(run_dir, client, backend, args.model, task_index, args.max_images, args.max_steps)

        tasks = [asyncio.create_task(run_one(run_dir)) for run_dir in pending]
        for i, future in enumerate(asyncio.as_completed(tasks), start=1):
            result = await future
            new_results.append(result)
            print(f"Progress: {i}/{len(pending)}, task {result['task_id']} done")

    results_by_key = {str(Path(result["run_dir"])): result for result in new_results}
    results = [results_by_key.get(str(run_dir)) or existing.get(str(run_dir)) for run_dir in run_dirs]
    results = [result for result in results if result is not None]
    all_scores = [score for result in results for score in (result.get("rubric_scores") or {}).values()]
    # Trajectory Efficiency = (1/N) Σ s_i / n_i, where s_i is the per-task average rubric score
    # (in [0,1]) and n_i is num_steps. Tasks with no recorded steps are skipped.
    eff_terms = [r["average_rubric_score"] / r["num_steps"]
                 for r in results
                 if isinstance(r.get("num_steps"), int) and r["num_steps"] > 0
                 and isinstance(r.get("average_rubric_score"), (int, float))]
    traj_eff = sum(eff_terms) / len(eff_terms) if eff_terms else 0.0
    summary = {
        "total_tasks": len(results),
        "total_rubrics": len(all_scores),
        "average_rubric_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "perfect_tasks": sum(1 for result in results if result.get("perfect")),
        "perfect_task_rate": round(sum(1 for result in results if result.get("perfect")) / len(results), 4) if results else 0.0,
        "trajectory_efficiency": round(traj_eff, 6),
        "trajectory_efficiency_x100": round(traj_eff * 100, 4),
        "errored_tasks": sum(1 for result in results if result.get("error")),
    }
    by_level = {}
    for result in results:
        level = task_index[result["task_id"]]["level"]
        stats = by_level.setdefault(level, {"total_tasks": 0, "total_rubrics": 0, "score_sum": 0, "perfect_tasks": 0, "eff_sum": 0.0, "eff_n": 0})
        scores = list(result["rubric_scores"].values())
        stats["total_tasks"] += 1
        stats["total_rubrics"] += len(scores)
        stats["score_sum"] += sum(scores)
        stats["perfect_tasks"] += int(bool(result.get("perfect")))
        n_steps = result.get("num_steps")
        avg_score = result.get("average_rubric_score")
        if isinstance(n_steps, int) and n_steps > 0 and isinstance(avg_score, (int, float)):
            stats["eff_sum"] += avg_score / n_steps
            stats["eff_n"] += 1
    if by_level:
        summary["by_level"] = {
            level: {
                "total_tasks": stats["total_tasks"],
                "total_rubrics": stats["total_rubrics"],
                "average_rubric_score": round(stats["score_sum"] / stats["total_rubrics"], 4) if stats["total_rubrics"] else 0.0,
                "perfect_tasks": stats["perfect_tasks"],
                "perfect_task_rate": round(stats["perfect_tasks"] / stats["total_tasks"], 4) if stats["total_tasks"] else 0.0,
                "trajectory_efficiency": round(stats["eff_sum"] / stats["eff_n"], 6) if stats["eff_n"] else 0.0,
                "trajectory_efficiency_x100": round((stats["eff_sum"] / stats["eff_n"]) * 100, 4) if stats["eff_n"] else 0.0,
            }
            for level, stats in by_level.items()
        }
    payload = {"summary": summary, "tasks": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to: {output_path}")
    if summary["total_rubrics"]:
        print(f"Summary: average_rubric_score={summary['average_rubric_score']:.4f} over {summary['total_rubrics']} rubric(s) in {summary['total_tasks']} task(s); {summary['perfect_tasks']}/{summary['total_tasks']} perfect ({summary['perfect_task_rate']:.1%}); trajectory_efficiency={summary['trajectory_efficiency']:.6f} ({summary['trajectory_efficiency_x100']:.3f} ×100)")
    else:
        print(f"Summary: 0 rubrics scored across {summary['total_tasks']} task(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge each rubric using the full trajectory (all screenshots + action history).")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--task-source-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file to load before reading API keys (default: ./.env if present).")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES, help="Keep the last N screenshots per trajectory; 0 = unlimited.")
    parser.add_argument("--max-steps", type=int, default=0, help="Only consider trajectory rows whose step_num <= this value; default/0 = unlimited.")
    parser.add_argument("--num-workers", type=int, default=1, help="Max concurrent runs.")
    parser.add_argument("--include-incomplete", action="store_true", default=False, help="Include runs without a numeric result.txt.")
    args = parser.parse_args()

    env_path = args.env_file if args.env_file is not None else Path(".env")
    if env_path.is_file():
        if load_dotenv is None:
            raise SystemExit(f"Found {env_path} but python-dotenv is not installed. Install it with `pip install python-dotenv` or remove the file.")
        load_dotenv(dotenv_path=env_path, override=False)
    elif args.env_file is not None:
        raise SystemExit(f"--env-file path does not exist: {env_path}")

    if args.api_key is None:
        args.api_key = os.getenv("OPENAI_API_KEY")
    if args.gemini_api_key is None:
        args.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
