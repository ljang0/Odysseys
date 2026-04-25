#!/usr/bin/env python3
"""Convert odysseys.json to OSWorld per-task JSON files."""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def build_example(item: Dict, domain: str) -> Dict:
    task_id = item["task_id"]
    instruction = item["confirmed_task"]
    website = item["website"]

    return {
        "id": task_id,
        "snapshot": "chrome",
        "instruction": instruction,
        "source": website,
        "config": [
            {
                "type": "launch",
                "parameters": {
                    "command": ["google-chrome", "--remote-debugging-port=1337"]
                },
            },
            {
                "type": "launch",
                "parameters": {
                    "command": [
                        "socat",
                        "tcp-listen:9222,fork",
                        "tcp:localhost:1337",
                    ]
                },
            },
            {
                "type": "chrome_open_tabs",
                "parameters": {"urls_to_open": [website]},
            },
        ],
        "trajectory": "trajectories/",
        "related_apps": ["chrome"],
        "evaluator": {"func": "infeasible"},
        "proxy": False,
        "fixed_ip": False,
        "possibility_of_env_change": "high",
        "metadata": {
            "domain": domain,
            "website": website,
            "reference_length": item["reference_length"],
            "level": item["level"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert odysseys.json to OSWorld format"
    )
    parser.add_argument(
        "--input", required=True, help="Path to odysseys.json"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory (examples written under <output-dir>/examples/<domain>/)"
    )
    parser.add_argument(
        "--domain", default="mind2web_chrome", help="Domain name"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        items: List[Dict] = json.load(f)

    examples_dir = Path(args.output_dir) / "examples" / args.domain
    examples_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        example = build_example(item, args.domain)
        out_path = examples_dir / f"{example['id']}.json"
        out_path.write_text(
            json.dumps(example, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {len(items)} task JSONs to {examples_dir}/")


if __name__ == "__main__":
    main()
