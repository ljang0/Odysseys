# Odysseys

A benchmark of 200 long-horizon web-agent tasks drawn from real browsing sessions and evaluated on the live Internet with rubric-based scoring.

* [Paper (PDF)](https://odysseys-website.pages.dev/assets/paper.pdf)
* [Website](https://odysseys-website.pages.dev/)
* [Leaderboard](https://odysseys-website.pages.dev/leaderboard.html)
* [Task browser](https://odysseys-website.pages.dev/viewer.html)

## Dataset

The 200 tasks live in [`data/odysseys.json`](data/odysseys.json), split into 45 easy, 46 medium, and 109 hard.

Each task has the fields below.

| Field | Description |
|---|---|
| `task_id` | SHA-1 identifier |
| `confirmed_task` | Natural-language prompt given to the agent |
| `website` | Starting URL (always `https://www.google.com`) |
| `level` | `easy`, `medium`, or `hard` |
| `reference_length` | Reference step count |
| `rubrics` | `{R1, R2, ...}` each with `requirement` and `verification` |
| `categories`, `num_categories` | SimilarWeb category tags |

Difficulty is assigned by step count and domain spread. Easy tasks use at most 5 steps and 3 domains. Medium tasks use 6 to 8 steps or 4 or more domains. Hard tasks exceed both thresholds.

## Run

We use the [OSWorld](https://github.com/xlang-ai/OSWorld) runner with Chrome in an Ubuntu VM, a 100-step budget, and maximum reasoning effort.

Convert tasks to per-file OSWorld examples.

```bash
python scripts/python/convert_odysseys_to_osworld.py \
  --input data/odysseys.json \
  --output-dir data/odysseys_cua_final
```

A pre-generated copy of all 200 tasks is checked in at [`data/odysseys_cua_final/`](data/odysseys_cua_final/).

## Score

For each rubric item we prompt an LLM judge (we use `gemini-3.1-flash-lite-preview`) with the trajectory's step screenshots and actions plus the rubric. A rubric is satisfied (1) if any step in the trajectory satisfies it, else 0.

Run the judge over a directory of OSWorld trajectories with [`scripts/python/run_full_trajectory_per_rubric.py`](scripts/python/run_full_trajectory_per_rubric.py). Each run directory must contain a `steps.jsonl` (or `traj.jsonl`) and the referenced screenshots; only runs with a numeric `result.txt` are scored unless `--include-incomplete` is set.

```bash
GEMINI_API_KEY=your-key python scripts/python/run_full_trajectory_per_rubric.py \
  --model gemini-3.1-flash-lite-preview \
  --runs-dir path/to/runs_dir \
  --task-source-json data/odysseys.json \
  --output path/to/eval_results_full_traj_per_rubric.json \
  --num-workers 16
```

OpenAI-compatible models are also supported via `OPENAI_API_KEY` and `--api-base`. API keys may be provided through environment variables, CLI flags, or a `.env` file in the current directory (or via `--env-file path/to/.env`); install `python-dotenv` to enable `.env` loading. Results are written as JSON with per-rubric scores plus an aggregate `summary` broken down by difficulty level.

We report two metrics per task. Averaged is the mean rubric score. Perfect is 1 if and only if every rubric is satisfied.

We also report Trajectory Efficiency, the per-task averaged rubric score divided by step count, averaged across tasks.

$$\text{Traj. Eff.} = \frac{1}{N} \sum_{i=1}^{N} \frac{s_i}{n_i}$$

## Results

Numbers reproduced from Table 2 of the paper at a 100-step budget.

| Model | O-M2W Judge | Rubric Avg | Perfect | Avg. Steps | Traj. Eff. |
|---|:-:|:-:|:-:|:-:|:-:|
| Opus 4.6 | **37.0** | **68.9** | **44.5** | 81.3 | 1.06 |
| GPT-5.4 | 21.5 | 55.4 | 33.5 | 64.4 | **1.15** |
| Sonnet 4.6 | 25.0 | 49.8 | 31.0 | 80.4 | 0.79 |
| GPT-5.4 Mini | 15.5 | 38.4 | 10.5 | **41.7** | 1.12 |
| Qwen-3.5-VL-9B | 18.0 | 42.6 | 13.5 | 78.3 | 0.75 |
| Qwen-3.5-VL-4B | 18.3 | 42.9 | 10.7 | 86.4 | 0.69 |
| Qwen-3.5-VL-35B-A3B | 9.5 | 28.5 | 6.5 | 86.1 | 0.42 |
| UI-TARS-1.5-7B | 2.0 | 10.0 | 1.0 | 76.6 | 0.23 |

Running Opus 4.6 with a 200-step budget raises the perfect rate from 44.5% to 76.5%.

## Citation

```bibtex
@article{jang2026odysseys,
  title={Odysseys: Benchmarking Web Agents on Realistic Long Horizon Tasks},
  author={Jang, Lawrence and Koh, Jing Yu and Fried, Daniel and Salakhutdinov, Ruslan},
  journal={arXiv preprint},
  year={2026}
}
```
