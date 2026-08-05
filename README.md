# 📊 Agent Evaluation Framework

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A statistical benchmarking framework comparing two configurations
("versions") of the Crop Advisory ReAct Agent: local
Qwen3-4B vs. Claude Sonnet 4.5 — the genuinely open question of local
vs. API reliability and cost. (A third version, Qwen2.5-1.5B, was
considered but dropped — its clear synthesis failures were already
well-established from the Crop Advisory ReAct Agent's own development, which is specifically
what motivated moving to Qwen3-4B in the first place; comparing it again
here wouldn't add new information.) Built on a semi-automated,
human-reviewed golden dataset and rigorous statistical methods (bootstrap
confidence intervals, permutation significance tests) rather than
single-number comparisons.

## Architecture

```
Golden dataset (40 human-reviewed Q&A pairs)
        ↓
Benchmark runner — same 40 questions through both agent versions
        ↓
Metrics: task accuracy (LLM-judge vs. ground truth),
         tool-use correctness/efficiency,
         hallucination/faithfulness (LLM-judge vs. actual retrieved
         context + memory — NOT ground truth, see Results),
         latency
        ↓
Statistical layer: BCa bootstrap CIs per version,
                    permutation test for pairwise significance
        ↓
Streamlit leaderboard (planned) — ranked results, failure-case drill-down
```

## Setup

```bash
# 1. Clone alongside Projects 1-3 as SIBLING directories
git clone https://github.com/linz21/agent_evaluation_framework.git
cd agent_evaluation_framework

# 2. Install dependencies
pip install -r requirements.txt
pip install -r ../agri_rag_literature_ga/requirements.txt   # for golden dataset drafting
pip install -r ../crop_advisory_react_agent/requirements.txt  # for the benchmark runner

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Validate the golden dataset
python scripts/validate_golden_dataset.py

# 5. Run the benchmark for both agent versions
python scripts/run_benchmark.py --version claude-sonnet-4.5
python scripts/run_benchmark.py --version qwen3-4b

# 6. Run the statistical analysis
python scripts/analyze_results.py
```

## Tech Stack

`numpy` + `scipy` (statistical methods) · Claude Sonnet 4.5 (golden
dataset drafting, one of 2 benchmarked agent versions, and LLM-judge,
with Opus as the primary judge to avoid self-evaluation bias) · Project
2's retriever (real literature context) · Project 1's live API (real
yield data) · `Streamlit` (planned, for the leaderboard)

## Results

### Final Statistical Comparison (Qwen3-4B vs. Claude Sonnet 4.5, n=40 each)

| Metric | Qwen3-4B | Claude Sonnet 4.5 | p-value | Significant? |
|---|---|---|---|---|
| Task Accuracy (higher better) | 0.725 [0.525, 0.825] | 0.650 [0.475, 0.775] | 0.633 | No |
| Hallucination Rate (lower better) | 0.100 [0.025, 0.200] | 0.025 [0.000, 0.075] | 0.364 | No |
| Tool Selection Correctness (higher better) | 0.675 [0.475, 0.775] | **0.950** [0.802, 0.975] | **0.004** | **Yes — Claude** |
| Latency, seconds (lower better) | 163.0 [126.4, 199.2] | **32.1** [23.4, 42.6] | **0.000** | **Yes — Claude** |

(95% BCa bootstrap confidence intervals shown in brackets; all values
from 10,000 bootstrap/permutation iterations, random seed 42 for
reproducibility.)

### Hallucination metric design — aligned with RAGAS, not ad-hoc

Confirmed via RAGAS/DeepEval documentation that hallucination/faithfulness
should be judged against the **retrieved context** the model actually had
access to, not ground truth — two separate, established metrics. Our
`judge_hallucination()` checks only against the real tool Observation
content and memory context from that specific run; `judge_task_accuracy()`
separately checks against ground truth. This went through 3 real,
tested design iterations before landing here — see inline docstrings in
`src/eval/metrics.py` for the full iteration history, each triggered by
an actual false positive found during testing.

## Project Structure

```
agent_evaluation_framework/
├── src/
│   ├── stats/
│   │   ├── bootstrap.py            # Percentile + BCa confidence intervals
│   │   ├── permutation.py          # Non-parametric significance testing
│   │   └── multiple_comparison.py  # Bonferroni + Benjamini-Hochberg
│   └── eval/
│       ├── question_bank.py        # 40 candidate questions, by category
│       ├── golden_dataset_builder.py  # Real-data-grounded drafting logic
│       ├── agent_runner.py         # Cross-project loading + version switching
│       └── metrics.py              # All 4 metrics; hallucination check
│                                    # aligned with RAGAS (see Results)
├── scripts/
│   ├── build_golden_dataset.py     # Interactive review/build CLI
│   ├── validate_golden_dataset.py  # Regression check
│   ├── validate_metrics.py         # Judge validation (cheap Haiku tests)
│   ├── run_benchmark.py            # Runs both versions, all 4 metrics
│   ├── inspect_results.py          # Full-detail per-question inspection
│   ├── diagnose_crispr_refusal.py  # Diagnostic for the CRISPR finding
│   ├── analyze_results.py          # Bootstrap CIs + permutation tests
│   └── retry_failed_judgments.py   # Retries only failed judge calls
├── data/
│   ├── golden_qa_pairs.json        # The 40-question golden dataset
│   └── results/                    # Per-version benchmark results
├── configs/config.yaml             # All settings — single source of truth
└── tests/test_metrics.py           # 13 regression tests
```
