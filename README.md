# 📊 Agent Evaluation Framework

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A statistical benchmarking framework for comparing two configurations
("versions") of the Crop Advisory ReAct Agent (Project 3): local
Qwen3-4B vs. Claude Sonnet 4.5 — the genuinely open question of local
vs. API reliability and cost. (A third version, Qwen2.5-1.5B, was
considered but dropped — its clear synthesis failures were already
well-established from Project 3's own development, which is specifically
what motivated moving to Qwen3-4B in the first place; comparing it again
here wouldn't add new information.) Built on a semi-automated,
human-reviewed golden dataset and rigorous statistical methods (bootstrap
confidence intervals, permutation significance tests) rather than
single-number comparisons.

## Why this project exists

Throughout Project 3's development, real, evidence-based findings emerged
about differences between these configurations — synthesis quality,
fabrication on tool failure, fabrication on memory verification — all
discovered through manual, ad-hoc testing of individual questions. This
project exists to make that comparison **systematic and statistically
rigorous**: a fixed golden dataset, consistent metrics, and formal
significance testing, rather than reading through transcripts by hand.

## Architecture

```
Golden dataset (40 human-reviewed Q&A pairs)
        ↓
Benchmark runner — same 40 questions through both agent versions
        ↓
Metrics: task accuracy (LLM-judge), tool-use efficiency,
         hallucination rate, latency
        ↓
Statistical layer: bootstrap CIs (percentile + BCa) per version,
                    permutation tests + multiple-comparison correction
                    for pairwise significance between versions
        ↓
Streamlit leaderboard — ranked results, drill-down into failure cases
```

## Current Status

**Built and validated:**
- ✅ `src/stats/` — bootstrap CI (percentile and BCa methods), permutation
  test, multiple-comparison correction (Bonferroni + Benjamini-Hochberg).
  All three empirically validated against known test cases (see Results).
- ✅ Golden dataset — 40 questions (10 literature, 10 yield, 10 multi-tool,
  10 out-of-scope), each grounded in real retrieved data or a real API
  call, semi-automated drafting with mandatory human review on every
  single entry. See `data/golden_qa_pairs.json`.

**Not yet built:**
- ⬜ `src/eval/metrics.py` — task accuracy (LLM-as-judge), tool-use
  efficiency, hallucination rate, latency tracking
- ⬜ `src/eval/agent_runner.py` — runs the golden dataset through each of
  Project 3's 2 agent versions
- ⬜ `src/leaderboard/` — Streamlit app for ranked results + failure-case
  drill-down
- ⬜ Actually running the benchmark and analyzing results

## Setup

```bash
# 1. Clone this repo alongside Projects 1-3 as SIBLING directories
git clone https://github.com/linz21/agent_evaluation_framework.git
cd agent_evaluation_framework

# 2. Install dependencies
pip install -r requirements.txt
# Project 2's own dependencies also needed for golden dataset drafting
pip install -r ../agri_rag_literature_ga/requirements.txt

# 3. Set your Anthropic API key (used for drafting + will be used for
#    the Claude agent version + LLM-judge evaluation)
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Validate the golden dataset
python scripts/validate_golden_dataset.py
```

> **Cost note:** the golden dataset was deliberately scoped to 40
> questions (not the originally planned 100) given a real $5 API budget
> constraint — see Results below for the honest statistical tradeoff this
> involves. The benchmark run ahead (40 questions × 3 versions × LLM-judge
> evaluation) will use meaningfully more API budget; consider your own
> limits before running the full pipeline.

## Tech Stack

`numpy` + `scipy` (statistical methods) · Claude Sonnet 4.5 (golden
dataset drafting, one of 2 benchmarked agent versions, and LLM-judge) ·
Project 2's retriever (real literature context for drafting) · Project 1's
live API (real yield data for drafting) · `Streamlit` (planned, for the
leaderboard)

## Results

**Statistics module — validated against known test cases, not just
implemented and trusted:**

| Method | Validation | Result |
|--------|-----------|--------|
| Percentile bootstrap | Known normal distribution (true mean=10) | CI correctly captured true mean |
| BCa bootstrap | Symmetric normal data | Bias/acceleration correction ≈ 0, as expected |
| BCa bootstrap | Skewed proportion (92/100 binary accuracy) | Correction meaningfully differed from percentile — confirms BCa matters for this project's actual metric type |
| Permutation test | Identical distributions | p=0.73, correctly not significant |
| Permutation test | Distributions shifted by 20 | p≈0.0000, correctly significant |
| Permutation test | One-sided variant | Correctly detected direction |
| Bonferroni correction | 3 known p-values [0.01, 0.03, 0.04] | Correctly flagged only p=0.01 as significant |
| Benjamini-Hochberg | Same 3 p-values | Correctly more permissive than Bonferroni (all 3 significant) |

**Golden dataset construction — 6 real bugs found and fixed** (see
`data/golden_qa_pairs.json`'s construction history in commit messages for
full detail): a path-resolution bug, literature context truncated too
aggressively (cutting off abstract conclusions), a drafting token limit
too low (causing real truncated answers to be accidentally accepted
twice before being caught), a genuine Claude API refusal on benign
agricultural nanotechnology vocabulary being silently treated as a valid
answer, the multi-tool category never calling the real yield API at all,
and a stray Chroma index almost committed (same root cause as a bug found
in Project 3).

**Scope reduction, stated honestly:** originally planned as 100 questions
per the initial project spec; reduced to 40 after a real budget
constraint. The honest statistical cost: wider bootstrap confidence
intervals and reduced power to detect real differences between agent
versions via the permutation tests. This is a real, acknowledged
limitation — not hidden — and was a deliberate tradeoff against actually
completing a working, fully-validated benchmark within budget.

## Project Structure

```
agent_evaluation_framework/
├── src/
│   ├── stats/
│   │   ├── bootstrap.py            # Percentile + BCa confidence intervals
│   │   ├── permutation.py          # Non-parametric significance testing
│   │   └── multiple_comparison.py  # Bonferroni + Benjamini-Hochberg
│   ├── eval/
│   │   ├── question_bank.py        # 40 candidate questions, by category
│   │   └── golden_dataset_builder.py  # Real-data-grounded drafting logic
│   └── leaderboard/                 # (planned) Streamlit leaderboard app
├── scripts/
│   ├── build_golden_dataset.py     # Interactive review/build CLI
│   └── validate_golden_dataset.py  # Regression check against known-bad patterns
├── data/
│   └── golden_qa_pairs.json        # The 40-question golden dataset
├── configs/config.yaml             # All settings — single source of truth
└── tests/
```
