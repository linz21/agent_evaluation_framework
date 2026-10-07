# 📊 Agent Evaluation Framework

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A statistical benchmarking framework comparing configurations
("versions") of the Crop Advisory ReAct Agent: local
Qwen3-4B vs. Claude (Sonnet 4.5, and now its replacement Sonnet 5.5) — the
genuinely open question of local vs. API reliability and cost. Built on a semi-automated,
human-reviewed golden dataset and rigorous statistical methods (bootstrap
confidence intervals, permutation significance tests) rather than
single-number comparisons.

**🔗 Live demo:** [crop-agent-eval.streamlit.app](https://crop-agent-eval.streamlit.app/)

## Architecture

```
Golden dataset (40 human-reviewed Q&A pairs)
        ↓
Benchmark runner — same 40 questions through every agent version
        ↓
Metrics: task accuracy (LLM-judge vs. ground truth),
         tool-use correctness/efficiency,
         hallucination/faithfulness (LLM-judge vs. actual retrieved
         context + memory — NOT ground truth, see Results),
         latency
        ↓
Statistical layer: BCa bootstrap CIs per version,
                    paired tests (McNemar / Wilcoxon) for pairwise
                    significance, plus unpaired permutation tests
        ↓
Streamlit leaderboard (streamlit run src/leaderboard/app.py) — ranked
results with confidence intervals, failure-case drill-down
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

# 5. Run the benchmark for each agent version
python scripts/run_benchmark.py --version qwen3-4b
python scripts/run_benchmark.py --version claude-sonnet-5.5   # current Claude version
# (claude-sonnet-4.5 results are already saved; that model retires 2026-11-30,
#  so it can't be re-run after that)

# 6. Run the statistical analysis (unpaired, then paired)
python scripts/analyze_results.py
python scripts/analyze_paired.py
```

## Tech Stack

`numpy` + `scipy` (statistical methods) · Claude Sonnet 4.5 / 5.5 (golden
dataset drafting, two benchmarked agent versions, with Claude Opus as the
LLM judge — a different model from every agent version, to avoid
self-evaluation bias) · Agricultural RAG System's retriever (real literature context) · Corn
Yield Prediction's live API (real
yield data) · `Streamlit` (leaderboard UI)

## Model update (Oct 2026)

Claude Sonnet 4.5 is deprecated (retires 2026-11-30), so the code and config
now target **Claude Sonnet 5.5**: `configs/config.yaml` has a
`claude-sonnet-5.5` agent version, the drafting model is `claude-sonnet-5-5`,
and the API calls were updated for 5.5 (no `temperature`; the reply is read
from the text block because a thinking block may come first; judge defaults
point at `claude-opus-4-8`). The Sonnet 4.5 results are kept and labeled as
historical — they are a record of what was tested, and the leaderboard only
reads saved result files, so it keeps working after 4.5 retires. The golden
dataset was drafted with 4.5 (then human-reviewed).

## Results

### Final Statistical Comparison (n=40 questions per version)

| Metric | Qwen3-4B | Claude Sonnet 4.5 | Claude Sonnet 5.5 |
|---|---|---|---|
| Task Accuracy (higher better) | 0.875 [0.700, 0.925] | 0.825 [0.650, 0.900] | **0.950** [0.800, 0.975] |
| Hallucination Rate (lower better) | 0.125 [0.025, 0.225] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Tool Selection Correctness (higher better) | 0.750 [0.575, 0.850] | 0.925 [0.775, 0.975] | **0.975** [0.850, 1.000] |
| Latency, seconds (lower better) | 180.9 [144.0, 214.0] | 34.1 [24.7, 44.9] | **24.5** [17.7, 32.6] |

(95% BCa bootstrap confidence intervals in brackets; 10,000
bootstrap/permutation iterations, random seed 42.)
### Pairwise significance tests (p-values)

Every version answered the same 40 questions, so the main test is **paired**
(`scripts/analyze_paired.py`): McNemar's exact test for the yes/no metrics,
Wilcoxon signed-rank for latency. The original **unpaired** permutation test
(`scripts/analyze_results.py`) is shown for reference.

| Comparison | Metric | Unpaired p | Paired p |
|---|---|---|---|
| Qwen3-4B vs. Sonnet 5.5 | Task accuracy | 0.418 | 0.375 |
| | Hallucination | 0.056 | 0.063 |
| | Tool selection | 0.006 | **0.004** |
| | Latency | <0.001 | **<0.001** |

**How to read this**

- **Speed is the clearest result.** Both Claude versions are far faster than
  local Qwen3-4B (about 24-34 s vs. 181 s). Part of the latency, especially on
  literature and multi-tool questions, likely comes from the tools themselves
  (the literature tool appears to load and run a local generator), not only
  the model.
- **Tool selection:** Claude picks the right tools more often than Qwen
  (Sonnet 5.5: 9 questions where only 5.5 was right and none the other way).
  For Sonnet 4.5 vs. Qwen the paired test is significant (p = 0.039) while the
  unpaired test is not (p = 0.063) — a narrow result, and the paired analysis
  was added after the first results were seen.
- **Accuracy and hallucination are not significantly different** between
  Qwen and either Claude version. Sonnet 5.5 vs. 4.5 accuracy
  (0.950 vs. 0.825) is borderline (paired p = 0.063).
- Sonnet 5.5 missed 2 of 40 questions, both multi-tool questions.

### Limitations

- 40 questions and a single run per question: intervals are wide, and
  results near p = 0.05 should be treated as borderline.
- Accuracy and hallucination are scored by an LLM judge (Claude Opus) that
  was tested on clear synthetic cases but not measured against human graders.
- Several multi-tool golden answers depend on what the literature corpus
  happened to return, and those questions are hard for every version.


## Project Structure

```
agent_evaluation_framework/
├── src/
│   ├── stats/
│   │   ├── bootstrap.py            # Percentile + BCa confidence intervals
│   │   ├── permutation.py          # Non-parametric significance testing
│   │   ├── multiple_comparison.py  # Bonferroni + Benjamini-Hochberg
│   │   └── paired.py               # McNemar, Wilcoxon, paired sign-flip
│   ├── eval/
│   │   ├── question_bank.py        # 40 candidate questions, by category
│   │   ├── golden_dataset_builder.py  # Real-data-grounded drafting logic
│   │   ├── agent_runner.py         # Cross-project loading + version switching
│   │   └── metrics.py              # All 4 metrics; hallucination check
│   │                                # aligned with RAGAS (see Results)
│   └── leaderboard/
│       └── app.py                  # Streamlit leaderboard + drill-down
├── scripts/
│   ├── build_golden_dataset.py     # Interactive review/build CLI
│   ├── validate_golden_dataset.py  # Regression check
│   ├── validate_metrics.py         # Judge validation (cheap Haiku tests)
│   ├── run_benchmark.py            # Runs both versions, all 4 metrics
│   ├── validate_results.py         # Comprehensive result-file validation
│   ├── inspect_results.py          # Full-detail per-question inspection
│   ├── diagnose_crispr_refusal.py  # Diagnostic for the CRISPR finding
│   ├── analyze_results.py          # Bootstrap CIs + unpaired permutation tests
│   ├── analyze_paired.py           # Paired tests (same questions, both versions)
│   └── retry_failed_judgments.py   # Retries only failed judge calls
├── data/
│   ├── golden_qa_pairs.json        # The 40-question golden dataset
│   └── results/                    # Per-version benchmark results
├── configs/config.yaml             # All settings — single source of truth
└── tests/test_metrics.py           # 13 regression tests
```
