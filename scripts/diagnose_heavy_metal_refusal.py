"""
Diagnostic for a real finding: Qwen3-4B incorrectly declined "What are
the effects of heavy metal contamination on agricultural soils?" as
out-of-scope, on the leaderboard drill-down. This is the same CLASS of
issue found and diagnosed earlier with Claude's CRISPR refusal (a
scope-boundary heuristic overreacting to generic "agricultural"/"crop"
phrasing rather than "corn" specifically) — just a different question,
different model version. This script checks reproducibility and tests
a corn-specific rephrasing.

Usage:
    python scripts/diagnose_heavy_metal_refusal.py
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, ".")

import yaml

from src.eval.agent_runner import run_question_on_version

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

project3_path = Path(cfg["target_project"]["path"]).resolve()
qwen_version = next(v for v in cfg["agent_versions"] if v["id"] == "qwen3-4b")

questions_to_test = [
    ("Original (as in golden dataset)", "What are the effects of heavy metal contamination on agricultural soils?"),
    ("Corn-specific rephrasing", "What are the effects of heavy metal contamination on corn fields?"),
    ("Original, exact retry (checking determinism)", "What are the effects of heavy metal contamination on agricultural soils?"),
]

for label, question in questions_to_test:
    print("=" * 70)
    print(f"TEST: {label}")
    print(f"QUESTION: {question}")
    print("=" * 70)

    session_id = str(uuid.uuid4())
    result = run_question_on_version(question, session_id, qwen_version, project3_path)

    print(f"\nFULL TRANSCRIPT:\n{result['transcript']}")
    print(f"\nFINAL ANSWER:\n{result['answer']}")
    print(f"\nITERATIONS: {result['iterations']}")
    print()
