"""
Diagnostic for a real, reproducible finding: the agent (Claude Sonnet 4.5)
incorrectly declined "How is CRISPR used to improve crop stress
tolerance?" as out-of-scope, on two independent benchmark runs. This
script re-runs it with full verbose output to see the model's actual
reasoning, plus tests a corn-specific rephrasing to check whether the
generic "crop" (vs "corn") wording is the trigger.

Usage:
    python scripts/diagnose_crispr_refusal.py
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
claude_version = next(v for v in cfg["agent_versions"] if v["id"] == "claude-sonnet-4.5")

questions_to_test = [
    ("Original (as in golden dataset)", "How is CRISPR used to improve crop stress tolerance?"),
    ("Corn-specific rephrasing", "How is CRISPR used to improve corn's stress tolerance?"),
    ("Original, exact retry (checking determinism)", "How is CRISPR used to improve crop stress tolerance?"),
]

for label, question in questions_to_test:
    print("=" * 70)
    print(f"TEST: {label}")
    print(f"QUESTION: {question}")
    print("=" * 70)

    session_id = str(uuid.uuid4())
    result = run_question_on_version(question, session_id, claude_version, project3_path)

    print(f"\nFULL TRANSCRIPT:\n{result['transcript']}")
    print(f"\nFINAL ANSWER:\n{result['answer']}")
    print(f"\nITERATIONS: {result['iterations']}")
    print()
