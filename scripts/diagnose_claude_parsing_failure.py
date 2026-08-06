"""
Diagnostic for a real finding: Claude Sonnet 4.5's answer to "How does
Aspergillus flavus cause aflatoxin contamination in maize?" came back as
the fallback string "The agent did not produce a valid response." despite
search_literature succeeding with 5 real sources — suggesting
_parse_final_answer() failed to find "Final Answer:" in the raw
generation. This script re-runs it with full verbose output to see the
actual raw transcript and understand why.

Usage:
    python scripts/diagnose_claude_parsing_failure.py
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

question = "How does Aspergillus flavus cause aflatoxin contamination in maize?"
session_id = str(uuid.uuid4())

result = run_question_on_version(question, session_id, claude_version, project3_path)

print("=" * 70)
print(f"FULL RAW TRANSCRIPT:\n{result['transcript']}")
print("=" * 70)
print(f"\nPARSED FINAL ANSWER:\n{result['answer']}")
print(f"\nITERATIONS: {result['iterations']}")
