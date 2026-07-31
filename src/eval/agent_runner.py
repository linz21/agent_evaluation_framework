"""
Runs a single golden-dataset question through a specific agent version
(Qwen3-4B or Claude Sonnet 4.5), handling three real complications:

1. NAMESPACE COLLISION: both this project and Project 3 use the
   top-level package name `src` — same issue found building Project 3
   itself and this project's own golden_dataset_builder.py. Handled the
   same way: temporarily swap sys.modules['src'] during the load.

2. WORKING-DIRECTORY COUPLING: Project 3's own code has several
   relative-path assumptions (its config, audit logs, memory files) that
   only resolve correctly when Project 3's directory is the current
   working directory — same issue found in Project 3's own dependency on
   Project 2. Handled with a temporary os.chdir() during the actual
   agent.run() call, not just during import.

3. VERSION SWITCHING: Project 3's ReactAgent reads its LLM provider/model
   from a config file. To run different "versions" through the same
   agent code without modifying Project 3's own checked-in config.yaml,
   this loads that real config, overrides ONLY the llm.provider/model
   fields for the version under test, and writes the result to a
   temporary file passed as config_path — everything else (system
   prompt, max_iterations, memory settings) stays exactly as Project 3
   actually ships it.
"""

import importlib.util
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

_ReactAgent = None
_project3_src_module = None


def _load_project3_agent_class(project3_path: Path):
    """
    Loads Project 3's ReactAgent class AND caches the loaded `src` module
    object itself (not just the class) — needed because ReactAgent does
    LAZY, runtime imports inside its own methods (e.g. `from
    src.memory.redis_memory import SessionMemory` executes when
    agent.run() actually runs, not when this module is first loaded). A
    real test confirmed this: swapping sys.modules['src'] only during
    this initial load, then restoring it immediately after, caused
    "No module named 'src.memory'" and "No module named 'src.guardrails'"
    errors — the swap had already been reverted by the time agent.run()
    made those lazy imports. Fixed by caching the src module object here
    so run_question_on_version() can RE-APPLY the same swap for the
    entire duration of the actual run() call, not just during loading.
    """
    global _ReactAgent, _project3_src_module
    if _ReactAgent is not None:
        return _ReactAgent

    this_project_src = sys.modules.get("src")
    project3_src_init = project3_path / "src" / "__init__.py"
    project3_src_spec = importlib.util.spec_from_file_location(
        "src", project3_src_init, submodule_search_locations=[str(project3_path / "src")]
    )
    project3_src_module = importlib.util.module_from_spec(project3_src_spec)
    sys.modules["src"] = project3_src_module
    _project3_src_module = project3_src_module  # cache for reuse during run()

    original_cwd = os.getcwd()
    try:
        os.chdir(project3_path)

        spec = importlib.util.spec_from_file_location(
            "project3_react_agent", project3_path / "src" / "agent" / "react_agent.py"
        )
        react_agent_module = importlib.util.module_from_spec(spec)
        sys.modules["project3_react_agent"] = react_agent_module
        spec.loader.exec_module(react_agent_module)
        _ReactAgent = react_agent_module.ReactAgent

    finally:
        os.chdir(original_cwd)
        if this_project_src is not None:
            sys.modules["src"] = this_project_src
        else:
            sys.modules.pop("src", None)

    return _ReactAgent


def _build_version_config_file(project3_path: Path, version: dict) -> str:
    """
    Loads Project 3's REAL config.yaml (preserving system prompt,
    max_iterations, memory settings, everything else), overrides only
    the llm provider/model fields for this specific version, and writes
    the result to a temp file. Returns the temp file's path — caller is
    responsible for deleting it after use.
    """
    with open(project3_path / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["llm"]["provider"] = version["provider"]
    if version["provider"] == "local":
        cfg["llm"]["model_name"] = version["model_name"]
    elif version["provider"] == "anthropic":
        cfg["llm"]["anthropic_model"] = version["anthropic_model"]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def run_question_on_version(question: str, session_id: str, version: dict,
                            project3_path: Path) -> dict:
    """
    Runs ONE question through ONE agent version end-to-end. Returns
    Project 3's normal run() result dict (answer, transcript, iterations)
    unchanged — metrics are computed separately (see src/eval/metrics.py)
    from this raw result, keeping this function focused only on the
    cross-project mechanics.

    IMPORTANT: re-applies the sys.modules['src'] swap for the ENTIRE
    duration of agent instantiation AND the run() call, not just while
    loading the ReactAgent class — see _load_project3_agent_class's
    docstring for why this is necessary (lazy runtime imports inside
    ReactAgent's own methods).
    """
    ReactAgent = _load_project3_agent_class(project3_path)
    config_path = _build_version_config_file(project3_path, version)

    this_project_src = sys.modules.get("src")
    original_cwd = os.getcwd()
    try:
        sys.modules["src"] = _project3_src_module
        os.chdir(project3_path)
        agent = ReactAgent(config_path=config_path, session_id=session_id)
        result = agent.run(question, verbose=False)
    finally:
        os.chdir(original_cwd)
        if this_project_src is not None:
            sys.modules["src"] = this_project_src
        else:
            sys.modules.pop("src", None)
        os.unlink(config_path)

    return result
