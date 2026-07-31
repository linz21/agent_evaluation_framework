"""
Candidate question bank for building the golden Q&A dataset — 10 per
category (40 total), organized by category to match the target
distribution in configs/config.yaml.

REDUCED FROM AN ORIGINAL 100-QUESTION PLAN (45/25/20/10) after a real,
practical cost concern was raised: with a $5 Anthropic API budget, the
full pipeline (golden dataset drafting + running Claude Sonnet 4.5 as one
of 3 benchmarked agent versions + LLM-as-judge evaluation across all 3
versions) could total 500-700+ API calls at 100 questions. 40 questions
is a more realistic, still statistically meaningful size given this
constraint — the honest tradeoff is wider bootstrap confidence intervals
and somewhat less statistical power to detect real differences between
agent versions, a real but acceptable limitation given the budget.

Literature questions here are the exact first 10 from the original
45-question list — the same ones already validated during Project 3's
own retrieval testing (confirmed good corpus coverage), and the same
ones already being drafted/reviewed in this session before the scope
was reduced, so no earlier work is wasted.
"""

LITERATURE_QUESTIONS = [
    "How does nitrogen timing affect corn yield?",
    "What is the function of the ZmWRKY74 gene in maize?",
    "How does Aspergillus flavus cause aflatoxin contamination in maize?",
    "What are the applications of UAV imaging in crop monitoring?",
    "How is CRISPR used to improve crop stress tolerance?",
    "What precision agriculture technologies improve crop management?",
    "How does drought stress affect corn physiology?",
    "What role does zinc play in soil-plant systems?",
    "How can biocontrol methods like Bacillus species be used in sustainable agriculture?",
    "What are the effects of heavy metal contamination on agricultural soils?",
]

YIELD_QUESTIONS = [
    ("What corn yield should I expect in Illinois in 2024?", "Illinois", 2024),
    ("What corn yield should I expect in Iowa in 2025?", "Iowa", 2025),
    ("What is the predicted corn yield for Nebraska in 2024?", "Nebraska", 2024),
    ("What yield should I expect for corn in Minnesota in 2025?", "Minnesota", 2025),
    ("What is the forecasted corn yield for Indiana in 2024?", "Indiana", 2024),
    ("How much corn yield can I expect in Ohio in 2025?", "Ohio", 2025),
    ("What is the predicted corn yield for Wisconsin in 2024?", "Wisconsin", 2024),
    ("What corn yield forecast is there for Kansas in 2025?", "Kansas", 2025),
    ("What yield should I plan for corn in Missouri in 2024?", "Missouri", 2024),
    ("What is the expected corn yield for South Dakota in 2025?", "South Dakota", 2025),
]

MULTI_TOOL_QUESTIONS = [
    # (full_question_text, state, year, literature_query)
    #
    # RESTRUCTURED from a plain list of question strings after a real bug
    # was found: the original drafting logic used the FULL compound
    # question as the literature search query, which pulled in "yield
    # prediction methodology" papers (ML models, SHAP values) rather than
    # actual agronomic "practices" papers, since "yield"/"predict"
    # keywords dominated the semantic match. It also never called
    # Project 1's real yield API at all for this category. Both issues
    # are fixed by storing state/year explicitly (same as YIELD_QUESTIONS)
    # and a SEPARATE, narrowly-targeted literature_query per question.
    (
        "What corn yield should I expect in Illinois in 2024, and what farming practices could help improve it?",
        "Illinois", 2024, "What farming practices improve corn yield?"
    ),
    (
        "What yield can I expect for corn in Iowa in 2025, and how does nitrogen timing affect that?",
        "Iowa", 2025, "How does nitrogen timing affect corn yield?"
    ),
    (
        "Given a corn yield forecast for Nebraska in 2024, what agronomic practices support higher yields?",
        "Nebraska", 2024, "What agronomic practices support higher corn yields?"
    ),
    (
        "What corn yield should I expect in Minnesota in 2025, and how could drought stress affect that forecast?",
        "Minnesota", 2025, "How does drought stress affect corn yield?"
    ),
    (
        "What is the predicted yield for Indiana in 2024, and what soil health practices could improve it?",
        "Indiana", 2024, "What soil health practices improve corn yield?"
    ),
    (
        "What corn yield should I expect in Ohio in 2025, and what role does planting density play in yield outcomes?",
        "Ohio", 2025, "What role does planting density play in corn yield?"
    ),
    (
        "Given the corn yield forecast for Wisconsin in 2024, what pest management practices are recommended?",
        "Wisconsin", 2024, "What pest management practices are recommended for corn?"
    ),
    (
        "What yield should I expect in Kansas in 2025, and how does irrigation management affect that?",
        "Kansas", 2025, "How does irrigation management affect corn yield?"
    ),
    (
        "What is the predicted corn yield for Missouri in 2024, and what nitrogen management practices could help?",
        "Missouri", 2024, "What nitrogen management practices improve corn yield?"
    ),
    (
        "What corn yield should I expect in South Dakota in 2025, and how might precision agriculture technologies improve it?",
        "South Dakota", 2025, "What precision agriculture technologies improve crop management?"
    ),
]

OUT_OF_SCOPE_QUESTIONS = [
    "What's the best way to train a dog?",
    "How do I bake a chocolate cake?",
    "What's the weather like in Paris?",
    "Can you recommend a good science fiction novel?",
    "How do I fix a flat bicycle tire?",
    "What's the capital of Australia?",
    "How do I invest in the stock market?",
    "What's a good workout routine for beginners?",
    "How do I learn to play the guitar?",
    "What's the best programming language for web development?",
]
