# AGENTS.md

## Project Objective
Build a course-project-level scientific paper summarization agent in Python.

The system should automatically process a scientific paper, identify its main sections, extract key academic information, generate structured summaries, produce a concise academic-style final summary, and perform a basic faithfulness check to reduce unsupported claims.

The project should prioritize:
- reliability
- interpretability
- modular design
- ease of explanation in a course presentation
- straightforward local execution

## Scope
The system is intended for scientific paper summarization, especially for research papers in technical domains such as Statistics and Biology.

Current project scope includes:
1. document loading
2. section identification
3. key information extraction
4. structured summary generation
5. final academic summary generation
6. basic faithfulness / support checking

Out of scope unless explicitly requested:
- large-scale production deployment
- overly complex orchestration frameworks
- full factual verification systems
- heavy optimization for scale
- unnecessary frontend development

## dataset
- pip install aclsum

## Run with Real LLM (DeepSeek-compatible)

Set environment variables before running:

```bash
export SMART_LLM__API_KEY="<your-api-key>"
export SMART_LLM__BASE_URL="https://api.deepseek.com/v1"
export SMART_LLM__MODEL_NAME="deepseek-chat"
```

Then run:

```bash
python -m src.main
```

Notes:
- The pipeline automatically reads `SMART_LLM__API_KEY`, `SMART_LLM__BASE_URL`, and `SMART_LLM__MODEL_NAME`.
- If an LLM call fails, the system falls back to the local heuristic summarizer.
- Do not hardcode API keys in code or config files.

## Engineering Principles
1. Keep the implementation simple and modular.
2. Prefer robust, explainable solutions over complex designs.
3. Make the pipeline easy to inspect and debug.
4. Preserve clear separation of concerns across modules.
5. Use readable Python with moderate abstraction.
6. Avoid unnecessary dependencies.
7. Prefer course-project-sized solutions over industrial overengineering.
8. When multiple valid options exist, choose the simplest robust one.

## Expected Repository Structure
Unless the repository already has a better structure, prefer organizing files like this:

```text
summarization-agent/
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml
│   └── prompts.yaml
├── data/
│   ├── samples/
│   └── outputs/
├── scripts/
│   ├── run_demo.py
│   └── evaluate.py
├── src/
│   ├── main.py
│   ├── pipeline.py
│   ├── agent/
│   │   ├── controller.py
│   │   └── state.py
│   ├── parser/
│   │   ├── document_loader.py
│   │   └── section_splitter.py
│   ├── extractor/
│   │   └── key_info_extractor.py
│   ├── summarizer/
│   │   ├── structured_summarizer.py
│   │   └── final_summarizer.py
│   ├── verifier/
│   │   └── faithfulness_checker.py
│   ├── llm/
│   │   ├── client.py
│   │   └── prompts.py
│   └── utils/
│       └── io.py
├── tests/
│   ├── test_parser.py
│   └── test_pipeline.py
└── docs/
    ├── literature_review.md
    └── experiment_notes.md
