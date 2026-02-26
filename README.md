# Codex Test: LangGraph CodeAct Experiment

This is a test application designed to tinker with the `langgraph-codeact` library. It explores autonomous agent behavior using the "CodeAct" pattern, where the agent interacts with a Python execution environment to solve problems.

Most of the core logic and setup for this is pulled directly from the [langgraph-codeact documentation](https://github.com/langchain-ai/langgraph-codeact).

## Features
- **Autonomous Code Execution**: The agent can write and execute Python code to solve complex problems.
- **Readable Terminal Output**: Customized streaming and final result formatting for better visibility.
- **Script Archiving**: Automatically extracts and saves the final Python script used for calculations to the `generated_scripts/` directory with a unique timestamp (e.g., `train_meeting_calculation_202602261100.py`).
- **Serialization Safety**: Filters the execution context to ensure only serializable data is saved by the LangGraph checkpointer.

## What this is
- `sandbox/`: a hardened-ish local Python code runner (FastAPI) with:
  - strict builtins
  - import allowlist
  - file writes only under /workspace/<session_id>/
  - timeout + process kill
- `agent/`: LangGraph + langgraph-codeact agent that calls the sandbox over HTTP.

## Setup
1) Put your OpenAI key in `agent/.env`:
   OPENAI_API_KEY=...

2) Run:
   docker-compose up --build

You'll see logs from both services. The agent runs once and exits.
Generated scripts are saved to: `agent/generated_scripts/`
