
# Codex Test: LangGraph CodeAct + Sandbox Experiment

This project demonstrates how to combine LangGraph's CodeAct agent pattern with a secure Python sandbox for autonomous code execution. The setup is ideal for developers experimenting with agentic workflows, safe code evaluation, and tool integration.

This will run 2 operations 
1. The old "2 trains leave cities at `<time>`, when do they meet"
2. A test running PCA on a random list of numbers, which should call a run_pca tool provided via the sandbox. 

## Architecture Overview

- **Sandbox (FastAPI server)**
  - Runs untrusted Python code in a restricted environment
  - Enforces import allowlists, strict builtins, and file access controls
  - Limits execution time and kills runaway processes
  - Exposes a simple HTTP API for code execution and session management

- **Agent (LangGraph + CodeAct)**
  - Uses LangGraph's CodeAct pattern to generate, execute, and reason about Python code
  - Calls the sandbox API to run code securely and retrieve results
  - Supports tool registration (e.g., math, PCA) and context serialization
  - Archives generated scripts and outputs for reproducibility

## Key Features

- Autonomous agent can solve problems by writing and running Python code
- Sandbox prevents dangerous imports, restricts file writes, and enforces timeouts
- Agent streams readable output and saves scripts to `generated_scripts/`
- Only serializable context variables are checkpointed for safety

## Getting Started

1. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

2. (Optional) Set sandbox environment variables for customization:
   - `SANDBOX_ALLOWED_IMPORTS` (comma-separated list)
   - `SANDBOX_MAX_SECONDS` (timeout)
   - `SANDBOX_WORKDIR`, `SANDBOX_OUTPUT_DIR`

3. Start both services (agent and sandbox):
   - If using Docker Compose:
     ```
     docker-compose up --build
     ```
   - Or run each service manually

4. The agent will run, interact with the sandbox, and exit. Scripts are saved in `generated_scripts/`.

## Experimentation Tips

- Add new tools to the agent by defining Python functions and including them in the `tools` list
- Modify sandbox restrictions for different security profiles
- Inspect archived scripts for reproducibility or debugging
- Try different prompts to see how the agent uses tools and code execution

## References

- [LangGraph CodeAct Documentation](https://github.com/langchain-ai/langgraph-codeact)
- [FastAPI](https://fastapi.tiangolo.com/)

---
For questions or suggestions, open an issue or reach out to the project maintainers.
