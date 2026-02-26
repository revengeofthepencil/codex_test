# Codex Test: LangGraph CodeAct Experiment

This is a test application designed to tinker with the `langgraph-codeact` library. It explores autonomous agent behavior using the "CodeAct" pattern, where the agent interacts with a Python execution environment to solve problems.

Most of the core logic and setup for this is pulled directly from the [langgraph-codeact documentation](https://github.com/langchain-ai/langgraph-codeact).

## Features
- **Autonomous Code Execution**: The agent can write and execute Python code to solve complex problems.
- **Readable Terminal Output**: Customized streaming and final result formatting for better visibility.
- **Script Archiving**: Automatically extracts and saves the final Python script used for calculations to the `generated_scripts/` directory with a unique timestamp (e.g., `train_meeting_calculation_202602261100.py`).
- **Serialization Safety**: Filters the execution context to ensure only serializable data is saved by the LangGraph checkpointer.

## Setup and Execution

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

### Prerequisites
- **uv** installed on your system.
- An **OpenAI API key**.

### Installation

1. **Set up the environment**:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

2. **Sync dependencies**:
   Run the following command to create a virtual environment and install all required packages:
   ```bash
   uv sync
   ```

### Running the Application

To execute the main agent:
```bash
uv run main.py
```

The agent will process the challenge, stream its thoughts to the terminal, and archive its final working script in the `generated_scripts` folder.
