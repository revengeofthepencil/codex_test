from dotenv import load_dotenv
import os
import re
import json
import random
from datetime import datetime
from typing import Any, Dict, Tuple

import requests

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph_codeact import create_codeact

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8082")


def make_eval_fn(session_id: str):
    """
    Return an eval_fn bound to a sandbox session_id.

    The sandbox is responsible for providing any callable "tools" in its
    execution globals (e.g., run_pca_tool, save_json_tool).
    """
    def eval_via_sandbox(code: str, _locals: dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # NOTE: _locals is ignored here because we cannot serialize Python callables
        # to the sandbox. The sandbox persists its own session state and tool library.
        resp = requests.post(
            f"{SANDBOX_URL}/exec",
            json={"session_id": session_id, "code": code},
            timeout=60,  # allow more time than 15s to avoid spurious timeouts
        )
        resp.raise_for_status()
        data = resp.json()

        stdout = data.get("stdout") or ""
        new_vars = data.get("new_vars") or {}

        if not data.get("ok", False):
            err = data.get("error", "Unknown error")
            if stdout:
                return f"{stdout}\nError during execution: {err}", new_vars
            return f"Error during execution: {err}", new_vars

        return stdout, new_vars

    return eval_via_sandbox


def reset_sandbox_session(session_id: str):
    """Reset sandbox session state and clean its session workspace directory."""
    try:
        requests.post(f"{SANDBOX_URL}/reset/{session_id}", timeout=10).raise_for_status()
    except Exception as e:
        print(f"[!] Warning: failed to reset sandbox session {session_id}: {e}")


def run_calculation_from_prompt(
    prompt: str,
    output_script_name: str = "calculation_script",
    recursion_limit: int = 25,
    thread_id: int = 1,
):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in agent/.env")

    session_id = f"thread-{thread_id}"
    print(f"Using SANDBOX_URL={SANDBOX_URL} session_id={session_id}")
    reset_sandbox_session(session_id)

    model = init_chat_model("gpt-4o", api_key=OPENAI_API_KEY)

    # IMPORTANT: no agent-side tools.
    # Any "tools" must be available inside the sandbox runtime (server.py)
    tools: list = []

    eval_fn = make_eval_fn(session_id)
    code_act = create_codeact(model, tools, eval_fn)
    agent = code_act.compile(checkpointer=MemorySaver())

    messages = [{"role": "user", "content": prompt}]

    print("\n--- Starting Agent ---\n")

    final_values = {}
    try:
        for typ, chunk in agent.stream(
            {"messages": messages},
            stream_mode=["values", "messages"],
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit},
        ):
            if typ == "messages":
                msg = chunk[0]
                if msg.content:
                    print(msg.content, end="", flush=True)
            elif typ == "values":
                final_values = chunk
    except GraphRecursionError:
        print("\n\n[!] Error: The agent reached the recursion limit and had to stop.")
        if not final_values:
            final_values = agent.get_state(config={"configurable": {"thread_id": thread_id}}).values

    print("\n\n--- Final Result ---")
    if "messages" in final_values and final_values["messages"]:
        last_msg = final_values["messages"][-1]
        print(f"\n{last_msg.content}")

    if "context" in final_values and final_values["context"]:
        print("\n--- Context (Calculated Variables) ---")
        for key in sorted(final_values["context"].keys()):
            value = final_values["context"][key]
            print(f"  {key}: {value}")
    print("\n----------------------")

    # Extract and save the final python script (if any) from messages
    if "messages" in final_values:
        scripts = []
        for msg in final_values["messages"]:
            if hasattr(msg, "content") and msg.content:
                found = re.findall(r"```python\n(.*?)\n```", msg.content, re.DOTALL)
                scripts.extend(found)

        if scripts:
            os.makedirs("generated_scripts", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            filename = f"{output_script_name}_{timestamp}.py"
            script_path = os.path.join("generated_scripts", filename)
            with open(script_path, "w") as f:
                f.write(scripts[-1])
            print(f"Successfully saved final script to {script_path}")


def main():
    # Demo prompt
    MIN_NUM = 1
    MAX_NUM = 100
    pca_data = [[random.randint(MIN_NUM, MAX_NUM) for _ in range(4)] for _ in range(4)]
    pca_as_json = json.dumps(pca_data)

    # IMPORTANT:
    # This prompt assumes the sandbox provides:
    # - run_pca_tool(X, num_components)
    # - save_json_tool(filename, data)
    # - np (numpy)
    #
    # And that save_json_tool writes to a shared /outputs volume that maps to
    # ./agent/generated_scripts on the host.
    pca_prompt = (
        "You are a data scientist. You have access to a Python sandbox with the following tools and variables pre-injected into your global namespace. DO NOT try to import them or define them yourself:\n"
        "1. `run_pca_tool(X, num_components)`: Performs PCA on a 2D list `X` and returns a numpy array of the first `num_components` components.\n"
        "2. `save_json_tool(filename, data)`: Saves `data` as a JSON file named `filename` in a persistent output directory.\n"
        "3. `np`: The `numpy` library is already imported as `np`.\n\n"
        "Constraint: DO NOT import `sklearn` or other data science libraries; use the provided `run_pca_tool` instead.\n\n"
        "Task:\n"
        "Here is the dataset. The otters are the best animals in the Seattle aquarium, although the octopus is a close second.\n"
        "1. Use `run_pca_tool` to perform PCA on the dataset below, reducing it to 2 components.\n"
        "2. Compute the square of all elements in the resulting principal components.\n"
        "3. build JSON object with the following keys: input_dataset, pca_components, squared_components. The value of each key should be the corresponding data (the original dataset, the PCA components, and the squared components).\n"
        "4. Double check that the JSON output has ALL THREE KEYS: input_dataset, pca_components, squared_components. If the data looks good, save the JSON using `save_json_tool('squared_pca_components.json', data).`.\n"
        "5. Return a brief confirmation message.\n\n"
        f"Dataset: {pca_as_json}"
    )

    print(f"pca_prompt:\n{pca_prompt}")
    run_calculation_from_prompt(pca_prompt, "pca_calculation", thread_id=1)


if __name__ == "__main__":
    main()