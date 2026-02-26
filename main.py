from dotenv import load_dotenv
import builtins
import contextlib
import io
import os
import re
import json
import random
import numpy as np
from datetime import datetime
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from langgraph_codeact import create_codeact

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
load_dotenv()


def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # Store original keys before execution
    original_keys = set(_locals.keys())

    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, builtins.__dict__, _locals)
        result = f.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"
    except Exception as e:
        result = f"Error during execution: {repr(e)}"

    # Determine new variables created during execution
    new_keys = set(_locals.keys()) - original_keys
    
    # Filter for serializable values only (exclude modules, types, and functions)
    new_vars = {}
    for key in new_keys:
        val = _locals[key]
        if isinstance(val, (int, float, str, bool, list, dict, set, tuple, type(None))):
            new_vars[key] = val
            
    return result, new_vars


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

# pulled from https://www.askpython.com/python/examples/principal-component-analysis
def run_pca_tool(X: list[list[float]], num_components: int) -> np.ndarray:
    """
    Perform Principal Component Analysis (PCA) on a dataset.

    Args:
        X (list[list[float]]): The input dataset as a 2D list or array-like structure, where each row is a sample and each column is a feature.
        num_components (int): The number of principal components to retain.

    Returns:
        np.ndarray: The reduced dataset with shape (n_samples, num_components).

    Example:
        >>> data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        >>> reduced = run_pca(data, 2)
        >>> print(reduced.shape)
        (3, 2)
    """
    arr = np.array(X)
    print(f"Running PCA on dataset with shape {arr.shape} to reduce to {num_components} components.")

    # Step 1: Center the data
    X_meaned = arr - np.mean(arr, axis=0)

    # Step 2: Compute covariance matrix
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step 4: Sort eigenvalues and eigenvectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Step 5: Select subset of eigenvectors
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step 6: Transform the data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced

tools = [
    add,
    run_pca_tool
]


def format_message(message: Any) -> str:
    """Format a LangChain message for readable terminal output."""
    role = "Unknown"
    if hasattr(message, "type"):
        role = message.type.capitalize()
    
    content = message.content
    if not content and hasattr(message, "tool_calls") and message.tool_calls:
        content = f"Tool Calls: {message.tool_calls}"
    
    return f"\n[bold]{role}:[/bold] {content}"


def run_calculation_from_prompt(prompt, output_script_name = "calculation_script", recursion_limit=25):
    model = init_chat_model("gpt-4o", api_key=OPENAI_API_KEY)
    code_act = create_codeact(model, tools, eval)
    agent = code_act.compile(checkpointer=MemorySaver())
    
    messages = [{
        "role": "user",
        "content": prompt
    }]

    print("\n--- Starting Agent ---\n")
    
    final_values = {}
    try:
        for typ, chunk in agent.stream(
            {"messages": messages},
            stream_mode=["values", "messages"],
            config={"configurable": {"thread_id": 1}, "recursion_limit": recursion_limit},
        ):
            if typ == "messages":
                msg = chunk[0]
                if msg.content:
                    print(msg.content, end="", flush=True)
            elif typ == "values":
                final_values = chunk
    except GraphRecursionError:
        print("\n\n[!] Error: The agent reached the recursion limit and had to stop.")
        # Attempt to grab the latest values from the checkpointer if final_values is empty
        if not final_values:
            final_values = agent.get_state(config={"configurable": {"thread_id": 1}}).values

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

    # Extract and save the final python script
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
    # let's try something simple with extraneous information to see if the agent can focus on the relevant details
    train_prompt = "A train leaves Chicago for Los Angeles at 3:00PM traveling 50mph. The conductor has cold pizza for breakfast and everyone on the train agrees his breath smells horrible. Another train leaves Los Angeles for Chicago at 3:22PM travelling 40mpg. Asssuming they run on parallel tracks, what time will they meet if the conductor drinks a double espresso? The distance between Chicago and Los Angeles is 2,017 miles. Time Bandits is my favorite movie."
    run_calculation_from_prompt(train_prompt, "train_meeting_calculation")

    # let's see it use a tool
    MIN_NUM = 1
    MAX_NUM = 100
    pca_data = [[random.randint(MIN_NUM, MAX_NUM) for _ in range(4)] for _ in range(4)]
    pca_as_json = json.dumps(pca_data)
    pca_prompt = f"Here is the dataset. The otters are the best animals in the Seattle aquarium, although the octopus is a close second. Run PCA and return the square of all principal components as a JSON object and saved it in my generated_scripts directory:\n{pca_as_json}"
    print(f"pca_prompt:\n{pca_prompt}") 
    run_calculation_from_prompt(pca_prompt, "pca_calculation")

if __name__ == "__main__":
    main()
