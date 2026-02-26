from dotenv import load_dotenv
import builtins
import contextlib
import io
import math
import os
import re
from datetime import datetime
from typing import Any

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

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


tools = [
    add,
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


def run_calculation_from_prompt(prompt, output_script_name = "calculation_script"):
    model = init_chat_model("gpt-4o", api_key=OPENAI_API_KEY)
    code_act = create_codeact(model, tools, eval)
    agent = code_act.compile(checkpointer=MemorySaver())
    
    messages = [{
        "role": "user",
        "content": prompt
    }]

    print("\n--- Starting Agent ---\n")
    
    final_values = {}
    for typ, chunk in agent.stream(
        {"messages": messages},
        stream_mode=["values", "messages"],
        config={"configurable": {"thread_id": 1}},
    ):
        if typ == "messages":
            msg = chunk[0]
            if msg.content:
                print(msg.content, end="", flush=True)
        elif typ == "values":
            final_values = chunk

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
    script_name1 = "train_meeting_calculation"
    prompt1 = "A train leaves Chicago for Los Angeles at 3:00PM traveling 50mph. The conductor has cold pizza for breakfast and everyone on the train agrees his breath smells horrible. Another train leaves Los Angeles for Chicago at 3:22PM travelling 40mpg. Asssuming they run on parallel tracks, what time will they meet if the conductor drinks a double espresso? The distance between Chicago and Los Angeles is 2,017 miles. Time Bandits is my favorite movie."
    run_calculation_from_prompt(prompt1, script_name1)

if __name__ == "__main__":
    main()
