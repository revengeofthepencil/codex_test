from dotenv import load_dotenv
import builtins
import contextlib
import io
import math
import os
from typing import Any

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from langgraph_codeact import create_codeact

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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
    new_vars = {key: _locals[key] for key in new_keys}
    return result, new_vars


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

def sin(a: float) -> float:
    """Take the sine of a number."""
    return math.sin(a)

def cos(a: float) -> float:
    """Take the cosine of a number."""
    return math.cos(a)

def radians(a: float) -> float:
    """Convert degrees to radians."""
    return math.radians(a)

def exponentiation(a: float, b: float) -> float:
    """Raise one number to the power of another."""
    return a**b

def sqrt(a: float) -> float:
    """Take the square root of a number."""
    return math.sqrt(a)

def ceil(a: float) -> float:
    """Round a number up to the nearest integer."""
    return math.ceil(a)

tools = [
    add,
    multiply,
    divide,
    subtract,
    sin,
    cos,
    radians,
    exponentiation,
    sqrt,
    ceil,
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


def main():
    load_dotenv()
    
    model = init_chat_model("gpt-4o", api_key=OPENAI_API_KEY)
    code_act = create_codeact(model, tools, eval)
    agent = code_act.compile(checkpointer=MemorySaver())
    
    messages = [{
        "role": "user",
        "content": "A train leaves Chicago for Los Angeles at 3:00PM traveling 50mph. The conductor has cold pizza for breakfast and everyone on the train agrees his breath smells horrible. Another train leaves Los Angeles for Chicago at 3:22PM travelling 40mpg. Asssuming they run on parallel tracks, what time will they meet if the conductor drinks a double espresso? The distance between Chicago and Los Angeles is 2,017 miles."
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



if __name__ == "__main__":
    main()
