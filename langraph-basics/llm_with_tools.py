from dotenv import load_dotenv
import random
 # Load environment variables from .env file
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv() 

@tool
def get_temperature() -> float:
    """Get the temperature value for any specified city."""
    print("executing get_temperature tool")
    return float(random.uniform(20.0, 50.0))  # Default to a random value between 20.0 and 50.0 if not set

@tool
def triple_temperature(temp: float) -> float:
    """Triple the temperature value."""
    print(" triple_temperature tool")
    return temp * 3

tools = [get_temperature, triple_temperature]

llm = init_chat_model(
    model="qwen3.5:4b",
    model_provider="ollama",
    model_kwargs={"temperature": 0},
).bind_tools(tools).with_config({"run_name": "llm_call"})




