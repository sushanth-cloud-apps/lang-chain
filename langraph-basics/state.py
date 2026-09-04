from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Annotated


class State(TypedDict):
    """A state object that can be used to store information about the current state of the agent."""
    messages: Annotated[List,add_messages]


