
from llm_with_tools import llm,tools
from langgraph.prebuilt import ToolNode
from state import State

def llm_node(state : State):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

