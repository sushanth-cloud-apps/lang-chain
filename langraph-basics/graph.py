from langgraph.graph import END, StateGraph , START
from state import State
from nodes import llm_node, tool_node
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

AGENT_NODE = "llm_node"
TOOL_NODE = "tool_node"
ACT = "act"
LAST = -1



def should_continue(state: State) -> bool:
    """Determine whether the agent should continue or not based on the last message."""
    last_message = state["messages"][LAST]
    if not last_message.tool_calls:
        return END
    return ACT

def create_graph():
    workflow = StateGraph(State)
    workflow.add_node(AGENT_NODE, llm_node)
    workflow.set_entry_point(AGENT_NODE)
    workflow.add_node(ACT, tool_node)
    workflow.add_conditional_edges(AGENT_NODE, should_continue, {ACT: ACT , END:END})
    workflow.add_edge(ACT, AGENT_NODE)
    return workflow

    




if __name__ == "__main__":
    print("creating and running the graph")
    graph = create_graph().compile()

    initial_state = {
        "messages": [HumanMessage(content="What is the temperature in New York City? Get the temperature and triple it.")
        ]
    }

    langfuse_handler = CallbackHandler()
    result =graph.invoke(
    initial_state,
    config={
        "callbacks": [langfuse_handler],
        "run_name": "temperature-agent",
        "metadata": {
            "environment": "development",
            "agent": "temperature-agent",
            "version": "1.0"
        },
        "tags": [
            "langgraph",
            "temperature-agent"
            ]
        }
    )
    print("Final state:", result["messages"][LAST].content)

