from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

MODEL = "qwen3.5:4b"


@tool
def get_product_price(product_name: str) -> str:
    """Return the price of a product."""
    print(f" >>>Executing get product_price with product_name: {product_name}")
    product_prices = {
        "iphone": "10.00",
        "charger": "20.00",
        "headphones": "200",
    }
    return product_prices.get(product_name.lower(), "Product not found")


@tool
def apply_discount(discount_tier: str, product_price: str) -> str:
    """Apply a percentage discount to a product price."""
    print(
        f" >>>Executing apply discount with discount_tier: {discount_tier} and product_price: {product_price}"
    )
    discount_tiers = {
        "silver": 0.05,
        "gold": 0.10,
        "platinum": 0.15,
    }

    product_price_value = float(product_price.strip("$"))
    print(f" >>>Parsed product price value: {product_price_value}")
    discount = discount_tiers.get(discount_tier.lower(), 0)
    discounted_price = product_price_value - (product_price_value * discount)
    return f"${discounted_price:.2f}"


def build_messages(question: str):
    return [
        SystemMessage(
            content=(
                "You are a helpful assistant that can answer product price questions and "
                "apply discounts based on tier. Return the final discounted price in your final answer."
            )
        ),
        HumanMessage(content=question),
    ]


def execute_tool_calls(ai_message, tools_dict):
    if not ai_message.tool_calls:
        return []

    tool_messages = []
    for tool_call in ai_message.tool_calls:
        print(f" >>> Tool call detected: {tool_call}")
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            print(f" >>> Tool {tool_name} not found.")
            continue

        tool_response = tool_to_use.invoke(tool_args)
        tool_messages.append(
            ToolMessage(content=str(tool_response), tool_call_id=tool_call["id"])
        )

    return tool_messages


def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model=MODEL, model_provider="ollama", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer product price questions and apply discounts based on tier."),
        ("human", "{question}"),
    ])

    chain = prompt | llm_with_tools
    print(f" >>> Running agent with question: {question}")

    first_response = chain.invoke({"question": question})

    if not first_response.tool_calls:
        print(" >>> No tool call detected. Final response from AI:")
        print(first_response.content)
        return first_response.content

    messages = build_messages(question) + [first_response] + execute_tool_calls(first_response, tools_dict)
    final_response = llm.invoke(messages)

    print(" >>> Final response from AI:")
    print(final_response.content)
    return final_response.content


if __name__ == "__main__":
    question = "What is the price of an iphone and apply gold discount on it. Reply with final discounted price ?"
    run_agent(question)

