from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage

MAX_ITERATIONS = 10
MODEL = "qwen3.5:4b"

@tool
def get_product_price(product_name: str) -> str:
    # Simulate a product price lookup
    """ This function is used to get the price of a product based on its name. It returns the price as a string. """

    print(f" >>>Executing get product_price with product_name: {product_name}")
    product_prices = {
    "iphone": "10.00",
    "charger" : "20.00",
    "headphones": "200"
    }

    return product_prices.get(product_name.lower(), "Product not found")

@tool
def apply_discount(discount_tier:str , product_price: str) -> str:
    """" This function applies a discount to the product price based on the discount tier. It returns the discounted price as a string.
    """
    print(f" >>>Executing apply discount with discount_tier: {discount_tier} and product_price: {product_price}")
    discount_tiers = {
        "silver": 0.05,  # 5% discount
        "gold": 0.10,    # 10% discount
        "platinum": 0.15 # 15% discount
    }

    product_price_value = float(product_price.strip('$'))
    print(f" >>>Parsed product price value: {product_price_value}")
    discount = discount_tiers.get(discount_tier.lower(), 0) 
    discounted_price = product_price_value - (product_price_value * discount)
    
    return f"${discounted_price:.2f}"




def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model=MODEL, model_provider="ollama",temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f" >>> Running agent with question: {question}")

    messages = [
        SystemMessage(content="your are helpful assistant that can answer questions about product prices and apply discounts based on the tiers and provide final price"),
        HumanMessage(content=question)
    ]

    for iteration in range(MAX_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                print(f" >>> Tool call detected: {tool_call}")
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_to_use = tools_dict.get(tool_name)
                if tool_to_use:
                    tool_response = tool_to_use.invoke(tool_args)
                    messages.append(ToolMessage(content=tool_response,tool_call_id=tool_call["id"]))
                else:
                    print(f" >>> Tool {tool_name} not found.")
        else:
            print(" >>> No tool call detected. Final response from AI:")
            print(ai_message.content)
            break
    
if __name__ == "__main__":
    question = "What is the price of an iphone and apply gold discount on it. Reply with final discounted price ?"
    run_agent(question)

