from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from constants._constants import ConstantsLLM
from tools_with_chat.tools import TOOLS


def printw(message: str) -> None:
    """
    Print a message with a border.
    """
    print(f"--------------\n{message}\n")


def main() -> None:
    """
    Main function to demonstrate the usage of the add and matching functions.
    """
    # Define the LLM model to use
    model = ConstantsLLM.ADVANCED
    llm = ChatOllama(model=model)
    printw(f"Using LLM model: {model}")

    # Bind tools
    llm_using_tool = llm.bind_tools([TOOLS["add"], TOOLS["matching"]])

    # Query -> create tool parsers
    query = "Can you give me the match for 'hello'?"
    messages = [HumanMessage(content=query)]
    ai_msg = llm_using_tool.invoke(messages)
    printw(f"AI message: {ai_msg}")
    messages.append(ai_msg)

    # Call the tools
    for tool_call in ai_msg.tool_calls:
        selected_tool = TOOLS[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    printw(f"Tool message: {tool_msg}")

    # Final AI message
    final_ai_msg = llm.invoke(messages)
    printw(f"Final AI message: {final_ai_msg.content}")


if __name__ == "__main__":
    main()
