# main_chatbot.py
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# MODEL = "deepseek-r1:8b"
MODEL = "gemma3:1b"  # 778MB, lightweight ?


def main():
    print("Running OllamaLLM tests...")
    # test_discussion()
    # test_discussion_with_memory()
    test_message_persistence()
    print()
    print("Test completed successfully.")


def test_message_persistence():
    # see https://python.langchain.com/docs/tutorials/chatbot/#message-persistence
    model = OllamaLLM(model=MODEL)
    workflow = StateGraph(state_schema=MessagesState)
    config = {"configurable": {"thread_id": "abcd123"}}

    def call_model(state: MessagesState) -> MessagesState:
        """Call the LLM with the current messages."""
        response = model.invoke(state["messages"])
        return {"messages": response}

    # Define the single node in the graph
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # Example: 1) Give my name
    query = "Hi ! I'm Patrick"
    print(f"Query: {query}")
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config=config)
    output["messages"][-1].pretty_print()

    # 2) Ask what is my name
    query = "What's my name ?"
    print(f"Query: {query}")
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config=config)
    output["messages"][-1].pretty_print()

    # 3) same while changing thread
    config["configurable"]["thread_id"] = "efgh456"
    query = "What's my name ?"
    print(f"Query: {query}")
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config=config)
    output["messages"][-1].pretty_print()


def test_llm():
    """Test the OllamaLLM with a simple query."""
    # Initialize the OllamaLLM with the specified model
    llm = OllamaLLM(model=MODEL)
    llm.invoke("What is the capital of France?")
    # will not print anything


def test_llm_with_stream():
    llm = OllamaLLM(model=MODEL, streaming=True)
    response = llm.invoke("What is the capital of France?")
    for chunk in response:
        print(chunk, end="", flush=True)


def test_discussion():
    """Test the OllamaLLM with a discussion. No memory, just a simple chat."""
    llm = OllamaLLM(model=MODEL)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = llm.invoke(user_input)
        print(f"LLM: {response}")


def test_discussion_with_memory():
    """Test the OllamaLLM with a discussion and memory."""
    pass


if __name__ == "__main__":
    main()
