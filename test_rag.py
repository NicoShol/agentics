# https://python.langchain.com/docs/tutorials/rag/

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langgraph.graph import StateGraph, START
from typing_extensions import List, TypedDict

MODEL = "gemma3:1b"  # 778MB, lightweight ?
LLM_MODEL = "gpt-oss:20b"  # 778MB, lightweight ?
EMBEDDING_MODEL = "nomic-embed-text"  # For embeddings, not LLM


# QUESTION = "What is Task Decomposition?"  # OK Example question about the content
QUESTION = "Can you summarize LLM Powered Autonomous Agents?"  # OK Example question to summarize the content itself (generic)
# QUESTION = "Can you summarize all your context ?"  # UNABLE: Example question to summarize the content itself (generic)


def main():
    print("Running RAG tests...")
    test_rag()
    print("RAG test completed successfully.")


# custom type used in the state graph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def test_rag():
    # Loading the web page
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # Splitting the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print(f"Split into {len(all_splits)} text chunks.")

    # Indexing chunks
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = InMemoryVectorStore(embedding=embedding)
    vector_store.add_documents(documents=all_splits)

    # Getting prompt
    prompt = hub.pull("rlm/rag-prompt")
    breakpoint()

    # Initializing the LLM
    llm = OllamaLLM(model=LLM_MODEL)

    def retrieve(state: State) -> dict:
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State) -> dict:
        # NB: prompt comes from 'hu', so need to check what is its type
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = llm.invoke(messages)
        return {"answer": response}

    # Graph shenanigans TODO: understand this better
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile()

    response = graph.invoke({"question": QUESTION})
    print("Response:", response["answer"])


if __name__ == "__main__":
    main()
