"""
Simple script to test the RAG (Retrieval-Augmented Generation) functionality.
You can provide a question to the RAG system, and it will retrieve relevant context
from a web page and generate an answer using a language model.
"""

from pathlib import Path
from typing_extensions import List, TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langgraph.graph import StateGraph, START

from constants._constants import ConstantsEmbedding, ConstantsLLM
from document_loaders import DocumentLoader, split_text


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class VectorStoreInitializer:
    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model

    def initialize(self) -> VectorStore:
        embedding = OllamaEmbeddings(model=self.embedding_model)
        vector_store = InMemoryVectorStore(embedding=embedding)
        return vector_store


# class LLMGraph:
#     def __init__(self, llm_model, vector_store: VectorStore, prompt):
#         self.llm_model = llm_model
#         self.vector_store = vector_store
#         self.prompt = prompt

#     def retrieve(self, state: State) -> dict:
#         retrieved_docs = self.vector_store.similarity_search(state["question"])
#         return {"context": retrieved_docs}

#     def generate(self, state: State) -> dict:
#         # NB: prompt comes from 'hu', so need to check what is its type
#         docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#         messages = self.prompt.invoke(
#             {"question": state["question"], "context": docs_content}
#         )
#         response = self.llm_model.invoke(messages)
#         return {"answer": response}

#     def compile(self):
#         """Compile the graph with the defined nodes and edges."""
#         graph_builder = StateGraph(State)
#         graph_builder.add_node("retrieve", self.retrieve)
#         graph_builder.add_node("generate", self.generate)
#         graph_builder.add_edge(START, "retrieve")
#         graph_builder.add_edge("retrieve", "generate")
#         return graph_builder.compile()


class AskingWithRAG:
    def __init__(
        self,
        question: str,
        vector_store: VectorStore,
        llm_model: OllamaLLM,
        language: str = "English",
        response_size: str = "medium",
        streaming: bool = False,
    ):
        self.question = question
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.language = language
        self.response_size = response_size
        self.streaming = streaming

    def customize_prompt_based_on_response_size(self):
        """Customize the prompt based on the response size."""
        if self.response_size == "short":
            return "Please keep your answer concise."
        elif self.response_size == "medium":
            return "Provide a detailed answer."
        elif self.response_size == "long":
            return "Elaborate extensively on the topic."
        else:
            raise ValueError("Invalid response size specified.")

    def run(self) -> str:
        """Run the RAG system to get an answer to the question."""
        context = self.vector_store.similarity_search(self.question, k=10)
        prompt = (
            f"You are a helpful assistant. You have to answer in {self.language} and not another language. "
            f"{self.customize_prompt_based_on_response_size()}\n"
            f"Context: {context}\n"
            f"Question: {self.question}\n"
            f"Answer:"
        )
        if self.streaming:
            response = ""
            for chunk in self.llm_model.invoke(prompt):
                print(chunk, end="", flush=True)
                response += chunk
            print()
            return response
        else:
            response = self.llm_model.invoke(prompt)
            return response


# class AskingWithRAGGraph:
#     def __init__(self, question: str, llm_graph: LLMGraph):
#         self.question = question
#         self.llm_graph = llm_graph

#     def run(self):
#         """Run the RAG system to get an answer to the question."""
#         initial_state = {"question": self.question, "context": [], "answer": ""}
#         response = self.llm_graph.compile().invoke(initial_state)
#         return response["answer"]


def main(
    question: str,
    language: str = "English",
    response_size: str = "medium",
    streaming: bool = False,
):
    # Load documents from a web page
    loader = DocumentLoader()
    # docs = loader.from_url("https://lilianweng.github.io/posts/2023-06-23-agent/")
    DATA_PATH = Path(__file__).parent / "data"
    docs = loader.from_pdf(
        DATA_PATH / "promesse_compromis.pdf"  # Adjust the path as needed
    )
    OUTPUT_DIR = Path(__file__).parent / "output"
    print(f"Loaded {len(docs)} documents from the web page.")
    # Split the documents into smaller chunks
    splitted_doc = split_text("\n\n".join(doc.page_content for doc in docs))
    print(f"Split into {len(splitted_doc)} text chunks.")

    # Initialize vector store with embeddings
    vector_store = VectorStoreInitializer(ConstantsEmbedding.NOMIC).initialize()
    vector_store.add_documents(splitted_doc)
    print(f"Added {len(splitted_doc)} documents to the vector store.")

    # Ask a question using RAG
    print(f"Asking question: {question}")
    model = ConstantsLLM.ADVANCED
    # model = ConstantsLLM.LIGHT
    llm = OllamaLLM(model=model, streaming=streaming)
    print(f"Using LLM model: {model}")
    rag_system = AskingWithRAG(
        question, vector_store, llm, language, response_size, streaming
    )
    print("Running RAG system...")
    response = rag_system.run()
    print("RAG system completed.")

    print("Response:\n", response)

    # Write the response to a Markdown file
    with open("response.md", "w") as f:
        f.write(response)


if __name__ == "__main__":
    QUESTION = "Trouve-moi tout les articles du code civil concernant les promesses de vente, ordonné par importance"  # Adjust the question as needed
    LANGUAGE = "French"  # or "English", etc.
    RESPONSE_SIZE = "medium"  # or "medium", "long"
    STREAMING = True  # Set to True to enable streaming
    main(QUESTION, LANGUAGE, RESPONSE_SIZE, STREAMING)
