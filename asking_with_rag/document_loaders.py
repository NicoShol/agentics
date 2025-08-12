import bs4
import PyPDF2
from langchain_core.documents.base import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """Class to load documents from a web page."""

    def __init__(self):
        pass

    def from_url(self, url: str) -> list[Document]:
        """Load documents from a given URL."""
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        return docs

    def from_pdf(self, pdf_path: str) -> list[Document]:
        """Load documents from a PDF file."""
        # Implement PDF loading logic here
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            documents = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text))
            print(f"Loaded {len(documents)} documents from {pdf_path}.")
            return documents


def split_text(text: str) -> list[Document]:
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    return [Document(page_content=split) for split in splits]
