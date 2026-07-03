import os
import wget
from Configuration import Config
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_data():
    """Load text data from a specified URL and return it as a list of documents.

    This function downloads a text file from the URL defined in the Config class,
    saves it locally, and then uses the TextLoader to read the contents of the file.
    The loaded documents are returned as a list.

    Returns:
        list: A list of documents loaded from the text file.
    """
    filename = 'companyPolicies.txt'
    url = getattr(Config, 'CORPUS_URL', None)

    if not url or not isinstance(url, str):
        raise ValueError("Config.CORPUS_URL must be a non-empty string pointing to your corpus URL.")

    try:
        wget.download(url, out=filename)
        print('Data downloaded')
    except Exception as exc:
        raise RuntimeError(f"Failed to download corpus from '{url}': {exc}") from exc

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Downloaded file not found: '{filename}'.")

    if os.path.getsize(filename) == 0:
        raise RuntimeError(f"Downloaded file '{filename}' is empty.")

    try:
        loader = TextLoader(filename, encoding='utf-8')
        documents = loader.load()
    except Exception as exc:
        raise RuntimeError(f"Failed to load text data from '{filename}': {exc}") from exc

    if not documents:
        raise RuntimeError(f"No documents were loaded from '{filename}'.")

    return documents


def split_data(documents, chunk_size=1000, chunk_overlap=0):
    """Split the loaded documents into smaller chunks.

    This function takes a list of documents and splits them into smaller chunks
    based on the specified chunk size and overlap. It uses the CharacterTextSplitter
    to perform the splitting.

    Args:
        documents (list): A list of documents to be split.
        chunk_size (int): The maximum size of each chunk. Default is 1000 characters.
        chunk_overlap (int): The number of characters to overlap between chunks. Default is 0.

    Returns:
        list: A list of document chunks.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    document_chunks = text_splitter.split_documents(documents)
    return document_chunks


def embed_data(document_chunks):
    """Embed the document chunks using a specified embedding model.

    This function takes a list of document chunks and generates embeddings for each chunk
    using the specified embedding model. The embeddings are returned as a list.

    Args:
        document_chunks (list): A list of document chunks to be embedded.

    Returns:
        list: A list of embedded document chunks.
    """
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_documents(document_chunks, embeddings)  # store the embedding in docsearch using Chromadb
    return docsearch