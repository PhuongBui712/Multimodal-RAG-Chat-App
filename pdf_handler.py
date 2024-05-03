import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain.schema.document import Document


def create_cache_dir(directory=None):
    if not directory:
        directory = './.cache'

    os.makedirs('./.cache', exist_ok=True)
    return directory


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)

    return loader.load()


def load_pdf_directory(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()

    return loader.load()


def split_pdfs(pdfs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False
    )

    return splitter.split_documents(pdfs)


def extract_pdf(uploaded_pdf):
    cache_dir = create_cache_dir()
    cache_dir = os.path.join(cache_dir, 'temp_files')
    os.makedirs(cache_dir, exist_ok=True)

    files = []
    for file in uploaded_pdf:
        file_path = os.path.join(cache_dir, file.name)

        with open(file_path, 'wb') as w:
            w.write(file.getvalue())

        files.append(file_path)

    return files
