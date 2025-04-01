import os
import re
import tempfile
import uuid

import streamlit as st
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# 🔧 Clean filename for Chroma (removes "(1)" etc.)
def clean_filename(filename):
    return re.sub(r'\s\(\d+\)', '', filename)

# 📄 Extract text from uploaded PDF
def get_pdf_text(uploaded_file):
    temp_file = None
    try:
        input_file = uploaded_file.read()
        if not input_file:
            raise ValueError("Cannot read an empty file")

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        return documents

    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {str(e)}")
        return []

    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# ✂️ Split long documents into chunks for embeddings
def split_document(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if c.page_content and c.page_content.strip()]

# 🧠 Embedding function (Ollama)
def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

# 🗂️ Create vectorstore from chunks
def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    if not valid_chunks:
        raise ValueError(f"No valid content to process in {file_name}. Skipping.")

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in valid_chunks]
    unique_chunks = list({uuid: chunk for uuid, chunk in zip(ids, valid_chunks)}.values())

    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        collection_name=clean_filename(file_name),
        embedding=embedding_function,
        ids=ids,
        persist_directory=vector_store_path
    )
    vectorstore.persist()
    return vectorstore

# 🏗️ Higher-level function to go from text to vectorstore
def create_vectorstore_from_texts(documents, file_name):
    chunks = split_document(documents)
    embeddings = get_embedding_function()
    return create_vectorstore(chunks, embeddings, file_name)
