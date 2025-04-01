from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama

import uuid

# 🧠 Prompt Template for RAG
PROMPT_TEMPLATE = """
You are an assistant analyzing annual reports. Please do not lie and only answer based on the PDF information.

Context:
{context}

Question:
{question}

Answer:
"""

# 🔍 Optional: Contextualize the user's query (you can modify this later)
def contextualize_query(query):
    return f"Provide a summary of this document: {query}"

# 🧠 Embedding function
def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

# 📝 Format context documents for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 💬 Retrieval pipeline
def query_document(vectorstore, query="Extract relevant details from the uploaded file."):
    llm = ChatOllama(model='llama3.2', temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    if not query or not isinstance(query, str):
        raise ValueError(f"Invalid query: {query}")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )

    return rag_chain.invoke(query)
