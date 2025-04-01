# Clean filename function
def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    
    return new_filename

#Get PDF text 
def get_pdf_text(uploaded_file): 

    #create a temporary file to avoid empty list error
    temp_file = None

    try:
        # Read file content
        input_file = uploaded_file.read()

        # Check if file is empty
        if not input_file:
            raise ValueError("Cannot read an empty file")

        # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
        # it can't work directly with file-like objects or byte streams that we get from Streamlit's uploaded_file)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        # Load PDF document
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        return documents
    
    except Exception as e:
        # Catch any error, print it for debugging purposes
        st.error(f"An error occurred while processing the PDF: {str(e)}")
        return []

    finally:
        # Ensure the temporary file is deleted when we're done with it
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

#split the text from the PDF document into chuncks
def split_document(documents, chunk_size, chunk_overlap):    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap,
                                                  length_function=len,
                                                  separators=["\n\n", "\n", " "])
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    # Ensure we only keep valid chunks (non-empty and non-None)
    chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
    
    return chunks

#Get embeddings for the vectorstore
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

#Vectorstore to hold the text chunks and embeddings
def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    """
    Create a vector store from a list of text chunks.
    """
    # Filter out any chunks with None or empty content
    valid_chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
    
    # Ensure that we only process valid chunks
    if not valid_chunks:
        raise ValueError(f"No valid content to process in {file_name}. Skipping.")
    
    # Process valid chunks (this is where the embedding happens)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in valid_chunks]
    unique_chunks = list({uuid: chunk for uuid, chunk in zip(ids, valid_chunks)}.values())

    # Create the vector store
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),
                                        embedding=embedding_function, 
                                        ids=ids, 
                                        persist_directory=vector_store_path)
    
    vectorstore.persist()
    
    return vectorstore

#Further refine the vecotorstore
def create_vectorstore_from_texts(documents, file_name):
    """
    Create a vector store from a list of texts.
    """

    # Split the documents into chunks
    chunks = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3 define embedding function
    embedding_function = get_embedding_function()

    # Step 4 create a vector store  
    vectorstore = create_vectorstore(chunks, embedding_function, file_name)
    
    return vectorstore
