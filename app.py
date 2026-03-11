import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

st.title("AI PDF Chatbot (Free & Local)")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

all_docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp_" + uploaded_file.name)

        # Save uploaded file temporarily
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Check if PDF has text
        text_content = "".join([doc.page_content for doc in docs]).strip()
        if not text_content:
            st.warning(f"{uploaded_file.name} seems to be a scanned/image PDF. Text cannot be extracted.")
            continue

        all_docs.extend(docs)

    if not all_docs:
        st.error("No readable text found in uploaded PDFs.")
        st.stop()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = splitter.split_documents(all_docs)
    st.success(f"Created {len(documents)} text chunks from PDFs")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Initialize LLM
    llm = Ollama(model="llama3")

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # Chat interface
    query = st.chat_input("Ask a question about the PDFs")
    if query:
        with st.spinner("Thinking..."):
            result = qa.invoke({"query": query})

        st.chat_message("user").write(query)
        st.chat_message("assistant").write(result["result"])