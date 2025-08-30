import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# Secure the API Key (load it from environment or secret manager)
os.environ["GROQ_API_KEY"] = "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"  # Change to secure key loading

# PDF Loading and Splitting
loader = PyPDFLoader("text-2.pdf")
pages = loader.load()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(pages)

# Initialize Embeddings (Ensure this is using a supported model)
embeddings = HuggingFaceEmbeddings()

# Initialize the Vector Store (Chroma)
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# Set up retriever for the vector store
retriever = vectorstore.as_retriever()

# Initialize Groq LLM (ensure the API key and model are valid)
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],  # Ensure correct API key
    model="Llama3-8b-8192",  # Check if model is available in Groq
    temperature=0.7  # Adjust as needed
)

# Create RetrievalQA Chain for RAG (Retrieval Augmented Generation)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Streamlit UI
st.title("Q&A using RAG")

# Input field for the user's query
user_query = st.text_input("Ask a question about the content in the PDF:")

if user_query:
    try:
        answer = rag_chain.invoke(user_query)
        st.write(answer['result'])  # Display the result
        # Remove source_documents from being displayed
        # If you want to exclude source documents entirely:
        # Just skip the part where you display them
        if 'source_documents' in answer:
            del answer['source_documents']  # Removes the source documents field
    except Exception as e:
        st.error(f"Error while processing the query: {e}")
