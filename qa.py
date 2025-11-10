import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA



os.environ["GROQ_API_KEY"] = "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"  

loader = PyPDFLoader("text-2.pdf")
pages = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings()


vectorstore = Chroma.from_documents(docs, embedding=embeddings)


retriever = vectorstore.as_retriever()


llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],  # Ensure correct API key
    model="llama-3.1-8b-instant",  # Check if model is available in Groq
    temperature=0.7  # Adjust as needed
)

# Create RetrievalQA Chain for RAG (Retrieval Augmented Generation)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


st.title("Q&A using RAG")

user_query = st.text_input("Ask a question about the content in the PDF:")

if user_query:
    try:
        answer = rag_chain.invoke(user_query)
        st.write(answer['result'])  # Display the result
       
        if 'source_documents' in answer:
            del answer['source_documents']  # Removes the source documents field
    except Exception as e:
        st.error(f"Error while processing the query: {e}")



