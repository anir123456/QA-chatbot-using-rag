import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


os.environ["GROQ_API_KEY"] = "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"


loader = PyPDFLoader("text-3.pdf")
pages = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(pages)


embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding=embeddings)


retriever = vectorstore.as_retriever()


llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model= "Llama3-8b-8192",
    temperature=0.7
)



# 🔄 RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
st.title("QandA using RAG")

user_query = st.text_input("Ask a question in the file")

if user_query:
    answer = rag_chain.invoke(user_query)
    st.write(answer["result"])

    