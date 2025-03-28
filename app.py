import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] =  os.getenv('OPENAI_API_KEY')

st.title("ChatGroq with Llama3 Demo")

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions: {input}
"""
)

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('./us_census')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)  

prompt1 = st.text_input("Enter your question from documents")

if st.button("Document Embeddings"):
    vector_embedding()
    st.write("Vector Store Db is ready")

import time



if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retreiver = st.session_state.vectors.as_retriever()
    retreival_chain = create_retrieval_chain(retreiver, document_chain)
    start = time.process_time()
    response = retreival_chain.invoke({"input":prompt1})
    print(f"Response time {time.process_time() - start}")
    st.write(response['answer'])

    # This tells us that where from the document our data is being retreived 
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks 
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------")


