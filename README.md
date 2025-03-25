# ChatGroq with Llama3 and FAISS

This repository contains a Streamlit-based application that integrates Groq's Llama3-8b-8192 model with FAISS for document retrieval and question answering. The application allows users to upload PDFs, generate vector embeddings, and query documents using natural language.

## Features
- **PDF Document Processing**: Loads and splits PDFs into manageable text chunks.
- **Vector Embeddings with FAISS**: Converts document chunks into vector embeddings for efficient retrieval.
- **ChatGroq with Llama3 Integration**: Uses the Llama3-8b-8192 model to generate accurate responses.
- **Real-Time Q&A**: Users can query uploaded documents for instant responses.
- **Document Similarity Search**: Displays retrieved document chunks for transparency.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/itzuditsharma/Document-Chatbot-Groq.git
   cd Document-Chatbot-Groq
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Fill in your OpenAI API key and Groq API key in the `.env` file.

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Dependencies
- `streamlit`
- `langchain`
- `langchain_groq`
- `langchain_openai`
- `faiss`
- `python-dotenv`
- `PyPDF2`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Example Code
```python
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
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

st.title("ChatGroq with Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

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
```

## Contributing
Feel free to fork this repository and submit pull requests with improvements.

## License
This project is licensed under the MIT License.

# Results
![image](https://github.com/user-attachments/assets/766feb77-7faa-4b3c-afb0-7d03c87f787b)
![image](https://github.com/user-attachments/assets/6c90f586-431e-41b5-8455-4029178e73fa)

![image](https://github.com/user-attachments/assets/aec80cb2-2114-43fe-a22f-2943fbdec2dd)
![image](https://github.com/user-attachments/assets/6f493677-bd3e-4097-ae11-dd5358d04cd5)






