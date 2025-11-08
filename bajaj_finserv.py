from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFMinerLoader, UnstructuredPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.output_parsers import StrOutputParser 
import streamlit as st 
import pandas as pd 
import os 
from typing import Literal
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

st.title("RAG Chatbot")
st.subheader("Smart chatbot that helps to Q&A on pdf")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message('user'):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('assistant'):
            st.markdown(message.content)

# path = "bajaj_finserv_factsheet_Oct.pdf"

# Use PDFMinerLoader for reliable text and layout extraction
loader = PDFMinerLoader("bajaj_finserv_factsheet_Oct.pdf")

# For additional structured extraction (tables, elements)
unstructured_loader = UnstructuredPDFLoader(
    "bajaj_finserv_factsheet_Oct.pdf",
    mode="elements",
    strategy="fast"  # or "accurate" for better extraction
)

# Load documents from both loaders
docs_pdfminer = loader.load()
docs_unstructured = unstructured_loader.load()
docs = docs_pdfminer + docs_unstructured  # Combine documents

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
chunks = splitter.split_documents(docs)
embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L12-v2')

vs = FAISS.from_documents(chunks, embeddings)

retriver = vs.as_retriever(search_type='similarity', search_kwargs={'k':10})

prompt = ChatPromptTemplate([
    ('system',"""Give answer using only provided documents. Strictly follow the given instruction: 
    1. Read the provided Fund Factsheet PDF (which includes text, tables, charts, and images).
    2. Extract and store all useful information (text, numbers, data from tables or charts).
    3. Answer user questions accurately using only the data from the document.
    4. Perform simple to complex calculations, such as:
        •	Calculate CAGR or average returns
        •	Compare fund performance over different time periods
        •	Analyse asset allocation (e.g., debt vs. equity)
        •	Explain risk metrics like the Sharpe ratio
    5. if user asked questions like: List top 5 holdings of the Consumption Fund with weights or Which of the listed equity funds has the highest 3-year return? = then gives a data in tabular format.
    """),
    ('human',"question {question}\n\n context {context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    'context':retriver | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
    
})
llm = ChatGroq(api_key="", temperature=0) #type:ignore

chain = parallel | prompt | llm | StrOutputParser()

user_input = st.chat_input("Asked a Question: ")
if user_input is not None:
    with st.chat_message('user'):
        st.markdown(user_input)
        
        st.session_state.messages.append(HumanMessage(user_input))
    
    response = chain.invoke(user_input)
    with st.chat_message('assistant'):
        st.markdown(response)
        # Try to visualize any data in the response
        # visualize_data(response)
        
        st.session_state.messages.append(AIMessage(response))


