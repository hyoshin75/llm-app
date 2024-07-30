from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
#추가
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, JSONLoader, UnstructuredMarkdownLoader, PyMuPDFLoader)
from langchain_openai import OpenAIEmbeddings
import chromadb, faiss
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

st.set_page_config(page_title="챗봇", page_icon="⭐", layout='wide')
st.header('인공지능 보조교사')
st.markdown("#### **Knowledgebase 생성기**")

embedding_model = OpenAIEmbeddings()
# FAISS vector store 관련
faiss_dir = './faiss'
if(os.path.isdir(faiss_dir) == False):
    vs = FAISS(
            embedding_function=embedding_model, index=faiss.IndexFlatL2(1536),
            docstore=InMemoryDocstore(), index_to_docstore_id={})
else :
    vs = FAISS.load_local(faiss_dir, embedding_model,
            allow_dangerous_deserialization=True)

def vs_add_file(file_path):
    if file_path.endswith('.txt'):
        text_loader = TextLoader(file_path)
        raw_doc = text_loader.load()
    elif file_path.endswith('.md'):
        markdown_loader = UnstructuredMarkdownLoader(file_path)
        raw_doc = markdown_loader.load()
    elif file_path.endswith('.pdf'):
        pdf_loader = PyMuPDFLoader(file_path)
        raw_doc = pdf_loader.load()
    elif file_path.endswith('.json'):
        json_loader = JSONLoader(file_path)
        raw_doc = json_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, chunk_overlap  = 100)
    docs = text_splitter.split_documents(raw_doc)

    if(len(docs)>0):
        vs.add_documents(docs)

    vs.save_local(faiss_dir)

def save_file(file):
    import os
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path

uploaded_files = st.file_uploader("Choose a data file", accept_multiple_files=True)
for file in uploaded_files:
    file_path = save_file(file)
    vs_add_file(file_path)

if(len(uploaded_files)):
    st.markdown("#### 업로드 완료 되었습니다.")