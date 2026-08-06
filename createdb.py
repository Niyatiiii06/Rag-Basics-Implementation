#load pdf
#split into chunks
#create embeddings
#store into vdb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()   

doc= PyPDFLoader("doc_loaders/deeplearning.pdf").load()

splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

chunks= splitter.split_documents(doc)

embeddings= MistralAIEmbeddings(model="mistral-embed")

vectorstore= Chroma.from_documents(chunks, embeddings,persist_directory="chroma_db")
