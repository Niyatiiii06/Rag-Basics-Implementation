from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document 
docs= [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]
embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore= Chroma.from_documents(docs, embeddings,persist_directory="chroma_db")
results= vectorstore.similarity_search("What is used for data analysis?", k=2)
for r in results:
    print(r.page_content, r.metadata)

retriever = vectorstore.as_retriever()

docs= retriever.invoke("What is used for data analysis?")
for d in docs:
    print(d.page_content)