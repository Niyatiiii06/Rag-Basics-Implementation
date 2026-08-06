from langchain_community.document_loaders import PyPDFLoader
docs= PyPDFLoader("doc_loaders/GRU.pdf").load()
print(len(docs))
