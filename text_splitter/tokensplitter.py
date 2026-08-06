from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data= PyPDFLoader("doc_loaders/GRU.pdf").load()
splitter= TokenTextSplitter(
chunk_size=100,
chunk_overlap=10)

chunks= splitter.split_documents(data)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[50].page_content}")
#for i in chunks:
   # print(i.page_content)