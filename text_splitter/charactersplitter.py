from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter= CharacterTextSplitter(
separator=" ",
chunk_size=10,
chunk_overlap=1)
data= TextLoader("doc_loaders/notes.txt").load()

chunks= splitter.split_documents(data)
for i in chunks:
    print(i.page_content)