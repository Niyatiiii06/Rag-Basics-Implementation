from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.apple.com/ae/iphone-17-pro/")
data = loader.load()
print(data[0].page_content)