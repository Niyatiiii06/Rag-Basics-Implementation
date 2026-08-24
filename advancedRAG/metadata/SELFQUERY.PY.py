from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

from langchain_chroma import Chroma


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# 2. LOAD DOCUMENTS
# ==================================================

documents = TextLoader(
    "data/document.txt"
).load()


# ==================================================
# 3. ADD METADATA
# ==================================================

for doc in documents:

    doc.metadata["department"] = "HR"
    doc.metadata["year"] = 2026
    doc.metadata["document_type"] = "policy"


# ==================================================
# 4. SPLIT DOCUMENTS
# ==================================================

chunks = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
).split_documents(documents)


# ==================================================
# 5. CREATE EMBEDDINGS
# ==================================================

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)


# ==================================================
# 6. CREATE VECTOR STORE
# ==================================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


# ==================================================
# 7. CREATE LLM
# ==================================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)


# ==================================================
# 8. DEFINE STRUCTURED OUTPUT
# ==================================================

class SearchFilters(BaseModel):

    department: str | None = Field(
        default=None,
        description="Department such as HR, Finance, or Engineering"
    )

    year: int | None = Field(
        default=None,
        description="Year related to the document"
    )

    document_type: str | None = Field(
        default=None,
        description="Type of document such as policy or guide"
    )


class SearchQuery(BaseModel):

    query: str = Field(
        description="The semantic search query"
    )

    filters: SearchFilters = Field(
        description="Metadata filters extracted from the user question"
    )


# ==================================================
# 9. CREATE STRUCTURED LLM
# ==================================================

structured_llm = llm.with_structured_output(
    SearchQuery
)


# ==================================================
# 10. EXTRACT QUERY + FILTERS
# ==================================================

def extract_query_and_filters(
    question: str
) -> SearchQuery:

    prompt = f"""
You are a search query analyzer.

Extract:

1. A semantic search query.
2. Metadata filters.

Available metadata fields:

- department: HR, Finance, Engineering
- year: integer
- document_type: policy, guide

Only extract filters that are explicitly
or clearly implied by the user's question.

User question:

{question}
"""

    result = structured_llm.invoke(
        prompt
    )

    return result


# ==================================================
# 11. CLEAN EMPTY FILTERS
# ==================================================

def clean_filters(
    filters: SearchFilters
) -> dict:

    return filters.model_dump(
        exclude_none=True
    )


# ==================================================
# 12. SELF-QUERY RETRIEVAL
# ==================================================

def self_query_retrieval(
    question: str
):

    # LLM extracts query and filters
    result = extract_query_and_filters(
        question
    )

    # Semantic query
    search_query = result.query

    # Metadata filters
    filters = clean_filters(
        result.filters
    )

    # Retrieve documents
    docs = vectorstore.similarity_search(
        query=search_query,
        k=5,
        filter=filters if filters else None
    )

    return docs


# ==================================================
# 13. USER QUESTION
# ==================================================

question = input(
    "\nAsk a question: "
)


# ==================================================
# 14. RETRIEVE DOCUMENTS
# ==================================================

docs = self_query_retrieval(
    question
)


# ==================================================
# 15. BUILD CONTEXT
# ==================================================

context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# ==================================================
# 16. FINAL PROMPT
# ==================================================

prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY
the provided context.

If the answer is not available in the context,
say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""


# ==================================================
# 17. GENERATE ANSWER
# ==================================================

response = llm.invoke(
    prompt
)


# ==================================================
# 18. PRINT ANSWER
# ==================================================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)