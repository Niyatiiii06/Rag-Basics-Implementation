from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)
from langchain_chroma import Chroma


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# 2. LOAD DOCUMENTS
# ==========================================

documents = TextLoader(
    "data/document.txt"
).load()


# ==========================================
# 3. ADD METADATA
# ==========================================

for doc in documents:

    doc.metadata["department"] = "Finance"
    doc.metadata["access_level"] = "employee"


# ==========================================
# 4. SPLIT DOCUMENTS
# ==========================================

chunks = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
).split_documents(documents)


# ==========================================
# 5. CREATE EMBEDDINGS
# ==========================================

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)


# ==========================================
# 6. CREATE VECTOR STORE
# ==========================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


# ==========================================
# 7. CREATE LLM
# ==========================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)


# ==========================================
# 8. DEFINE ACCESS RULES
# ==========================================

users = {
    "intern": {
        "allowed_access": [
            "public"
        ]
    },

    "employee": {
        "allowed_access": [
            "public",
            "employee"
        ]
    },

    "manager": {
        "allowed_access": [
            "public",
            "employee",
            "confidential"
        ]
    }
}


# ==========================================
# 9. GET USER ACCESS LEVELS
# ==========================================

def get_access_levels(
    user_role: str
) -> list[str]:

    return users[user_role][
        "allowed_access"
    ]


# ==========================================
# 10. SECURE RETRIEVAL
# ==========================================

def retrieve_with_access_control(
    question: str,
    user_role: str
):

    allowed_levels = get_access_levels(
        user_role
    )

    filters = {
        "access_level": {
            "$in": allowed_levels
        }
    }

    docs = vectorstore.similarity_search(
        query=question,
        k=5,
        filter=filters
    )

    return docs


# ==========================================
# 11. USER INPUT
# ==========================================

user_role = input(
    "Enter your role: "
).lower()

question = input(
    "Ask a question: "
)


# ==========================================
# 12. RETRIEVE SECURELY
# ==========================================

docs = retrieve_with_access_control(
    question,
    user_role
)


# ==========================================
# 13. BUILD CONTEXT
# ==========================================

context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# ==========================================
# 14. FINAL PROMPT
# ==========================================

prompt = f"""
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


# ==========================================
# 15. GENERATE ANSWER
# ==========================================

response = llm.invoke(prompt)

print("\nANSWER:\n")

print(response.content)