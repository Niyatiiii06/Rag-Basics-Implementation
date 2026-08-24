import uuid

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

from langchain_chroma import Chroma


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# 2. CREATE / LOAD DOCUMENT
# =========================================================

document = Document(
    page_content="""
Employees receive 20 days of annual leave.
Unused leave can be carried forward for up to 5 days.
Employees must submit leave requests at least 7 days
in advance.

Sick leave is available for employees who are medically
unwell. A medical certificate may be required for
extended sick leave.

Employees can also request work from home when they
have a valid reason. Work from home requests must be
approved by the manager.
""",
    metadata={
        "source": "employee_policy.txt"
    }
)


# =========================================================
# 3. CREATE PARENT CHUNKS
# =========================================================

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

parent_chunks = parent_splitter.split_documents(
    [document]
)


# =========================================================
# 4. GIVE EACH PARENT A UNIQUE ID
# =========================================================

for parent in parent_chunks:

    parent_id = str(uuid.uuid4())

    parent.metadata["parent_id"] = parent_id


# =========================================================
# 5. CREATE CHILD CHUNKS
# =========================================================

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

child_chunks = []

for parent in parent_chunks:

    children = child_splitter.split_documents(
        [parent]
    )

    child_chunks.extend(children)


# =========================================================
# 6. CREATE PARENT STORE
# =========================================================

parent_store = {}

for parent in parent_chunks:

    parent_id = parent.metadata["parent_id"]

    parent_store[parent_id] = parent


# =========================================================
# 7. CREATE EMBEDDINGS
# =========================================================

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)


# =========================================================
# 8. STORE CHILD CHUNKS IN CHROMA
# =========================================================

vectorstore = Chroma.from_documents(
    documents=child_chunks,
    embedding=embeddings,
    persist_directory="child_chroma_db"
)


# =========================================================
# 9. CREATE LLM
# =========================================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)


# =========================================================
# 10. PARENT-CHILD RETRIEVAL
# =========================================================

def parent_child_retrieval(
    question: str,
    k: int = 5
):

    # -----------------------------------------------
    # Search SMALL child chunks
    # -----------------------------------------------

    children = vectorstore.similarity_search(
        query=question,
        k=k
    )


    # -----------------------------------------------
    # Store unique parent documents
    # -----------------------------------------------

    parents = []

    seen_parent_ids = set()


    # -----------------------------------------------
    # Find parent for every retrieved child
    # -----------------------------------------------

    for child in children:

        parent_id = child.metadata[
            "parent_id"
        ]


        # -------------------------------------------
        # Avoid duplicate parents
        # -------------------------------------------

        if parent_id not in seen_parent_ids:

            parent = parent_store[
                parent_id
            ]

            parents.append(parent)

            seen_parent_ids.add(
                parent_id
            )


    return parents


# =========================================================
# 11. ASK USER QUESTION
# =========================================================

question = input(
    "\nAsk a question: "
)


# =========================================================
# 12. RETRIEVE PARENT DOCUMENTS
# =========================================================

parents = parent_child_retrieval(
    question
)


# =========================================================
# 13. BUILD CONTEXT
# =========================================================

context = "\n\n".join(
    parent.page_content
    for parent in parents
)


# =========================================================
# 14. CREATE FINAL PROMPT
# =========================================================

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


# =========================================================
# 15. SEND FULL PROMPT TO LLM
# =========================================================

response = llm.invoke(
    prompt
)


# =========================================================
# 16. PRINT FINAL ANSWER
# =========================================================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)