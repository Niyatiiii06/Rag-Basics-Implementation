from langchain_mistralai import ChatMistralAI

from langchain.retrievers import (
    ContextualCompressionRetriever
)

from langchain.retrievers.document_compressors import (
    LLMChainExtractor
)


# ==========================================
# 1. LLM
# ==========================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)


# ==========================================
# 2. NORMAL RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5
    }
)


# ==========================================
# 3. CREATE COMPRESSOR
# ==========================================

compressor = LLMChainExtractor.from_llm(
    llm
)


# ==========================================
# 4. WRAP RETRIEVER
# ==========================================

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)


# ==========================================
# 5. RETRIEVE
# ==========================================

question = input(
    "Ask a question: "
)

docs = compression_retriever.invoke(
    question
)


# ==========================================
# 6. BUILD CONTEXT
# ==========================================

context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# ==========================================
# 7. FINAL PROMPT
# ==========================================

prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""


# ==========================================
# 8. FINAL LLM
# ==========================================

response = llm.invoke(
    prompt
)

print(response.content)