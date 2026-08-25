from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
)


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 2. MODELS
# ============================================================

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
)


# ============================================================
# 3. VECTOR STORE
# ============================================================

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)


# ============================================================
# 4. BASE RETRIEVER
# ============================================================

base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.5,
    },
)


# ============================================================
# 5. CONTEXT COMPRESSOR
# ============================================================

compressor = LLMChainExtractor.from_llm(
    llm
)


# ============================================================
# 6. COMPRESSION RETRIEVER
# ============================================================

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)


# ============================================================
# 7. RETRIEVAL FUNCTION
# ============================================================

def retrieve_context(
    question: str,
):
    """
    Retrieve relevant documents and
    compress their content based on
    the user's question.
    """

    documents = compression_retriever.invoke(
        question
    )

    return documents


# ============================================================
# 8. CONTEXT BUILDER
# ============================================================

def build_context(
    documents,
) -> str:

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "unknown",
        )

        context_parts.append(
            f"""
[Source {index}: {source}]

{document.page_content}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# 9. GENERATION
# ============================================================

def generate_answer(
    question: str,
    context: str,
) -> str:

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY
the information provided in the context.

Rules:
- Do not invent information.
- If the context does not contain the answer,
  say that you don't know.
- Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# ============================================================
# 10. COMPLETE RAG PIPELINE
# ============================================================

def answer_question(
    question: str,
) -> str:

    documents = retrieve_context(
        question
    )

    if not documents:
        return "I don't know based on the available context."

    context = build_context(
        documents
    )

    return generate_answer(
        question,
        context,
    )


# ============================================================
# 11. APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "Contextual Compression RAG"
    )
    print(
        "Type 'exit' to quit."
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = answer_question(
            question
        )

        print(
            f"\nAnswer:\n{answer}"
        )