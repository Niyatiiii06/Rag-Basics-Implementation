from dotenv import load_dotenv

from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)


load_dotenv()


# ============================================================
# 1. MODELS
# ============================================================

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# 2. VECTOR STORE
# ============================================================

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)


# ============================================================
# 3. RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,
        "fetch_k": 50,
        "lambda_mult": 0.5,
    },
)


# ============================================================
# 4. RERANKER
# ============================================================

def rerank_documents(
    query: str,
    documents,
    top_k: int = 5,
):

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    scored_documents = []

    for document, score in zip(
        documents,
        scores,
    ):

        scored_documents.append(
            {
                "document": document,
                "score": float(score),
            }
        )

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_documents[:top_k]


# ============================================================
# 5. RETRIEVE + RERANK
# ============================================================

def retrieve_documents(
    question: str,
):

    documents = retriever.invoke(
        question
    )

    if not documents:
        return []

    reranked_results = rerank_documents(
        question,
        documents,
        top_k=5,
    )

    return [
        result["document"]
        for result in reranked_results
    ]


# ============================================================
# 6. BUILD CONTEXT
# ============================================================

def build_context(
    documents,
) -> str:

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================================
# 7. GENERATE ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str,
) -> str:

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY
the provided context.

Rules:
- Do not invent information.
- If the answer is not present in
  the context, say you don't know.
- Keep the answer factual and concise.

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
# 8. COMPLETE PIPELINE
# ============================================================

def answer_question(
    question: str,
) -> str:

    documents = retrieve_documents(
        question
    )

    if not documents:
        return (
            "I don't know based on "
            "the available context."
        )

    context = build_context(
        documents
    )

    return generate_answer(
        question,
        context,
    )


# ============================================================
# 9. APPLICATION
# ============================================================

if __name__ == "__main__":

    print("Reranked RAG")
    print("Type 'exit' to quit.")

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