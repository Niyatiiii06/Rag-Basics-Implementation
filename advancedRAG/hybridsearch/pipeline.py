from retriever import hybrid_retrieve
from reranker import DocumentReranker


MAX_CANDIDATES = 30
FINAL_TOP_K = 5


class HybridRerankPipeline:

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        llm,
    ):

        self.vector_retriever = (
            vector_retriever
        )

        self.bm25_retriever = (
            bm25_retriever
        )

        self.llm = llm

        self.reranker = DocumentReranker(
            top_k=FINAL_TOP_K
        )


    def retrieve(self, query):

        # Hybrid retrieval
        candidates = hybrid_retrieve(
            query=query,
            vector_retriever=(
                self.vector_retriever
            ),
            bm25_retriever=(
                self.bm25_retriever
            ),
            max_candidates=MAX_CANDIDATES,
        )

        if not candidates:
            return []

        # Cross-encoder reranking
        reranked = self.reranker.rerank(
            query,
            candidates,
        )

        # Return only documents
        return [
            result["document"]
            for result in reranked
        ]


    def build_context(self, documents):

        return "\n\n".join(
            document.page_content
            for document in documents
        )


    def generate_answer(
        self,
        query,
        context,
    ):

        prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY
the provided context.

Rules:
- Do not invent information.
- If the answer is not present,
  say you don't know.
- Keep the answer concise and factual.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content


    def answer(self, query):

        # Retrieve + rerank
        documents = self.retrieve(
            query
        )

        if not documents:
            return (
                "I don't know based on "
                "the available information."
            )

        # Build context
        context = self.build_context(
            documents
        )

        # Generate answer
        return self.generate_answer(
            query,
            context,
        )