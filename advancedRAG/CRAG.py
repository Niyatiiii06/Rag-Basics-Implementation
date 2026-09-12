from langchain_core.prompts import ChatPromptTemplate


class CRAGEvaluator:

    def __init__(self, llm, threshold: float = 0.5):
        self.llm = llm
        self.threshold = threshold

        self.prompt = ChatPromptTemplate.from_template("""
You are evaluating retrieval quality for a RAG system.

Question:
{query}

Document:
{document}

Give a relevance score from 0 to 1.

0 = irrelevant
1 = directly useful for answering the question

Return ONLY the number.
""")

    def evaluate(self, query, documents):
        scored_documents = []

        for document in documents:
            response = self.llm.invoke(
                self.prompt.format(
                    query=query,
                    document=document.page_content,
                )
            )

            try:
                score = float(response.content.strip())
            except ValueError:
                score = 0.0

            scored_documents.append((document, score))

        relevant = [
            document
            for document, score in scored_documents
            if score >= self.threshold
        ]

        return relevant, scored_documents