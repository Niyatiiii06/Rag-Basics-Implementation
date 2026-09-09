from typing import List

from langchain_mistralai import ChatMistralAI


class QueryDecomposer:
    """
    Breaks a complex user query into smaller
    independent search queries.
    """

    def __init__(
        self,
        llm: ChatMistralAI,
        max_sub_queries: int = 4,
    ):
        self.llm = llm
        self.max_sub_queries = max_sub_queries

    def decompose(
        self,
        query: str,
    ) -> List[str]:

        prompt = f"""
You are a query decomposition system for a RAG pipeline.

Your task is to determine whether the user's query
needs to be broken into smaller search queries.

If the query is simple and can be answered with one
search, return the original query unchanged.

If the query is complex, break it into smaller,
independent questions that can each be searched
against a document collection.

Rules:
- Return at most {self.max_sub_queries} queries.
- Each query must be self-contained.
- Do not answer the questions.
- Do not add information that is not present in the
  original query.
- Return one query per line.
- Do not number the queries.
- Do not use bullet points.

User query:
{query}
"""

        response = self.llm.invoke(prompt)

        sub_queries = [
            line.strip()
            for line in response.content.splitlines()
            if line.strip()
        ]

        # Remove accidental numbering/bullets
        cleaned_queries = []

        for sub_query in sub_queries:

            sub_query = sub_query.lstrip(
                "0123456789.-) "
            )

            if sub_query:
                cleaned_queries.append(
                    sub_query
                )

        # Safety limit
        cleaned_queries = cleaned_queries[
            :self.max_sub_queries
        ]

        # Fallback
        if not cleaned_queries:
            return [query]

        return cleaned_queries

if __name__ == "__main__":

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
    )

    decomposer = QueryDecomposer(
        llm=llm,
        max_sub_queries=4,
    )

    query = input("Enter a query: ")

    sub_queries = decomposer.decompose(query)

    print("\nOriginal Query:")
    print(query)

    print("\nSub-Queries:")

    for i, sub_query in enumerate(
        sub_queries,
        start=1,
    ):
        print(f"{i}. {sub_query}")