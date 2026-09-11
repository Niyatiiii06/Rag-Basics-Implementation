from langchain_mistralai import ChatMistralAI
from typing import List
from langchain_core.prompts import ChatPromptTemplate


class QueryDecomposer:

    def __init__(self, llm, max_sub_queries: int = 4):
        self.llm = llm
        self.max_sub_queries = max_sub_queries

        self.prompt = ChatPromptTemplate.from_template(
            """
You are a query decomposition system for a RAG pipeline.

Break the user's complex question into smaller,
independent sub-questions that can be retrieved separately.

Rules:
- If the query is already simple, return it unchanged.
- Create at most {max_sub_queries} sub-queries.
- Each sub-query should be self-contained.
- Do not answer the questions.
- Return only one sub-query per line.

User query:
{query}
"""
        )

    def decompose(self, query: str) -> List[str]:
        # -----------------------------------------
        # Generate sub-queries
        # -----------------------------------------

        response = self.llm.invoke(
            self.prompt.format(
                query=query,
                max_sub_queries=self.max_sub_queries,
            )
        )

        # -----------------------------------------
        # Clean LLM output
        # -----------------------------------------

        queries = [
            line.strip("-• ").strip()
            for line in response.content.splitlines()
            if line.strip()
        ]

        # -----------------------------------------
        # Remove duplicates
        # -----------------------------------------

        unique_queries = list(dict.fromkeys(queries))

        # -----------------------------------------
        # Fallback
        # -----------------------------------------

        if not unique_queries:
            return [query]

        return unique_queries[:self.max_sub_queries]
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