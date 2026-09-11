from langchain_core.prompts import ChatPromptTemplate


class HyDE:

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
Generate a hypothetical document that could contain the
answer to the user's question.

Do not explain your reasoning.
Write only the hypothetical document.

Question:
{query}
""")

    def generate(self, query: str) -> str:
        # Generate hypothetical document
        response = self.llm.invoke(
            self.prompt.format(query=query)
        )
        return response.content.strip()