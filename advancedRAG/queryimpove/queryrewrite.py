llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

def rewrite_query(question):
    prompt = f"""
    Rewrite the user's question into a clear,
    specific search query.
    Preserve the original meaning.
    Do not answer the question.
    User question:
    {question}"""
    response = llm.invoke(prompt)
    return response.content.strip()

question = input(
    "\nAsk a question: ")

rewritten_query = rewrite_query(
    question)

docs = retriever.invoke(
    rewritten_query)

context = "\n\n".join(
    doc.page_content
    for doc in docs)

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
Answer:"""

response = llm.invoke(
    prompt)