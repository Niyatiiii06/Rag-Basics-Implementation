retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5})

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0)

def expand_query(question):
    prompt = f"""
    Expand the user's search query by adding
    important related keywords, concepts, and
    technical terms.

    Keep the original meaning.

    Return one expanded search query.

    User query:
    {question}"""

    response = llm.invoke(prompt)
    return response.content.strip()

question = input(
    "\nAsk a question: ")

expanded_query = expand_query(
    question)

docs = retriever.invoke(
    expanded_query)

context = "\n\n".join(
    doc.page_content
    for doc in docs)

prompt = f"""
You are a helpful AI assistant.
Answer the user's question using ONLY
the provided context
If the answer is not available in the context,
say that you don't know.
Context:
{context}
Question:
{question}
Answer:"""

response = llm.invoke(
    prompt)