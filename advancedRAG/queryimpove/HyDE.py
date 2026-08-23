def generate_hypothetical_document(question):

    prompt = f"""
Write a hypothetical answer to the user's question.

The answer should contain useful factual information
and terminology that might appear in relevant documents.

Do not say that you are generating a hypothetical answer.

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content.strip()

question= input('you :')
hypothetical_doc = generate_hypothetical_document(question)
docs = retriever.invoke(hypothetical_doc)
context='\n\n'.join(doc.page_content for doc in docs)
response= llm.invoke(prompt)