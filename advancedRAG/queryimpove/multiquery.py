def generate_queries(question):
    prompt= f''' Generate 3 diff search queries. preserve the meaning in all 3 versions of the same questoin. the user question is {question}'''
    response= llm.invoke(prompt)
    queries= response.content.split('\n')
    return queries

def remove_duplicates(docs):
    unique_docs={}
    for doc in docs:
        unique_docs[doc.page_content]=doc
    return list(unique_docs.values())

def multi_query_retrievel(question,retriever):
    generated_queries= generate_queries(question)
    queries=[question]+ generated_queries
    all_docs=[]
    for query in queries:
        docs= retriever.invoke(query)
        all_docs.extend(docs)
    final_docs= remove_duplicates(all_docs)
    return final_docs

question= input('You: ')
retrieved_docs= multi_query_retrievel(question,retriever)