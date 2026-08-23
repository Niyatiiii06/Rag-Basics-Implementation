def rrf(result_list, rrf_k=60):
    scores={}
    for results in result_list:
        for rank,doc in enumerate(results,start=1):
           doc_id = doc.page_content
            if doc_id not in scores:
                scores[doc_id] = {
                    "score": 0,"doc": doc}

            scores[doc_id]["score"] += (
                1 / (rrf_k + rank))
    ranked = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    return [
        item["doc"]
        for item in ranked
    ] 

def generate_queries(question):
    prompt= f''' Generate 3 diff search queries. preserve the meaning in all 3 versions of the same questoin. the user question is {question}'''
    response= llm.invoke(prompt)
    queries= response.content.split('\n')
    return queries

def multi_query_retrievel(question,retriever):
    generated_queries= generate_queries(question)
    queries=[question]+ generated_queries
    result_list=[]
    for query in queries:
        docs= retriever.invoke(query)
        result_list.append(docs)
    final_docs= rrf(result_list)
    return final_docs