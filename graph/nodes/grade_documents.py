from typing import Dict, Any

from graph.chains.retrieval_grader import retrieval_grader

from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    
    print("---Check document relevance to the question---")
    
    question = state["question"]
    documents = state["documents"]
    
    #this would contain all the relevant docs, initially it is empty, however once to travers to each docs for relevancy check, based on it we will fill it.
    filteres_docs = []
    #Initially setting it to false
    web_search = False
    
    for d in documents:
      score = retrieval_grader.invoke({"question": question, "document": d.page_content})
      grade = score.binary_score
      
      if grade.lower() == 'yes':
        print("-- GRADE DOCUMENT: RELEVANT--")
        filteres_docs.append(d)
      else:
        print("-- GRADE DOCUMENT: NOT RELEVANT--")
        web_search = True
        continue
    return {"documents": filteres_docs, "question": question, "web_search": web_search}
        
