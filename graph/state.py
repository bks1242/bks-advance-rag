from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
      question: question
      generation: LLM generation
      web_search: whether to add the search
      documents: list of documents

    """

    documents: List[str]
    question: str
    generation: str
    web_search: bool
