# Testing GenAI application is tricky
# No idempotency
# Relying on 3rd party
# cost
# we can use cheap models for testing purpose

from dotenv import load_dotenv

load_dotenv()

from pprint import pprint

from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.router import RouteQuery, question_router


# sample test case
def test_foo() -> None:
    assert 1 == 1


def test_retrievel_grader_answer_yes() -> None:

    # here we are retrieving the document based on the question and then passing the first content to the grader, first content beacause it is highly likely that it would be relevant
    question = "agent memory"
    docs = retriever.invoke(question)
    docs_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": docs_text}
    )

    assert res.binary_score == "yes"


def test_retrievel_grader_answer_no() -> None:

    question = "How to make burger"
    docs = retriever.invoke(question)
    docs_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": docs_text}
    )

    assert res.binary_score == "yes"


def test_generation_chain() -> None:
    question = "How to make burger"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
