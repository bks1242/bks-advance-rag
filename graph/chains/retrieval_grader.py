# retrieval grader chain
# structured output
# document Grader Node
# here we will recieve original question
# and based on the question we will retrieve the documents and see if it is relevant or not, we are going to run this chain for each document we retrieve
# if it not relevant then we will do the websearch
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
import os

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_type="azure",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    openai_api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are grader assessing relevance of a retrieved document  to a user question.\n
If the document  contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n{document} \n\n user question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
