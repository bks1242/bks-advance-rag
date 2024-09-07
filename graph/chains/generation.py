from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()
import os

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
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
