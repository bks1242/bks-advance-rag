# load articles to documents
# chunk documents
# Embed
# Store in chroma DB / pinecone vector store

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_EMBEDDING_OPENAI_DEPLOYMENT"],
    openai_api_key=os.environ["AZURE_EMBEDDING_OPENAI_KEY"],
    azure_endpoint=os.environ["AZURE_EMBEDDING_OPENAI_API_BASE"],
    openai_api_type=os.environ["AZURE_EMBEDDING_OPENAI_API_TYPE"],
    openai_api_version=os.environ["AZURE_EMBEDDING_OPENAI_API_VERSION"],
)

# urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
# ]

urls = []

# load the docs from the urls
docs = [WebBaseLoader(url).load() for url in urls]
# convert it into the docs list
docs_list = [item for sublist in docs for item in sublist]

# split using the RecursiveCharacterTextSplitter using tiktoken encoder
# create the instance
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# pass the the documents list to split it.
docs_splits = text_splitter.split_documents(docs_list)

# vector_store = Chroma.from_documents(
#     documents=docs_splits, collection_name="rag-chroma", embedding=embeddings, persist_directory="./.chroma"
# )

# retriever = Chroma(
#     collection_name="rag-chroma",
#     persist_directory="./.chroma",
#     embedding_function=embeddings()
# ).as_retriever()

# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# PineconeVectorStore.from_documents(
#     docs_splits, embeddings, index_name="advance-rag-index"
# )

docsearch = PineconeVectorStore(embedding=embeddings, index_name="advance-rag-index")
retriever = docsearch.as_retriever()


if __name__ == "__main__":
    print("Ingestion....")
# ingest_docs()
