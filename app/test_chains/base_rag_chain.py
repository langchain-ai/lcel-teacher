import weaviate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Weaviate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.voyageai import VoyageEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel

# Keys
import os

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
WEAVIATE_DOCS_INDEX_NAME = "LangChain_agent_docs"

# Fine-tuned embd and vectorstore
def get_embeddings_model():
    if os.environ.get("VOYAGE_API_KEY") and os.environ.get("VOYAGE_AI_MODEL"):
        return VoyageEmbeddings(model=os.environ["VOYAGE_AI_MODEL"])
    return OpenAIEmbeddings(chunk_size=200)


def get_retriever():
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source", "title"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))

# Retriever
retriever = get_retriever()

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(model="gpt-4-1106-preview")

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str

chain = chain.with_types(input_type=Question)