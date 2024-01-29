import weaviate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.voyageai import VoyageEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

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

# Output
class FunctionOutput(BaseModel):
    prefix: str = Field(description="The prefix of the output")
    code_block: str = Field(description="The code block of the output")

# Create an instance of the PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=FunctionOutput)

# Get the format instructions from the output parser
format_instructions = parser.get_format_instructions()

# Create a prompt template with format instructions and the query
prompt = PromptTemplate(
    template = """You are a coding assistant with expertise in LangChain. \n 
    Here is relevant context: 
    \n ------- \n
    {context} 
    \n ------- \n
    Now, answer the user question based on the above provided documentation: {question}
    Output format instructions: \n {format_instructions}
    """,
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

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