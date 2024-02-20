import weaviate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.prompts import PromptTemplate
from langchain.embeddings.voyageai import VoyageEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool

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

## Data model
class code(BaseModel):
    """Code output"""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

## LLM
model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

# Tool
code_tool_oai = convert_to_openai_tool(code)

# LLM with tool and enforce invocation
llm_with_tool = model.bind(
    tools=[convert_to_openai_tool(code_tool_oai)],
    tool_choice={"type": "function", "function": {"name": "code"}},
)

# Parser
parser_tool = PydanticToolsParser(tools=[code])

# Create a prompt template with format instructions and the query
prompt = PromptTemplate(
    template = """You are a coding assistant with expertise in LangChain. \n 
    Here is relevant context: 
    \n ------- \n
    {context} 
    \n ------- \n
    Ensure any code you provide can be executed with all required imports and variables defined. \n
    Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. \n
    Here is the user question: \n --- --- --- \n {question}""",
    input_variables=["question","context"])

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_with_tool
    | parser_tool
)

# Add typing for input
class Question(BaseModel):
    __root__: str

chain = chain.with_types(input_type=Question)