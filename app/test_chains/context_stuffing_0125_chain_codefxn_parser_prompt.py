from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_community.utils.openai_functions import convert_pydantic_to_openai_function

# Load LCEL docs
url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# LCEL w/ PydanticOutputParser (outside the primary LCEL docs)
url = "https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start"
loader = RecursiveUrlLoader(
    url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
)
docs_pydantic = loader.load()

# LCEL w/ Self Query (outside the primary LCEL docs)
url = "https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/"
loader = RecursiveUrlLoader(
    url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
)
docs_sq = loader.load()

# Add 
docs.extend([*docs_pydantic, *docs_sq])

# Sort the list based on the URLs in 'metadata' -> 'source'
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))

# Concatenate the 'page_content' of each sorted dictionary
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

# Output
class CodeBlock(BaseModel):
    """Code extraction from the LLM."""
    prefix: str = Field(description="Text preamble explaining the context of the code solution.")
    code_block: str = Field(description="The code block that provides the solution.")

parser = PydanticOutputFunctionsParser(pydantic_schema=CodeBlock)
openai_functions = [convert_pydantic_to_openai_function(CodeBlock)]

# Create a prompt template with format instructions and the query
prompt = PromptTemplate(
    template = """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation: 
    \n ------- \n
    {context} 
    \n ------- \n
    Now, answer the user question based on the above provided documentation and ensure any code you provide can be executed with all required imports and variables defined: {question}
    """,
    input_variables=["question"],
)

model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview") 

chain = (
    {
        "context": lambda x: concatenated_content,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model.bind(functions=openai_functions)
    | parser
)

# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)