from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda


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
class FunctionOutput(BaseModel):
    prefix: str = Field(description="The prefix of the output")
    code_block: str = Field(description="The code block of the output")

# Create an instance of the PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=FunctionOutput)

# Get the format instructions from the output parser
format_instructions = parser.get_format_instructions()

def parse_answer_to_str(x):
    """
    Creates a formatted string output for the _code_parser series of models. 
    This can be added with: | RunnableLambda(parse_answer_to_str) to the end of the chain.
 
    The parser is designed to extract a 'prefix' and a 'code_block' from the input.
    The function then concatenates these two parts, separated by a specific delimiter,
    and returns the concatenated string. If parsing fails, the function returns the input as a string.

    Parameters:
    x: Runnable with output following schema as defined in FunctionOutput above

    Returns:
    str: A string that combines the parsed 'prefix' and 'code_block', separated by a delimiter.
         If parsing fails, returns the input converted to a string.

    Note:
    - The function relies on a global 'parser' object of type PydanticOutputParser.
    - The parser is expected to parse inputs into an instance of FunctionOutput, 
      which contains 'prefix' and 'code_block' attributes.
    """
    try:
        parsed_output = parser.invoke(x)
        prefix = parsed_output.prefix
        code_block = parsed_output.code_block
        return prefix + "\n--- -CODE- ---\n" + code_block
    except:
        return str(x)

# Create a prompt template with format instructions and the query
prompt = PromptTemplate(
    template = """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation: 
    \n ------- \n
    {context} 
    \n ------- \n
    Now, answer the user question based on the above provided documentation and ensure any code you provide can be executed with all required imports and variables defined: {question}
    Output format instructions: \n {format_instructions}
    """,
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview") 

chain = (
    {
        "context": lambda x: concatenated_content,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | RunnableLambda(parse_answer_to_str)
)

# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)