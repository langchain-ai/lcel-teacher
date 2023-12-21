import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

import weaviate
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.voyageai import VoyageEmbeddings
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
)
from langchain.vectorstores import Weaviate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

# Prompts
from .prompts import REPHRASE_TEMPLATE, RESPONSE_TEMPLATE

# Keys
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
WEAVIATE_DOCS_INDEX_NAME = "LangChain_agent_docs"


# Define the data structure for chat requests
class ChatRequest(BaseModel):
    question: str  # The question asked in the chat
    chat_history: Optional[List[Dict[str, str]]]  # Optional chat history


# Function to get the embeddings model based on environment variables
def get_embeddings_model() -> Embeddings:
    # Check for specific environment variables to determine the embeddings model
    if os.environ.get("VOYAGE_API_KEY") and os.environ.get("VOYAGE_AI_MODEL"):
        return VoyageEmbeddings(model=os.environ["VOYAGE_AI_MODEL"])
    # Default to OpenAI embeddings if the specific environment variables are not set
    return OpenAIEmbeddings(chunk_size=200)


# Function to initialize and return the retriever
def get_retriever() -> BaseRetriever:
    # Initialize Weaviate client with authentication and connection details
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    # Configure the Weaviate client with specific settings
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source", "title"],
    )
    # Return the configured retriever
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))


# Function to create a chain of retrievers
def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    # Template to condense the question
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    # Create a chain to process the question and retrieve relevant information
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    # Return a branch of runnables depending on whether there's chat history
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


# Function to format the retrieved documents
def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    # Iterate through each document and format it
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


# Function to serialize the chat history from a chat request
def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    # Convert each message in the chat history to the appropriate message type
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


# Function to create the answer chain
def create_question_anwser_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    # Create a retriever chain and configure it
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
        }
    ).with_config(run_name="RetrieveDocs")
    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            ("human", "{question}"),
        ]
    )

    # Create a response synthesizer using the defined prompt
    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    # Return the final chain of processes
    return (
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
        }
        | _context
        | response_synthesizer
    )


# Retriever
retriever = get_retriever()

# Sub-question prompt
sub_question_prompt = hub.pull("hwchase17/code-langchain-sub-question")

# Chain for sub-question generation
sub_question_chain = (
    RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
    | sub_question_prompt
    | ChatOpenAI(model="gpt-4-1106-preview")
    | SimpleJsonOutputParser()
)

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    streaming=True,
    temperature=0,
)

# Chain that answers questions
answer_chain = create_question_anwser_chain(
    llm,
    retriever,
)

# Chain for sub-question answering
sub_question_answer_chain = (
    sub_question_chain
    | (lambda x: [{"question": v} for v in x])
    | RunnablePassthrough.assign(answer=answer_chain).map()
)

# Prompt template for final answer
template = """You are an expert coder. You got a high level question:

<question>
{question}
</question>

Based on this question, you broke it down into sub questions and answered those. These are the results of that:

<subquestions>
{subq}
</subquestions>
    
Now, combine all the subquestion answers to generate a final code snippet writing the code that was asked for.
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model="gpt-4")

# Answer chain
chain = (
    RunnablePassthrough().assign(
        subq=sub_question_answer_chain
        | (
            lambda sub_questions_answers: "\n\n".join(
                [
                    f"Question: {q['question']}\n\nAnswer: {q['answer']}"
                    for q in sub_questions_answers
                ]
            )
        )
    )
    | prompt
    | llm
    | StrOutputParser()
)
