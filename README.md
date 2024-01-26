# LCEL-Teacher

## Introduction

[LangChain Expression Language](https://python.langchain.com/docs/expression_language/) has a number of important benefits, including transparent composability of LLM components, seamless support for prototyping and production (with [LangServe](https://python.langchain.com/docs/langserve)) using the same code, and a common [interface](https://python.langchain.com/docs/expression_language/interface) for every chain. But, there a learning curve for using LCEL. Here, we aim to build a coding assistant for LCEL. 

## Architecture

We explore three several architectures for LCEL-teacher in this repo, including:

* `Context stuffing` of LCEL docs into the LLM context window
* `RAG` using retrieval from a vector databases of all LangChain documentation  
* `RAG using multi-question and answer generation` using retrieval from a vector databases of all LangChain documentation  

![rag_code_langchain](https://github.com/langchain-ai/lcel-teacher/assets/122662504/1765a68b-e143-42be-8d1a-cefc177aa66f)

Code for each can be found in the `/app` directory.

## Environment 

We use Poetry for depeendency management. 
 
`Context stuffing` requires no vectorstore access because we will directly read the docs and stuff them into the LLM context windopw.
 
Both `RAG` approaches rely on an vectorstore index of LangChain documentation (Weaviate) with fine-tuned embeddings from Voyage:

* `WEAVIATE_URL`
* `WEAVIATE_API_KEY`
* `VOYAGE_API_KEY`
* `VOYAGE_AI_MODEL`

## Using the app

This repo is a LangServe app. We host it using hosted LangServe. To learn more [see this video](https://www.youtube.com/watch?v=EhlPDL4QrWY).

You can access it [here](https://code-langchain-deployment-455c22dd058e5e3194aec23-ffoprvkqsa-uc.a.run.app/code-langchain/playground/).

The deployment is [here](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/host/922b4b05-1ea1-475a-99e8-2a554b9c5101).

The steps to deploy it are shown below.

--- 

## Running locally and deployment

This repo was created following these steps:

**(1) Create a LangChain app.**

Run:
```
langchain app new .  
```

This creates two folders:
```
app: This is where LangServe code will live
packages: This is where your chains or agents will live
```

It also creates:
```
Dockerfile: App configurations
pyproject.toml: Project configurations
```

Add app dependencies to `pyproject.toml` and `poetry.lock`:
```
poetry add weaviate-client
poetry add langchainhub
poetry add openai
poetry add pandas
poetry add jupyter
poetry add tiktoken
poetry add scikit-learn
poetry add langchain_openai
```

Update enviorment based on the updated lock file:
```
poetry install
```

**(2) Add the chains**

Add our custom retrieval code to the `app` directory.

In our case, I add the various `_chain.py` files.

Each file simply has a LCEL chain defined. For example:

```
chain = (
    {
        "context": lambda x: concatenated_content,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
```

Now, we simply import the chain in `server.py`:
```
from app.context_stuffing_chain import chain as code_langchain_stuff
add_routes(app, code_langchain_stuff, path="/code-langchain")
```

Run locally
```
poetry run langchain serve
```

Simply, the invocation methods of our LCEl chain are mapped to HTTP endpoints in the LangServe app:
![Screenshot 2024-01-26 at 11 48 06 AM](https://github.com/langchain-ai/lcel-teacher/assets/122662504/46c4f65b-1719-4212-b450-142062fd0d5b)

For hosted LangServe, sign up in your LangSmith console on the `Deployments` tab and connect to your fork of this repo.

## Eval

In `eval/` you will see `eval.csv`.

Use this to create a LangSmith as a [dataset](https://smith.langchain.com/public/3b0fe661-e3ed-4d84-9d88-96c7ee8c4a2d/d), `lcel-teacher-eval`.

Run notebook to kick off eval:
```
poetry run jupyter notebook
```

Use `eval/eval_lcel_teacher.ipynb` to run evals.
