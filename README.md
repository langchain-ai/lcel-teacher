Code-Langchain

## Introduction

Code-langchain is a code assistant designed for LangChain. 

Simply ask a question related LangChain, and it will provide you with code.

The workflow uses research-assistant type sub-question generation-and-answering on the LangChain codebase, which is then passed as Q-A pairs for a second LLM call for final answer synthesis.

![image](https://github.com/langchain-ai/code-langchain-v2/assets/122662504/466544df-4a26-41f6-a29e-ac3a94028b23)

## Enviorment 

* Generate sub-questions based on the user-input
* Answer each using retrieval from LangChain codebase
* Pass those question-answer pairs to an LLM for final code synthesis

## App structure

We deploy this app on hosted LangServe.

This repo was created following these steps:

**(1) Create a LangChain template app.**

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

**(2) Create a LangChain template**

Run:
```
cd packages
langchain template new code-langchain
```

This creates a new template ([read more](https://github.com/langchain-ai/langchain/tree/master/templates#quick-start)).

We add our code-langchain specific logic to `packages/code_langchain/code_langchain/chain.py`.

We add our code-langchain specific dependencies to `packages/code_langchain/pyproject.toml`:
```
poetry add weaviate-client
poetry add langchainhub
```

**(3) Configure the app to use our template**

Update the app `pyproject.toml` to include our template:
```
code_langchain = {path = "packages/code_langchain", develop = true}
```

Update the app `app/server.py` to include our template:
```
from code_langchain import chain as code_langchain_run
add_routes(app, code_langchain_run, path="/code_langchain")
```

**(4) Test**

In the project root update the lock file with deps from both `.toml` files:
```
poetry lock
```

Update enviorment based on the lock file:
```
poetry install
```

Run:
```
poetry run langchain serve
```
