Code-Langchain

## Introduction

Code-langchain is a code assistant designed for LangChain. 

Simply ask a question related LangChain, and it will provide you with code.

It will:

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

It also create:
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

We add our code-langchain specific dependencies to `packages/code_langchain/pyproject.toml`.

**(3) Configure the app to use our template**

Update the app `pyproject.toml` to include our template:
```
code-langchain = {path = "packages/code_langchain", develop = true}
```

Update the app `app/server.py` to include our template:
```
from code_langchain import chain as code_langchain_run

add_routes(app, code_langchain_run, path="/code-langchain")
```
