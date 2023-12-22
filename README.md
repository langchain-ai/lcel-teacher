Code-Langchain

## Introduction

Code-langchain is a code assistant designed for LangChain. 

Simply ask a question related LangChain, and it will provide you with code.

The workflow uses research-assistant type sub-question generation-and-answering on the LangChain codebase, which is then passed as Q-A pairs for a second LLM call for final answer synthesis.

![image](https://github.com/langchain-ai/code-langchain-v2/assets/122662504/466544df-4a26-41f6-a29e-ac3a94028b23)

## Enviorment 

This will use the `chat-langchain` index in Weaviate with fine-tuned embeddings from Voyage. Set:

* `WEAVIATE_URL`
* `WEAVIATE_API_KEY`
* `VOYAGE_API_KEY`
* `VOYAGE_AI_MODEL`

## App structure

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

We won't need `packages`:
```
rm -rf packages
```

**(2) Add your runnable**

Create a file, `chain.py` with a runnable named `chain` that you want to execute.

Add `chain.py` to `app` directory.

Import the runnable in `server.py`:
```
from app.chain import chain as code_langchain_run
add_routes(app, code_langchain_run, path="/code-langchain")
```

Add your app dependencies to `pyproject.toml` and `poetry.lock`:
```
poetry add weaviate-client
poetry add langchainhub
```

Update enviorment based on the updated lock file:
```
poetry install
```

Run
```
poetry run langchain serve
```

## Eval

In `eval/` you will see `eval.csv`.

Update this to LangSmith as a dataset, `code-langchain-eval`.

Run notebook to kick off eval:
```
poetry run jupyter notebook
```

You can see an example eval [here](https://smith.langchain.com/public/747fea3b-7fa1-441b-8080-80f5e09ec518/d).