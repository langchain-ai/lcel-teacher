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

## Running the app

This is a hosted LangServe app. 

You can access it [here](https://code-langchain-deployment-455c22dd058e5e3194aec23-ffoprvkqsa-uc.a.run.app/code-langchain/playground/).

The deployment is [here](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/host/922b4b05-1ea1-475a-99e8-2a554b9c5101).

The steps to deploy it are shown below.

You can run it locally simply with:

```
from app.chain import chain as code_langchain_chain
question = "how to chat PDF with chroma? Use LCEL"
answer = code_langchain_chain.invoke({"question": question})
```

## Deployment

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
poetry add openai
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

Use this to create a LangSmith as a dataset, `code-langchain-eval`.

Run notebook to kick off eval:
```
poetry run jupyter notebook
```

You can see an example eval [here](https://smith.langchain.com/public/85ce2833-3ef3-44fe-a282-e50d51767653/d).