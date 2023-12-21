# code-langchain

Code-langchain is a code assistant designed for LangChain. 

Simply ask a question related LangChain, and it will provide you with code.

## Enviorment 

This will use the `chat-langchain` index in Weaviate with fine-tuned embeddings from Voyage. Be sure to set:

* `WEAVIATE_URL`
* `WEAVIATE_API_KEY`
* `VOYAGE_API_KEY`
* `VOYAGE_AI_MODEL`

## Run

```
from code_langchain import final_answer_chain
question = "how to chat PDF with chroma? Use LCEL"
answer = final_answer_chain.invoke({"question": question})
```

## Eval

See `eval_code_langchin` notebook for testing.