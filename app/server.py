from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.context_stuffing_chain import chain as code_langchain_stuff

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, code_langchain_stuff, path="/code-langchain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
