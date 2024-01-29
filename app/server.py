from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.deploy_chain import chain as chain_to_deploy

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, chain_to_deploy, path="/lcel-teacher")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
