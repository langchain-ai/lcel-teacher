import langsmith
import langsmith.env
from app.context_stuffing_chain import chain as code_langchain_stuff
from langchain.smith import RunEvalConfig
import uuid


if __name__ == "__main__":
    client = langsmith.Client()
    git_info = langsmith.env.get_git_info()
    branch, commit = git_info["branch"], git_info["commit"]
    project_name = f"code-langchain-{branch}-{commit[:4]}-{uuid.uuid4().hex[:4]}"
    eval_config = RunEvalConfig(
        evaluators=["qa"],
    )
    test_results = client.run_on_dataset(
        dataset_name="code-langchain-eval",
        llm_or_chain_factory=lambda: (lambda x: x["question"]) | code_langchain_stuff,
        project_name=project_name,
        evaluation=eval_config,
        verbose=True,
        project_metadata={"context": "regression-tests"},
    )
    test_results.get_aggregate_feedback()
