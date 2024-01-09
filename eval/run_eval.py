import langsmith
from app.context_stuffing_chain import chain as code_langchain_stuff
from langchain.smith import RunEvalConfig

# Config
if __name__ == "__main__":
    client = langsmith.Client()
    eval_config = RunEvalConfig(
        evaluators=["cot_qa"],
    )
    project_name = "code-langchain-eval"
    test_results= client.run_on_dataset(
        dataset_name="code-langchain-eval",
        llm_or_chain_factory=lambda: (lambda x: x["question"]) | code_langchain_stuff,
        evaluation=eval_config,
        verbose=True,
        project_metadata={"context": "regression-tests"},
    )
    