import langsmith

if __name__ == "__main__":
    client = langsmith.Client()
    client.upload_csv(
        csv_file="eval/eval.csv",
        input_keys=["question"],
        output_keys=["answer"],
        name="code-langchain-eval",
    )
