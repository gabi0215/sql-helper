from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langgraph_.graph import make_graph, multiturn_test, make_graph_for_test
from langgraph_.utils import get_runnable_config

app = FastAPI()

# make_graph: 전체 과정, make_graph_for_test: 질문 구체화 생략, multiturn_test: 질문 구체화만 진행
workflow = make_graph_for_test()
print("LLM WORKFLOW STARTED.")


class LLMWorkflowInput(BaseModel):
    user_question: str
    initial_question: int
    thread_id: str
    last_snapshot_values: dict | None


@app.post("/llm_workflow")
def llm_workflow(workflow_input: LLMWorkflowInput):
    processed_input = workflow_input.model_dump()

    config = get_runnable_config(30, processed_input["thread_id"])
    inputs = {
        "user_question": processed_input["user_question"],
        "context_cnt": 10,
        "max_query_fix": 2,
        "query_fix_cnt": -1,
        "sample_info": 3,
    }

    outputs = workflow.invoke(
        input=inputs,
        config=config,
    )
    return outputs


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
