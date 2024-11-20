from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from .node import (
    GraphState,
    question_evaluation,
    table_selection,
    non_sql_conversation,
    user_question_checker,
    user_question_analyze_checker,
    query_creation,
    question_analyze,
    question_refine,
    question_clarify,
)


def make_graph() -> CompiledStateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("질문 평가", question_evaluation)
    workflow.add_node("일반적인 대화", non_sql_conversation)
    workflow.add_node("질문 분석", question_analyze)
    workflow.add_node("추가 질문", question_clarify)
    workflow.add_node("질문 구체화", question_refine)
    workflow.add_node("Table 선택", table_selection)
    workflow.add_node("SQL 쿼리문 생성", query_creation)

    workflow.add_conditional_edges(
        "질문 평가",  # 사용자의 질문이 데이터 및 비즈니스와 관련 되어있는지 평가
        user_question_checker,  # check로 평가 데이터를 추출한 뒤, 아래의 조건에 따라 다른 노드에 전달
        {
            "0": "일반적인 대화",
            "1": "질문 분석",
        },
    )

    workflow.add_conditional_edges(
        "질문 분석",
        user_question_analyze_checker,
        {
            True: "추가 질문",
            False: "질문 구체화",
        },
    )

    workflow.add_edge("추가 질문", "질문 구체화")
    workflow.add_edge("질문 구체화", "Table 선택")
    workflow.add_edge("Table 선택", "SQL 쿼리문 생성")

    # 시작점을 설정합니다.
    workflow.set_entry_point("질문 평가")

    # 기록을 위한 메모리 저장소를 설정합니다.
    memory = MemorySaver()

    # 그래프를 컴파일합니다.
    app = workflow.compile(checkpointer=memory)

    return app
