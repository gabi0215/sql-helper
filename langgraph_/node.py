from typing import TypedDict, List
from task import (
    evaluate_user_question,
    simple_conversation,
    do_embodiment,
    do_extraction,
)


# GrpahState 정의
class GraphState(TypedDict):
    # Warning!
    # 그래프 내에서 사용될 모든 key값을 정의해야 오류가 나지 않는다.
    user_question: str  # 사용자의 질문
    user_question_eval: str  # 사용자의 질문이 SQL 관련 질문인지 여부
    final_answer: str
    # TODO
    # context_cnt가 동적으로 조절 되도록 알고리즘을 짜야 한다.
    context_cnt: int  # 사용자의 질문에 대답하기 위해서 정보를 가져올 context 갯수
    table_contexts: List[str]
    table_contexts_ids: List[int]


########################### 정의된 노드 ###########################
def question_evaluation(state: GraphState) -> GraphState:
    """사용자의 질문을 평가하는 작업을 진행하는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 평가한 결과가 추가된 그래프 상태
    """
    user_question = state["user_question"]
    # 사용자 질문 평가
    user_question_eval = evaluate_user_question(user_question)

    return GraphState(user_question_eval=user_question_eval)  # type: ignore


def embodying_and_extracting(state: GraphState) -> GraphState:
    """사용자의 질문을 가지고 구체화하고 정보를 추출하는 작업을 진행하는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 구체화한 결과와 사용자의 질문에서 추출된 정보 추가된 그래프 상태
    """
    user_question = state["user_question"]
    # 사용자 질문을 구체화
    embodied_question = do_embodiment(user_question)
    # 사용자 질문에서 필요 정보 추출
    extracted_data = do_extraction(user_question)

    return GraphState(
        embodied_question=embodied_question, extracted_data=extracted_data
    )  # type: ignore


def non_sql_conversation(state: GraphState) -> GraphState:
    """일상적인 대화를 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문에 대한 대답이 추가된 그래프 상태
    """
    user_question = state["user_question"]
    final_answer = simple_conversation(user_question)

    return GraphState(final_answer=final_answer)  # type: ignore


################### ROUTERS ###################
def user_question_checker(state: GraphState) -> str:
    """그래프 상태에서 사용자의 질문 분류 결과를 가져오는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        str: 사용자의 질문 분류 결과 ("1" or "0")
    """
    return state["user_question_eval"]
