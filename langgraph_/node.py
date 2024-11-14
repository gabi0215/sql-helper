from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS

from typing import TypedDict, List, Dict
from .task import (
    evaluate_user_question,
    simple_conversation,
    select_relevant_tables,
    extract_context,
    create_query,
)

# FAISS 객체는 serializable 하지 않아 Graph State에 넣어 놓을 수 없다.
from .faiss_init import get_vector_stores


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
    # TODO
    # 지금은 FAISS 벡터 DB를 쓰기에 아래와 같이 딕셔너리에 넣어놓지만, Redis DB 서버를 만들어서 이용할 경우에는 index가 들어가야 한다.
    # FAISS 객체는 serializable 하지 않아 Graph State에 넣어 놓을 수 없다. 노드 안에서 객체를 불러오는 것으로 한다.
    # vector_store_dict: Dict[str, VectorStore | FAISS]  # RAG를 위한 벡터 DB


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


def table_selection(state: GraphState) -> GraphState:
    """사용자의 질문과 연관된 테이블 메타 데이터를 검색하고 검수하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 검색된 context들과 검수를 통과한 context의 index list가 추가된 그래프 상태
    """
    user_qusetion = state["user_question"]
    context_cnt = state["context_cnt"]
    vector_store = get_vector_stores()["table_info"]  # table_ddl
    # FAISS 객체는 serializable 하지 않아 Graph State에 넣어 놓을 수 없다.
    # vector_store = state["vector_store_dict"]["table_info"] # table_ddl
    # 사용자 질문과 관련성이 있는 테이블+컬럼정보를 검색
    table_contexts = select_relevant_tables(
        user_question=user_qusetion, context_cnt=context_cnt, vector_store=vector_store
    )
    # 검색된 context를 검수
    table_contexts_ids = extract_context(
        user_question=user_qusetion, table_contexts=table_contexts
    )
    return GraphState(
        table_contexts=table_contexts,
        table_contexts_ids=table_contexts_ids,
    )  # type: ignore


def query_creation(state: GraphState) -> GraphState:
    """사용자 질문을 포함한 여러 정보들을 가지고 SQL 쿼리문을 생성하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문에 대한 SQL 쿼리문이 추가된 그래프 상태
    """
    user_qusetion = state["user_question"]
    table_contexts = state["table_contexts"]
    table_contexts_ids = state["table_contexts_ids"]

    sql_result = create_query(user_qusetion, table_contexts, table_contexts_ids)

    # TODO
    # 현재 SQL 쿼리문만을 출력하는 것이 중간 목표이기에 final_answer 필드에 저장했지만,
    # 향후에는 SQL 쿼리문을 저장하는 필드에 저장한 뒤, flow를 이어나가야 한다.
    return GraphState(
        final_answer=sql_result,
    )  # type: ignore


################### ROUTERS ###################
def user_question_checker(state: GraphState) -> str:
    """그래프 상태에서 사용자의 질문 분류 결과를 가져오는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        str: 사용자의 질문 분류 결과 ("1" or "0")
    """
    return state["user_question_eval"]
