from typing import TypedDict, List, Any
from .task import (
    evaluate_user_question,
    simple_conversation,
    select_relevant_tables,
    extract_context,
    create_query,
    analyze_user_question,
    refine_user_question,
    clarify_user_question,
    get_query_result,
    business_conversation,
)

# FAISS 객체는 serializable 하지 않아 Graph State에 넣어 놓을 수 없다.
from .faiss_init import get_vector_stores


# GrpahState 정의
class GraphState(TypedDict):
    # Warning!
    # 그래프 내에서 사용될 모든 key값을 정의해야 오류가 나지 않는다.
    user_question: str  # 사용자의 질문
    user_question_eval: str  # 사용자의 질문이 SQL 관련 질문인지 여부
    user_question_analyze: str  # 사용자 질문 분석
    final_answer: str
    # TODO
    # context_cnt가 동적으로 조절 되도록 알고리즘을 짜야 한다.
    context_cnt: int  # 사용자의 질문에 대답하기 위해서 정보를 가져올 context 갯수
    table_contexts: List[str]
    table_contexts_ids: List[int]
    need_clarification: bool  # 사용자 추가 질문(설명)이 필요한지 여부
    sql_query: str
    is_valid: bool
    max_query_fix: int
    query_fix_cnt: int
    query_result: List[Any]
    error_msg: str
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


def question_analyze(state: GraphState) -> GraphState:
    """질문 분석을 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 분석한 대답이 추가된 그래프 상태
    """
    user_question = state["user_question"]
    analyze_question = analyze_user_question(user_question)

    return GraphState(user_question_analyze=analyze_question)


def question_clarify(state: GraphState) -> GraphState:
    """사용자 질문이 모호할 경우 추가 질문을 통해 질문 분석을 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 분석한 대답이 추가된 그래프 상태
    """
    user_question_analyze = state["user_question_analyze"]
    user_question = state["user_question"]
    clarify_question = clarify_user_question(user_question, user_question_analyze)

    return GraphState(user_question_analyze=clarify_question)


def question_refine(state: GraphState) -> GraphState:
    """질문 구체화를 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문에 대한 대답이 추가된 그래프 상태
    """
    user_question_analyze = state["user_question_analyze"]
    user_question = state["user_question"]
    refine_question = refine_user_question(user_question, user_question_analyze)

    return GraphState(user_question=refine_question)


def table_selection(state: GraphState) -> GraphState:
    """사용자의 질문과 연관된 테이블 메타 데이터를 검색하고 검수하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 검색된 context들과 검수를 통과한 context의 index list가 추가된 그래프 상태
    """
    user_qusetion = state["user_question"]
    context_cnt = state["context_cnt"]
    vector_store = get_vector_stores()["db_info"]  # table_ddl + 실제 데이터
    # vector_store = get_vector_stores()["table_info"]  # table_ddl
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
    user_qusetion = state["user_question"]
    table_contexts = state["table_contexts"]
    table_contexts_ids = state["table_contexts_ids"]

    query_fix_cnt = state.get("query_fix_cnt", 0)
    is_valid = state.get("is_valid", True)

    if is_valid:
        sql_query = create_query(user_qusetion, table_contexts, table_contexts_ids)

    else:
        prev_query = state["sql_query"]
        error_msg = state["error_msg"]
        sql_query = create_query(
            user_qusetion,
            table_contexts,
            table_contexts_ids,
            is_valid=is_valid,
            prev_query=prev_query,
            error_msg=error_msg,
        )

    return GraphState(
        sql_query=sql_query,
        query_fix_cnt=query_fix_cnt + 1,
        is_valid=is_valid,
    )  # type: ignore


def query_validation(state: GraphState) -> GraphState:
    sql_query = state["sql_query"]
    try:
        query_result = get_query_result(command=sql_query, fetch="all")
        return GraphState(query_result=query_result)  # type: ignore

    except Exception as e:
        return GraphState(is_valid=False, error_msg=e)  # type: ignore


def sql_conversation(state: GraphState) -> GraphState:
    user_question = state["user_question"]
    sql_query = state["sql_query"]
    query_result = state["query_result"]
    final_answer = business_conversation(
        user_question, sql_query=sql_query, query_result=query_result
    )

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


def user_question_analyze_checker(state: GraphState) -> str:
    user_question_analyze = state["user_question_analyze"]
    analyze_question = analyze_user_question(user_question_analyze)

    keywords = ["[불명확]", "[확인필요]", "[에러]"]

    if any(keyword in analyze_question for keyword in keywords):
        state["need_clarification"] = True
    else:
        state["need_clarification"] = False

    return state["need_clarification"]


def query_checker(state: GraphState) -> str:
    is_valid = state["is_valid"]
    max_query_fix = state["max_query_fix"]
    query_fix_cnt = state["query_fix_cnt"]

    # 정해진 최대 재생성 횟수를 넘어서면 pass
    if is_valid or query_fix_cnt >= max_query_fix:
        return "KEEP"
    else:
        return "REGENERATE"
