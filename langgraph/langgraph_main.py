from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import pprint
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig


# GrpahState 정의
class GraphState(TypedDict):
    # Warning!
    # 그래프 내에서 사용될 모든 key값을 정의해야 오류가 나지 않는다.
    user_question: str  # 사용자의 질문
    user_question_eval: str  # 사용자 질문 평가
    embodied_question: str  # 사용자의 구체화 된 질문
    extracted_data: list[str]  # 사용자의 질문에서 추출된 정보
    category: str  # 사용자의 질문에 대해 대응해야하는 카테고리
    final_answer: str  # 서비스 흐름에서 사용자에게 전달될 최종 답변


# Node와 Edge 정의
# 세부 task
def evaluate_user_question(user_question: str) -> str:
    """사용자의 질문이 일상적인 대화문인지, 데이터 및 비즈니스와 관련된 질문인지를
    판단하는 역할을 담당하고 있는 함수입니다.
    현재는 LLM(gpt-4o-mini)을 통해 사용자의 질문을 판단하고 있습니다.

    Args:
        user_question (str): 사용자의 질문

    Returns:
        str: "1" : 데이터 또는 비즈니스와 관련된 질문, "0" : 일상적인 대화문
    """
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자의 입력을 SQL문으로 바꾸어주는 조직의 팀원입니다. 당신의 임무는 주어진 질문(user_question)이 데이터 또는 비즈니스와 관련된 일인지를 판단하는 것 입니다.",
            ),
            (
                "human",
                "주어진 질문(user_question)이 데이터 또는 비즈니스와 관련되어 있으면 1, 아니면 0을 출력하세요. `1`과 같이 결과만 작성하세요.\n\n#질문(user_question): {user_question}",
            ),
        ]
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini")  # type: ignore
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output


def simple_conversation(user_question: str) -> str:
    """사용자의 질문이 일상적인 대화문이라고 판단되었을 경우
    사용자와 일상적인 대화를 진행하는 함수입니다.
    현재는 LLM(gpt-4o-mini)으로 대응을 진행하고 있습니다.

    Args:
        user_question (str): 사용자의 일상적인 질문

    Returns:
        str: 사용자의 일상적인 질문에 대한 AI의 대답
    """
    output_parser = StrOutputParser()
    # TODO
    # 동일한 함수에서 저장된 prompt의 경로만 교체해서 효율성을 높이는 것을 고려해보아야 한다
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 SQL 헬퍼 입니다.",
            ),
            (
                "human",
                "{user_question}",
            ),
        ]
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini")  # type: ignore
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output


def do_embodiment(user_question: str) -> str:
    """데이터 및 비즈니스 관련 질문으로 판단된 사용자의 질문을 구체화하는 함수입니다.
    구체화를 어떤 식으로 진행할지는 아직 정해지지 않았으니, 일단은 사용자의 질문을 2배로 늘린 결과를 return 합니다.

    Args:
        user_question (str): 데이터 및 비즈니스 관련 사용자의 질문

    Returns:
        str: 사용자의 질문을 구체화한 결과
    """

    return user_question * 2


def do_extraction(user_question: str) -> list:
    """데이터 및 비즈니스 관련 질문으로 판단된 사용자의 질문에서 필요한 정보들을 추출하는 함수입니다.
    정보 추출을 어떤 식으로 진행할지는 아직 정해지지 않았으니, 일단은 사용자의 질문을 띄어쓰기 기준으로 split 한 결과를 return 합니다.

    Args:
        user_question (str): 데이터 및 비즈니스 관련 사용자의 질문

    Returns:
        list: 사용자의 질문에서 추출된 정보
    """
    return user_question.split()


def do_category_classification(
    user_question: str, embodied_question: str, extracted_data: list[str]
) -> str:
    """사용자의 질문에 대해서 우리가 정의한 기능 중 어떤 기능으로 대응해야 하는지 분류해주는 함수입니다.
    사용할 모델 또는 api 에 따라서 입력에 필요한 데이터가 달라질 수 있으며, 일단은 모두 입력 받아오도록 만들었습니다.
    아직 데이터 및 라벨에 대한 정의가 존재하지 않으니 임의로 생성하여 랜덤한 값을 return 합니다.

    Args:
        user_question (str): 데이터 및 비즈니스 관련 사용자의 질문
        embodied_question (str): 사용자의 질문을 구체화한 결과
        extracted_data (list[str]): 사용자의 질문에서 추출된 정보

    Returns:
        str: 기능 분류 결과
    """

    from random import choice

    labels = [
        "쿼리문 생성",
        "쿼리문 해설",
        "테이블 해설",
        "쿼리문 문법 검증",
        "테이블 및 컬럼 활용 안내",
    ]
    return choice(labels)

    # 노드


def user_question_checker(state: GraphState) -> str:
    """그래프 상태에서 사용자의 질문 분류 결과를 가져오는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        str: 사용자의 질문 분류 결과 ("1" or "0")
    """
    return state["user_question_eval"]


def category_checker(state: GraphState) -> str:
    """사용자의 질문을 기능적으로 분류한 결과를 반환하는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        str: 기능 분류 결과
    """
    return state["category"]


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


def classify_question_category(state: GraphState) -> GraphState:
    """사용자의 질문 및 기타 정보(현재 버전: 구체화된 질문, 추출된 정보)를 가지고
    사용자의 질문을 기능적으로 분류하는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 기능적으로 분류한 결과가 추가된 그래프 상태
    """
    user_question = state["user_question"]
    embodied_question = state["embodied_question"]
    extracted_data = state["extracted_data"]

    # 사용자 질문과 구체화 정보,필요 정보들을 고려하여 분류
    category = do_category_classification(
        user_question, embodied_question, extracted_data
    )

    return GraphState(category=category)  # type: ignore


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


def crete_query(state: GraphState) -> GraphState:
    """쿼리문 생성 기능의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def explain_query(state: GraphState) -> GraphState:
    """쿼리문 설명 기능의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def explain_table(state: GraphState) -> GraphState:
    """테이블 설명 기능의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def correct_query_grammar(state: GraphState) -> GraphState:
    """쿼리문 문법 검증의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def guide_utilization(state: GraphState) -> GraphState:
    """테이블 및 컬럼 활용 안내 기능의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore

    # Edges


workflow = StateGraph(GraphState)

workflow.add_node("질문 평가", question_evaluation)
workflow.add_node("일반적인 대화", non_sql_conversation)
workflow.add_node("질문 구체화 및 정보 추출", embodying_and_extracting)

workflow.add_node("질문 카테고리 분류", classify_question_category)
workflow.add_node("crete_query", crete_query)
workflow.add_node("explain_query", explain_query)
workflow.add_node("explain_table", explain_table)
workflow.add_node("correct_query_grammar", correct_query_grammar)
workflow.add_node("guide_utilization", guide_utilization)


workflow.add_conditional_edges(
    "질문 평가",  # 사용자의 질문이 데이터 및 비즈니스와 관련 되어있는지 평가
    user_question_checker,  # check로 평가 데이터를 추출한 뒤, 아래의 조건에 따라 다른 노드에 전달
    {
        "0": "일반적인 대화",
        "1": "질문 구체화 및 정보 추출",
    },
)

workflow.add_edge("질문 구체화 및 정보 추출", "질문 카테고리 분류")

workflow.add_conditional_edges(
    "질문 카테고리 분류",
    category_checker,
    {
        "쿼리문 생성": "crete_query",
        "쿼리문 해설": "explain_query",
        "테이블 해설": "explain_table",
        "쿼리문 문법 검증": "correct_query_grammar",
        "테이블 및 컬럼 활용 안내": "guide_utilization",
    },
)

# 시작점을 설정합니다.
workflow.set_entry_point("질문 평가")

# 기록을 위한 메모리 저장소를 설정합니다.
memory = MemorySaver()

# 그래프를 컴파일합니다.
app = workflow.compile(checkpointer=memory)


# set config
# RunnableConfig에 thread id로 추적 기록을 추적 가능하게 할 수 있습니다.
# recursion_limit은 최대 노드를 몇번 거치게 할 것인지에 대한 한계 값입니다.
config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "SQL-test"})

if __name__ == "__main__":
    inputs = GraphState(user_question="서울시의 2024 5월 부터 2024 10월까지의 강수량 평균을 알려줘")  # type: ignore
    outputs = app.invoke(input=inputs, config=config)
    print(outputs)
