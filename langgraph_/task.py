from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore

from typing import List
from pydantic import BaseModel, Field


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


def analyze_user_question(user_question: str) -> str:
    output_parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 사용자의 질문을 분석하여 데이터베이스 쿼리에 필요한 요소들을 파악하는 전문가입니다.
                    다음과 같은 요소들을 체계적으로 분석해주세요:
                    - 질문의 핵심 의도
                    - 필요한 데이터 항목
                    - 시간적 범위
                    - 데이터 필터링 조건
                    - 데이터 정렬 및 그룹화 요구사항
                    
                    분석 과정에서 발견된 문제점을 다음 태그를 사용하여 설명해주세요:
                    - [불명확] - 질문이 모호하거나 여러 해석이 가능한 경우
                    예시: "최근", "적절한" 같이 기준이 불분명한 표현
                    - [확인필요] - 추가 정보나 맥락이 필요한 경우
                    예시: 특정 용어의 정의나 기준이 명시되지 않은 경우
                    - [에러] - 논리적 모순이 발생한 경우""",
            ),
            (
                "human",
                """사용자 질문: {user_question}
                    
                    아래 형식으로 상세하게 분석해주세요:
                    - 주요 의도: 사용자가 질문의 핵심적으로 묻고자 하는 바는 무엇인가요?
                    - 필요한 데이터 항목: 사용자가 원하는 구체적인 데이터나 정보는 무엇인가요?
                    - 시간적 범위: 사용자가 원하는 시간적 범위는 어떻게 되나요?
                    - 데이터 필터링 조건: 사용자가 요청한 데이터에는 어떤 조건이나 필터링이 필요한가요?
                    - 데이터 정렬 및 그룹화: 사용자가 원하는 데이터의 정렬 기준이나 그룹화 방식은 무엇인가요?
                    
                    분석 과정에서 발견한 문제점을 아래에 태그와 함께 설명해주세요:
                    - [불명확]:
                    - [확인필요]:
                    - [에러]:""",
            ),
        ]
    )

    analyze_chain = ANALYZE_PROMPT | llm | output_parser
    analyze_question = analyze_chain.invoke({"user_question": user_question})

    return analyze_question


def clarify_user_question(user_question: str, user_question_analyze: str) -> str:
    output_parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    user_add_questions = []
    final_analysis = user_question_analyze
    CLARIFY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 데이터 분석가로서 [불명확], [확인필요], [에러] 태그 뒤에 명시된 문제를 파악하고 해결하기 위한 적절한 추가 질문을 합니다. 이를 통해 문제가 해결됬는지 판단하세요. 
                
                추가 질문 조건:
                1. 태그([불명확], [확인필요], [에러])로 표시된 문제만 질문하고 이외 추가적인 질문은 절대 금지
                2. 한 번에 하나의 질문만 제시
                3. 객관식 질문으로 구성
                4. 구체적이고 이해하기 쉬운 질문으로 작성
                5. 이전 질문 기록을 검토하여 중복 질문 방지
                
                종료 조건:
                1. [불명확], [확인필요], [에러] 태그에 해당하는 모든 문제가 해결된 경우
                2. 사용자가 더 이상의 추가 정보 제공을 원하지 않는 경우
                3. 이전 질문 기록에 중복된 질문이 반복될 경우

                응답 형식:
                1. 추가 질문이 필요한 경우:
                "질문:\n1) 옵션 1\n2) 옵션 2\n3) 옵션 3\n4) 직접 입력\n"
                
                2. 종료 조건을 만족한 경우:
                "종료\n최종분석:\n주요 의도:\n필요한 데이터 항목:\n시간적 범위:\n데이터 필터링 조건:\n데이터 정령 및 그룹화 요구사항:"
                """,
            ),
            (
                "human",
                """원래 사용자 질문:
                {user_question}
                
                초기 질문 분석:
                {user_question_analyze}
                
                이전 질문 기록:
                {collected_questions}
                
                지시사항:
                태그([불명확], [확인필요], [에러])가 표시된 모든 문제가 해결되면 분석을 종료하세요.
                """,
            ),
        ]
    )

    while True:
        clarify_chain = CLARIFY_PROMPT | llm | output_parser
        collected_questions = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(user_add_questions)
        )

        clarify_question = clarify_chain.invoke(
            {
                "user_question": user_question,
                "user_question_analyze": user_question_analyze,
                "collected_questions": collected_questions,
            }
        )

        if clarify_question.startswith("종료"):
            final_analysis = clarify_question.split("최종분석:")[1].strip()
            print(f"\n{final_analysis}")
            break

        print(f"\n{clarify_question}", flush=True)
        user_answer = input("답변을 입력하세요 (종료하려면 'q' 또는 '종료'): ").strip()
        print(f"사용자 답변: {user_answer}")

        if user_answer.lower() in ["q", "종료"]:
            break

        user_add_questions.append(
            f"\n질문: \n{clarify_question}\n답변: {user_answer}\n"
        )

    return final_analysis


def refine_user_question(user_question: str, user_question_analyze: str) -> str:
    output_parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    REFINE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 사용자의 질문을 더 명확하고 구체적으로 다듬는 전문가입니다. 주어진 질문과 분석을 바탕으로 더 구체적인 질문으로 재구성하세요.""",
            ),
            (
                "human",
                """사용자 질문: 
                {user_question}
                
                사용자 질문 분석: 
                {question_analyze}
                
                구체화된 질문:""",
            ),
        ]
    )

    refine_chain = REFINE_PROMPT | llm | output_parser
    refine_question = refine_chain.invoke(
        {"user_question": user_question, "question_analyze": user_question_analyze}
    )

    return refine_question


def select_relevant_tables(
    user_question: str, context_cnt: int, vector_store: VectorStore
) -> List[str]:
    """user_question과 관련성이 가장 높은 k(context_cnt)개의 document에서 page_content만 추출하여 리스트의 형태로 반환하는 함수입니다.
    입력으로 들어오는 vector_store는 반드시 MySQL 서버 내 테이블에 대한 메타데이터가 임베딩 되어 있어야 정상적으로 작동 합니다.
    관련성은 vetor store에 기본으로 내장된 유사도 검색 알고리즘을 사용합니다.

    Args:
        user_question (str): 사용자의 질문
        context_cnt (int): 반환할 context의 개수
        vector_store (VectorStore): MySQL 서버 내 테이블에 대한 메타데이터가 임베딩 되어있는 벡터 스토어

    Returns:
        List[str]: context가 포함된 리스트
    """
    relevant_tables = vector_store.similarity_search(user_question, k=context_cnt)
    table_contexts = [doc.page_content for doc in relevant_tables]

    return table_contexts


def extract_context(user_question: str, table_contexts: List[str]) -> List[int]:
    """벡터 스토어에서 검색으로 얻어낸 context들을 대상으로 사용자의 질문(user_question)에 기반한 SQL문을 생성함에 있어 필요한지를 판단한 후,
    필요한 context의 인덱스를 담은 리스트를 반환하는 함수입니다.

    Args:
        user_question (str): 사용자의 질문
        table_contexts (List[str]): context들을 담은 리스트

    Returns:
        List[int]: 필요한 context의 index를 담은 리스트
    """
    if not table_contexts:
        # 평가할 context가 없다면 빈 리스트 반환
        return []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 사용자의 질문을 SQL로 변환하는 데 필요한 정보를 식별하는 전문가입니다.
                
                당신의 임무:
                1. 주어진 사용자 질문을 분석하여 SQL 생성에 필요한 정보들을 식별
                2. 필요한 정보의 인덱스를 정수 리스트로 반환

                입력 형식:
                - 사용자 질문: SQL로 변환이 필요한 사용자의 질문
                - context: 사용자 질문을 SQL문으로 변환할 때 사용할 정보들

                반환 형식:
                - 필요한 context의 인덱스를 담은 정수 리스트
                - 필요한 context가 없는 경우 빈 리스트

                주의사항:
                - 인덱스는 제시된 순서대로 정렬하여 반환합니다
                - SQL 생성에 확실히 필요한 정보만 선택합니다""",
            ),
            (
                "human",
                """user_question:
                {user_question}
                
                context:
                {context}""",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    class context_list(BaseModel):
        """사용자 질문에 답하기 위해 필요한 맥락의 인덱스 목록"""

        ids: List[int | None] = Field(description="Ids of contexts.")

    structured_llm = llm.with_structured_output(context_list)
    context = ""
    for idx, table_info in enumerate(table_contexts):
        context += f"{idx}.\n{table_info}\n\n"

    chain = prompt | structured_llm

    output = chain.invoke({"user_question": user_question, "context": context})
    return output.ids  # type: ignore


def create_query(
    user_question: str, table_contexts: List[str], table_contexts_ids: List[int]
) -> str:
    """사용자의 질문(user_question)을 query로, 테이블의 메타데이터(table_contexts)를 context로 하여 답변을 생성하는 함수입니다.
    필요한 context의 index를 담은 table_contexts_ids를 이용해 필요한 context만 뽑아서 프롬프트에 넣습니다.

    Args:
        user_question (str): 사용자의 질문
        table_contexts (List[str]): context들을 담은 리스트
        table_contexts_ids (List[int]): 필요한 context의 index를 담은 리스트

    Returns:
        str: 모델이 생성한 SQL문
    """

    output_parser = StrOutputParser()
    # TODO
    # 동일한 함수에서 저장된 prompt의 경로만 교체해서 효율성을 높이는 것을 고려해보아야 한다
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자의 입력을 SQL문으로 바꾸어주는 조직의 팀원입니다. 당신의 임무는 주어진 질문(user_question)과 DB 이름 그리고 DB내 테이블의 메타 정보가 담긴(context)를 이용해서 주어진 질문에 걸맞는 SQL 쿼리문을 작성하는 것입니다. SQL 쿼리문 작성시 모든 테이블의 DB를 db.table 처럼 표시해주세요",
            ),
            (
                "human",
                """user_question: {user_question}
                context: {context}""",
            ),
        ]
    )

    context = ""
    for idx, table_info in enumerate(table_contexts):
        if idx in set(table_contexts_ids):
            context += table_info + "\n\n"

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question, "context": context})
    return output


##################### Experimental #####################


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
