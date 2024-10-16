import os
import json
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentType
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

import ast
import re


def few_shot(config):
    # .env 파일에서 환경 변수 로드
    load_dotenv()

    # JSON 파일 읽기
    with open(config.few_shot_path, "r", encoding="utf-8") as file:
        few_shots = json.load(file)

    # OpenAI 임베딩 방법을 사용하여 임베딩 정의, 비용 절감 가능한 부분(Xenova/text-embedding-ada-002 같은 허깅페이스 임베딩 모델을 사용하면 되지 않을까?)
    embeddings = OpenAIEmbeddings()

    # few_shots의 각 질문에 대해 Document 객체 생성
    few_shot_docs = [
        Document(page_content=question, metadata={"sql_query": few_shots[question]})
        for question in few_shots.keys()
    ]

    # Document에서 FAISS 벡터 데이터베이스 생성
    vector_db = FAISS.from_documents(few_shot_docs, embeddings)

    # FAISS 벡터 데이터베이스를 기반으로 검색기 생성
    retriever = vector_db.as_retriever()

    # tool 설명 정의
    tool_description = """
    이 도구는 유사한 예시를 이해하여 사용자 질문에 적용하는 데 도움이 됩니다.
    이 도구에 입력하는 내용은 사용자 질문이어야 합니다.
    """

    # 검색 도구 생성
    retriever_tool = create_retriever_tool(
        retriever, name="sql_get_similar_examples", description=tool_description
    )

    # 커스텀 도구 리스트에 검색 도구 추가
    custom_tool_list = [retriever_tool]

    # DB 연결 시 사용할 변수 정의
    mysql_uri = f"mysql+pymysql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database_schema}"

    # SQLite 데이터베이스 연결
    db = SQLDatabase.from_uri(mysql_uri)

    # OpenAI의 Chat 모델을 사용하여 LLM 정의
    llm = ChatOpenAI(
        model_name=config.model_name,
        temperature=0,
    )

    # SQL 데이터베이스 도구 키트 생성
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # 커스텀 서픽스 정의
    custom_suffix = """
    먼저 제가 알고 있는 비슷한 예제를 가져와야 합니다.
    예제가 쿼리를 구성하기에 충분하다면 쿼리를 작성할 수 있습니다.
    그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
    그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
    """

    # SQL 에이전트 생성
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=custom_tool_list,
        suffix=custom_suffix,
    )

    # 생성된 에이전트 반환
    return agent


def cardinaliy(config):
    # 쿼리 결과를 리스트로 변환하는 함수 정의
    def query_as_list(db, query):
        # 쿼리를 실행하고 결과를 저장
        res = db.run(query)
        # 텍스트 전처리
        res = [
            el for sub in ast.literal_eval(res) for el in sub if el
        ]  # 쿼리 결과를 리스트 형태로 변환하고, 중첩된 리스트를 풀어줌
        res = [
            re.sub(r"\b\d+\b", "", string).strip() for string in res
        ]  # 숫자를 제거하고 공백을 제거한 문자열 리스트로 변환
        # 리스트 반환
        return res

    # .env 파일에서 환경 변수 로드
    load_dotenv()

    # DB 연결 시 사용할 변수 정의
    mysql_uri = f"mysql+pymysql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database_schema}"

    # MySQL 데이터베이스 연결 설정
    db = SQLDatabase.from_uri(mysql_uri)

    # "employees" 테이블에서 성(lastName) 목록 가져오기
    artists = query_as_list(
        db, "SELECT DISTINCT lastName FROM employees"
    )  # 중복을 제거하여 프롬프트에 들어가는 쿼리 줄임
    # "employees" 테이블에서 이름(firstName) 목록 가져오기
    albums = query_as_list(db, "SELECT DISTINCT firstName FROM employees")

    # 이름과 성 목록을 합침
    texts = artists + albums

    # 텍스트 데이터를 벡터로 변환하기 위한 임베딩 생성
    embeddings = OpenAIEmbeddings()
    # 임베딩을 사용하여 벡터 데이터베이스 생성
    vector_db = FAISS.from_texts(texts, embeddings)
    # 벡터 검색기를 생성
    retriever = vector_db.as_retriever()

    # 검색기를 사용해 사용자 요청에 따라 이름 검색 도구 생성
    retriever_tool = create_retriever_tool(
        retriever,
        name="name_search",  # tool 이름, 모델에 해당 도구 사용요청시 사용
        description="이름, 성 데이터가 실제로 어떻게 쓰여졌는지 알아내는 데 사용합니다.",
    )

    # 사용자 정의 도구 목록 생성
    custom_tool_list = [retriever_tool]

    # LLM 설정
    llm = ChatOpenAI(
        model_name=config.model_name,
        temperature=0,
    )

    # SQL 데이터베이스 도구 세트 생성
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # 에이전트의 동작 방식을 정의하는 추가 지침 정의
    custom_suffix = """
    사용자가 고유명사를 기준으로 필터링해 달라고 요청하는 경우, 먼저 name_search 도구를 사용하여 철자를 확인해야 합니다.
    그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
    그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
    """

    # SQL 에이전트 생성
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=custom_tool_list,
        suffix=custom_suffix,
    )

    # 생성된 에이전트 반환
    return agent
