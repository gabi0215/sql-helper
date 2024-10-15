import os
import json
from dotenv import load_dotenv

from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# 새로운 방식
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

def few_shot():
    # .env 파일에서 환경 변수 로드
    load_dotenv()
    api_key = os.getenv("OPENAPI_API_KEY")

    # JSON 파일 경로
    file_path = './few_shots.json'
    
    # JSON 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
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
    host = '34.173.158.39'
    port = '3306'
    username = 'Ssac'
    password = 'test1234'
    database_schema = 'classicmodels'
    mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}"
    
    # SQLite 데이터베이스 연결
    db = SQLDatabase.from_uri(mysql_uri)
    
    # OpenAI의 Chat 모델을 사용하여 LLM 정의
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

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

# if __name__ == "__main__":
#     agent = few_shot()
#     agent.run("직원이 몇 명이야?")