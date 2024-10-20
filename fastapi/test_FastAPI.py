from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Request
from fastapi import Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import mysql.connector
import openai
import os
from dotenv import load_dotenv
from test_main import get_config

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
url = os.getenv("URL")
client = openai.Client(api_key=openai_api_key)
templates = Jinja2Templates(directory="templates")

# FastAPI 앱 생성
app = FastAPI()


# 요청 모델 정의
class QueryRequest(BaseModel):
    query: str


# 자연어 -> SQL 변환 함수
def generate_sql(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that converts natural language to SQL queries.",
                },
                {"role": "user", "content": f"'{query}'에 대한 SQL 쿼리를 작성해줘."},
            ],
            max_tokens=500,
        )
        print(response)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API오류: {str(e)}")
        raise HTTPException(status_code=500, datail="Error generating SQL from OpenAI")


# MySQL 쿼리 실행 함수
def execute_sql(sql_query: str):
    try:
        conn = mysql.connector.connect(get_config())
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        return result
    except mysql.connector.Error as e:
        print(f"MySQL 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="Error executing SQL query")
    except Exception as e:
        print(f"일반 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="General error occurred")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# POST 요청을 처리하는 엔드포인트
# @app.post("/query")
# async def query(request: QueryRequest):
#     natural_query = request.query

#     if not natural_query:
#         raise HTTPException(status_code=400, detail="No query provided")

#     # 자연어 쿼리를 SQL로 변환
#     sql_query = generate_sql(natural_query)

#     # 변환된 SQL을 MySQL에 실행
#     db_result = execute_sql(sql_query)

#     return {
#         "natural_query": natural_query,
#         "sql_query": sql_query,
#         "db_result": db_result
#     }


# POST 요청을 처리하는 엔드포인트
@app.post("/query")
async def query(query: str = Form(...)):
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    # 자연어 쿼리를 SQL로 변환
    sql_query = generate_sql(query)
    result_list = sql_query.split(r"```")
    start_index = result_list[1].find("SELECT")
    end_index = result_list[1].find(";")

    # 변환된 SQL을 MySQL에 실행
    # db_result = execute_sql(result_list)

    # return {"natural_query": query, "sql_query": sql_query, "db_result": db_result}
    return result_list[1][start_index : end_index + 1]


# 기본 경로 처리
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


# 쿼리를 입력하는 HTML구성 창으로 접속 처리
@app.get("/query_form")
async def query_form(request: Request):
    return templates.TemplateResponse("query_form.html", {"request": request})
