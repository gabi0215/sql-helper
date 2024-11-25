from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Request
from fastapi import Form
from fastapi.templating import Jinja2Templates
import openai
import os, re
from dotenv import load_dotenv
import pymysql

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
db_url = os.getenv("URL")


username = os.getenv("username")
password = os.getenv("password")
hostname = os.getenv("hostname")
port = int(os.getenv("port"))
database = os.getenv("database")

# print(url.username,url.password,url.hostname,url.port,url.path[1:])
# # 파싱된 정보 기반으로 딕셔너리 생성
db_config = {
    "user": username,
    "password": password,
    "host": hostname,
    "port": port,
    "database": database,  # 출력시 '/' 맨 앞 부분 제거
}
print(db_config)
client = openai.Client(api_key=openai_api_key)
templates = Jinja2Templates(directory="templates")

# FastAPI 앱 생성
app = FastAPI()


# 자연어 -> SQL 변환 함수
def generate_sql(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that converts natural language to SQL queries. You have to answer with SQL query ONLY.",
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


# pymySQL 쿼리 실행 함수
def execute_sql(sql_query: str):
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(**db_config)

        # cursor db에서 쿼리 실행 및 결과 가져오기 위한 객체입니다.
        cursor = conn.cursor()

        # SQL 쿼리를 실행합니다.
        cursor.execute(sql_query)

        # fecthall: 쿼리 결과의 모든 행을 리스트로 가져옵니다.
        result = cursor.fetchall()

        return result
    except pymysql.MySQLError as e:
        print(f"MySQL 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="Error executing SQL query")
    except Exception as e:
        print(f"일반 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="General error occurred")
    finally:
        # conn이 None이 아닐 경우에만 .close() 호출
        if conn:
            if cursor:  # cursor가 None이 아니면 닫기
                cursor.close()
            conn.close()  # conn이 None이 아니면 닫기


# POST 요청을 처리하는 엔드포인트
@app.post("/query")
async def query(query: str = Form(...)):
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    # 자연어 쿼리를 SQL로 변환
    sql_query = generate_sql(query)
    print(sql_query)
    pattern = r"(?<=\n).*(?=;)"
    result = re.search(pattern, sql_query).group()

    # 변환된 SQL을 MySQL에 실행
    print(result)
    db_result = execute_sql(result)

    return {"natural_query": query, "sql_query": sql_query, "db_result": db_result}

# 기본 경로 처리
@app.get("/")
async def read_root():
    return {"message": "Welcome to the sql-helper application!"}


# 쿼리를 입력하는 HTML구성 창으로 접속 처리
@app.get("/query_form")
async def query_form(request: Request):
    return templates.TemplateResponse("query_form.html", {"request": request})
