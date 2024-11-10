# fastapi

## 파이썬 환경 설정
```
pip install -r requirements.txt
변동사항은 파일을 확인해주세요.
```

## 현재 .env 양식
```
OPENAI_API_KEY=""
LANGCHAIN_API_KEY="your-api-key"
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT="your-project-name"
URL=""
username=""
password=""
hostname=""
port=
database=""
```

## 실행
```
# cli 명령어 입력 서버연결
uvicorn test_FastAPI:app --reload
```

```
# fastapi 폴더내에 test_FastAPI.py로 실행됩니다.
실행 후 도메인에서 /query_form 페이지로 이동하여 질문을 입력하면 연결된 db에서 정보를 추출하여 시각화합니다.
```

```
openai 를 통해 생성된 sql문을 프롬프팅과 정규표현식을 통해서 llm에게 sql문만 전달되게끔 코드 수정.

mysql.connector 를 통한 연결오류 문제 부분은 pymysql.connect로 코드 수정하여 해결.
```