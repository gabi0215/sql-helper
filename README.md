# fastapi

## 파이썬 환경 설정
```
pip install -r requirements.txt
변동사항은 파일을 확인해주세요.
```

## 현재 .env 양식
```
OPENAI_API_KEY="your-api-key"
LANGCHAIN_API_KEY="your-api-key"
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT="your-project-name"
URL="DB 주소"
```

## 실행
```
# cli 명령어 입력 서버연결
uvicorn test_FastAPI:app --reload
```

```
fastapi 폴더내에 test_FastAPI.py로 실행됩니다.
```

```
openai 를 통한 쿼리문 생성 오류 해결.
mysql.connector 를 통한 연결오류.
```