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
# test_main.py
MySQL 서버 주소 연결을 위한 get_config 불러오는 파일입니다.
```

```
실행은 fastapi 폴더내에 test_FastAPI.py로 실행됩니다.
현재 opneai 버전으로 인한 생성오류문제가 있어 수정이 필요합니다.
```