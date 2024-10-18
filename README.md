# sql-helper

## 파이썬 환경 설정
```
pip install -r requirements.txt
```

## 임시로 사용할 chinook.db 다운 받아서 압축 해제
```
wget https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip -O chinook.zip
sudo apt install unzip
unzip chinook.zip
```

## 현재 .env 양식
```
OPENAI_API_KEY="your-api-key"
LANGCHAIN_API_KEY="your-api-key"
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT="your-project-name"
URL="sqlite:///chinook.db"
```

## 실행
```
# RAG
python main.py
```

```
# Streamit
streamlit run streamlit_main.py
```