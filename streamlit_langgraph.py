import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph_.langgraph_main import text2sql


load_dotenv()


# GPT 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 세션 상태 초기화
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI
st.title("GPT로 SQL 쿼리 생성기")
user_input = st.text_input("데이터에 대해 질문하세요요:")


if st.button("쿼리 생성 및 실행"):
    response = text2sql(user_input)

    # response의 전체 내용 확인 (디버깅 목적)
    st.write("Response 전체 내용:")
    st.write(response)

    # 대화 기록에 추가
    st.session_state.conversation_history.append(
        {"question": user_input, "response": response}
    )

# 대화 기록 표시
st.subheader("대화 기록")
for entry in st.session_state.conversation_history:
    st.write(f"질문: {entry['question']}")
    st.write(f"Response 전체 내용: {entry['response']}")
