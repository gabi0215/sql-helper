import streamlit as st
import requests

# API 엔드포인트 설정
API_URL = "http://localhost:8000/llm_workflow"

# 세션 상태 초기화
def initialize_session_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "1"
    if "initial_question" not in st.session_state:
        st.session_state.initial_question = 1
    if "snapshot_values" not in st.session_state:
        st.session_state.snapshot_values = None

# 메인 앱 함수
def main():
    st.title("AI SQL 쿼리 생성기")
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 대화 기록 표시
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])
    
    # 사용자 입력 받기
    if prompt := st.chat_input("데이터에 대해 질문하세요. (더 이상 질문이 없으면 '해당 사항 없음' 입력하세요):"):
        # 사용자 메시지 표시 및 대화 기록에 추가
        st.chat_message("user").write(prompt)
        st.session_state.conversation_history.append({
            'role': 'user', 
            'content': prompt
        })
        
        try:
            # API 호출
            response = requests.post(
                API_URL,
                json={
                    "user_question": prompt,
                    "initial_question": st.session_state.initial_question,
                    "thread_id": st.session_state.thread_id,
                    "last_snapshot_values": st.session_state.snapshot_values,
                },
            )
            
            if response.status_code == 200:
                processed_info = response.json()
                st.session_state.snapshot_values = processed_info
                ask_user = processed_info["ask_user"]
                st.session_state.initial_question = 0
                
                # 응답 결정
                if ask_user == 0:
                    st.session_state.initial_question = 1
                    output = processed_info["final_answer"]
                else:
                    output = processed_info["collected_questions"][-1]
                
                # 어시스턴트 메시지 표시 및 대화 기록에 추가
                st.chat_message("assistant").write(output)
                st.session_state.conversation_history.append({
                    'role': 'assistant', 
                    'content': output
                })
            else:
                error_msg = "서버 처리 중 오류가 발생했습니다."
                st.error(error_msg)
                st.chat_message("assistant").write(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"서버 연결 오류: {str(e)}"
            st.error(error_msg)
            st.chat_message("assistant").write(error_msg)

# 앱 실행
if __name__ == "__main__":
    main()