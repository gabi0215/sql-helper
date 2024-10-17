from typing import TypedDict
import argparse

from node import GraphState
from graph import make_graph
from utils import get_runnable_config


if __name__ == "__main__":
    # config 설정
    config = get_runnable_config(recursion_limit=100, thread_id="TEST_RUN")
    # graph 생성
    graph = make_graph()

    # 입력을 위해 그래프 상태 만들기
    inputs = GraphState(user_question="서울시의 2024 5월 부터 2024 10월까지의 강수량 평균을 알려줘")  # type: ignore
    outputs = graph.invoke(input=inputs, config=config)

    print(outputs)
