import argparse
from sql_rag import few_shot, cardinaliy


def get_config():
    # parser 생성
    parser = argparse.ArgumentParser(
        prog="Get Config",
    )

    parser.add_argument("--model-name", default="gpt-4o-mini", type=str)

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    config = get_config()
    agent = cardinaliy(config)
    agent.invoke("다이안의 이메일을 알려줘")
    # agent = few_shot()
    # agent.invoke("직원이 몇 명이야?")
