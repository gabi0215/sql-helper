import argparse


def get_config():
    # parser 생성
    parser = argparse.ArgumentParser(
        prog="Get Config",
    )
    parser.add_argument("--model-name", default="gpt-4o-mini", type=str)

    parser.add_argument("--host", default="34.173.158.39", type=str)

    parser.add_argument("--port", default="3306", type=str)

    parser.add_argument("--username", default="Ssac", type=str)

    parser.add_argument("--password", default="test1234", type=str)

    parser.add_argument("--database-schema", default="classicmodels", type=str)

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    config = get_config()

    # agent = few_shot()
    # agent.invoke("직원이 몇 명이야?")
