python -m langgraph_.langgraph_main\
    --recursion-limit 50\
    --thread-id "TEST_RUN"\
    --model-name "gpt-4o-mini"\
    --user-question "자동 처리된 환불의 평균 금액과 수동 처리된 환불의 평균 금액은 얼마나 차이나나요?"\
    --max-query-fix 2\
    --sample-info 5
