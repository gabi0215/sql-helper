python -m langgraph_.langgraph_main\
    --recursion-limit 50\
    --thread-id "TEST_RUN"\
    --model-name "gpt-4o-mini"\
    --user-question "전자기기와 액세서리 카테고리별 환불 건수와 금액은 어떻게 되나요?"\
    --max-query-fix 2\
    --sample-info 5
