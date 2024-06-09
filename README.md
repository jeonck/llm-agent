# llm-agent

```
# 질의-응답 테스트를 수행 (앨런 튜링의 기계 지능에 대한 내용을 PDF 문서에서 검색)
response = agent_with_chat_history.invoke(
    {
        "input": "앨런 튜링의 기계 지능에 대한 내용을 PDF 문서에서 알려줘"
    },
    # 세션 ID를 설정 (여기서는 간단한 메모리 내 ChatMessageHistory를 사용)
    config={"configurable": {"session_id": "MyTestSessionID"}},
)
print(f"답변: {response['output']}")

# 질의-응답 테스트를 수행
response = agent_with_chat_history.invoke(
    {
        "input": "판교 카카오 프렌즈샵 아지트점의 전화번호를 검색하여 결과를 알려주세요."
    },
    # 세션 ID를 설정 (여기서는 간단한 메모리 내 ChatMessageHistory를 사용)
    config={"configurable": {"session_id": "MyTestSessionID"}},
)
print(f"답변: {response['output']}")

```
