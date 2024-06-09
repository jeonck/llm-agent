from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from dotenv import load_dotenv

# 환경 변수 로드 (API KEY 등의 민감한 정보 관리)
load_dotenv()

import os

# TAVILY API KEY를 환경 변수로 설정 (https://app.tavily.com/sign-in에서 발급)
os.environ["TAVILY_API_KEY"]

# 디버깅을 위한 프로젝트명 설정
# os.environ["LANGCHAIN_PROJECT"] = "AGENT TUTORIAL"

# TavilySearchResults 클래스를 langchain_community.tools.tavily_search 모듈에서 가져옴
from langchain_community.tools.tavily_search import TavilySearchResults

# TavilySearchResults 클래스의 인스턴스를 생성 (검색 결과를 최대 5개까지 가져옴)
search = TavilySearchResults(k=5)

# 검색 결과를 가져옴
# result = search.invoke("판교 카카오 프렌즈샵 아지트점의 전화번호는 무엇인가요?")

# PDF 파일 로드 (파일의 경로 입력)
loader = PyPDFLoader("AI핵심이론.pdf")

# 텍스트 분할기를 사용하여 문서를 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할
split_docs = loader.load_and_split(text_splitter)

# VectorStore를 생성 (문서 임베딩을 통해 효율적인 검색을 가능하게 함)
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성 (벡터 저장소를 기반으로 문서 검색을 수행)
retriever = vector.as_retriever()

# langchain 패키지의 tools 모듈에서 retriever 도구를 생성하는 함수를 가져옴
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="AI핵심이론 대해 PDF 문서에서 검색합니다. 'Prompt Engineering' 과 관련된 질문은 이 도구를 사용해야 합니다!",
)

# Agent가 사용할 도구 목록 정의
# tools 리스트에 search와 retriever_tool을 추가
tools = [search, retriever_tool]

# ChatOpenAI 클래스를 langchain_openai 모듈에서 가져옴 (LLM 모델 설정)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# hub에서 프롬프트를 가져옴 (필요에 따라 수정 가능)
prompt = hub.pull("hwchase17/openai-functions-agent")

# OpenAI 함수 기반 에이전트를 생성 (llm, tools, prompt를 인자로 사용)
agent = create_openai_functions_agent(llm, tools, prompt)

# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 설정
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 채팅 메시지 기록을 관리하는 객체를 생성
message_history = ChatMessageHistory()

# 채팅 메시지 기록이 추가된 에이전트를 생성
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대부분의 실제 시나리오에서 세션 ID가 필요 (여기서는 간단한 메모리 내 ChatMessageHistory를 사용)
    lambda session_id: message_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)

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