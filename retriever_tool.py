from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import pypdf
import os
from dotenv import load_dotenv
# API KEY 정보 로드
load_dotenv()

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("AI핵심이론.pdf")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할합니다.
split_docs = loader.load_and_split(text_splitter)

# VectorStore를 생성합니다.
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성합니다.
retriever = vector.as_retriever()


# PDf 문서에서 Query 에 대한 관련성 높은 Chunk 를 가져옵니다.
pdf_result = retriever.invoke("앨런 튜링의 기계 지능")[0].page_content
print(pdf_result)

