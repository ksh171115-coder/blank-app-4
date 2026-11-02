import os
import tempfile
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# --------------------------------------------------------------------
# 1. PDF -> retriever 도구 만들기
# --------------------------------------------------------------------
def build_pdf_tool(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        all_docs.extend(PyPDFLoader(tmp_path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    @tool
    def pdf_search(query: str) -> str:
        """Search the uploaded PDF documents. Use this first when the answer may be in the PDFs."""
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)

    return pdf_search


# --------------------------------------------------------------------
# 2. create_agent 로 에이전트 구성 (LangChain 1.x 표준)
# --------------------------------------------------------------------
def build_agent(tools):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = (
        "You are a helpful assistant for KEPCO KDN employees. "
        "First try `pdf_search`. If it has no relevant result, use `tavily_search`. "
        "Answer in Korean with a professional, friendly tone."
    )
    return create_agent(model, tools, system_prompt=system_prompt)


# --------------------------------------------------------------------
# 3. Streamlit UI
# --------------------------------------------------------------------
st.set_page_config(page_title="RAG + Web Search 챗봇", page_icon="🤖")
st.title("📄🔍 PDF RAG + 웹 검색 챗봇")

with st.sidebar:
    st.header("설정")
    openai_key = st.text_input("OpenAI API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")
    uploaded_files = st.file_uploader("PDF 업로드", type="pdf", accept_multiple_files=True)
    build = st.button("에이전트 만들기")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key

if build:
    if not (openai_key and tavily_key and uploaded_files):
        st.warning("API 키 2개와 PDF를 모두 입력하세요.")
    else:
        with st.spinner("문서 인덱싱 및 에이전트 생성 중..."):
            tools = [build_pdf_tool(uploaded_files), TavilySearch(max_results=5)]
            st.session_state.agent = build_agent(tools)
        st.success("준비 완료! 아래에 질문하세요.")

if "agent" in st.session_state:
    query = st.chat_input("질문을 입력하세요")
    if query:
        st.chat_message("user").write(query)
        with st.chat_message("assistant"):
            result = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            st.write(result["messages"][-1].content)
