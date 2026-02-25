# --- [중요] Streamlit Cloud에서 ChromaDB 오류 해결을 위한 코드 (가장 윗줄에 있어야 함) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ------------------------------------------------------------------------------------

import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# [수정된 부분] 옛날 주소 대신 새 주소 사용!
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 설정 및 초기화 ---
st.set_page_config(page_title="SAP S/4HANA Pro", layout="wide")

# 데이터를 영구 저장할 폴더 이름 (이 폴더가 '서가' 역할을 합니다)
PERSIST_DIRECTORY = "./chroma_db"

# 간단한 사용자 아이디/비번 관리 (실제 서비스에선 DB를 써야 하지만, 지금은 이렇게!)
# admin: 관리자 (PDF 업로드 가능), user: 일반 사용자 (채팅만 가능)
USERS = {
    "admin": "admin123",  # 관리자 ID : 비밀번호
    "user1": "guest123",  # 사용자 ID : 비밀번호
    "client": "sap2024"   # 또 다른 사용자
}

# --- 세션 상태 초기화 (로그인 상태 기억하기 위함) ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "user_role" not in st.session_state:
    st.session_state["user_role"] = ""

# --- 1. 로그인 화면 함수 ---
def login_page():
    st.title("🔒 SAP S/4HANA Assistant Login")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        
        if st.button("로그인"):
            if username in USERS and USERS[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                # admin이면 관리자 권한 부여
                st.session_state["user_role"] = "admin" if username == "admin" else "user"
                st.rerun() # 화면 새로고침
            else:
                st.error("아이디 또는 비밀번호가 틀렸습니다.")

# --- 2. 메인 앱 (로그인 성공 후) ---
def main_app():
    # 사이드바: 로그아웃 및 기본 정보
    with st.sidebar:
        st.write(f"환영합니다, **{st.session_state['username']}**님!")
        if st.button("로그아웃"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.rerun()
        
        st.divider()
        # API Key는 관리자가 아니더라도 동작하게 하려면 코드 내부에 심거나, 
        # 여기서는 편의상 입력받게 합니다. (실제론 환경변수 사용 권장)
        api_key = st.text_input("OpenAI API Key", type="password")
        
        menu = st.radio("메뉴 이동", ["💬 AI Chatbot", "📝 My Wiki"])

    if not api_key:
        st.warning("사이드바에 OpenAI API Key를 입력해주세요.")
        return

    # --- 관리자 전용 기능: 지식 데이터베이스 업데이트 ---
    # 오직 'admin' 계정으로 들어왔을 때만 이 화면이 보입니다.
    if st.session_state["user_role"] == "admin":
        with st.expander("🛠️ [관리자 메뉴] 지식 데이터(PDF) 추가하기"):
            st.info("이곳은 관리자만 볼 수 있습니다. 새로운 SAP 매뉴얼을 추가하세요.")
            uploaded_files = st.file_uploader("PDF 파일 업로드", type=["pdf"], accept_multiple_files=True)
            
            if st.button("DB에 저장 및 학습시키기"):
                if uploaded_files:
                    with st.spinner("문서를 분석하고 서가(DB)에 저장 중입니다..."):
                        documents = []
                        for uploaded_file in uploaded_files:
                            # 임시 저장
                            with open(uploaded_file.name, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # 로드 및 분할
                            loader = PyPDFLoader(uploaded_file.name)
                            docs = loader.load()
                            documents.extend(docs)
                            os.remove(uploaded_file.name) # 임시 파일 삭제

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        splits = text_splitter.split_documents(documents)

                        # **핵심**: persist_directory에 저장합니다. (영구 보존)
                        vectordb = Chroma.from_documents(
                            documents=splits, 
                            embedding=OpenAIEmbeddings(api_key=api_key),
                            persist_directory=PERSIST_DIRECTORY
                        )
                        st.success(f"성공! {len(documents)}개의 페이지가 데이터베이스에 추가되었습니다.")
                else:
                    st.warning("업로드할 파일이 없습니다.")

    # --- 기능 1: AI Chatbot (DB 활용) ---
    if menu == "💬 AI Chatbot":
        st.header("S/4HANA Expert AI")
        
        # 채팅 기록 관리
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # DB에서 검색해서 답변하기
            try:
                # 저장된 DB 불러오기
                embedding_function = OpenAIEmbeddings(api_key=api_key)
                vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)
                
                # 검색기 설정
                retriever = vectordb.as_retriever(search_kwargs={"k": 15}) # 관련 문서 3개 참조
                
                # LLM 설정
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=api_key)
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

                with st.chat_message("assistant"):
                    with st.spinner("문서 검색 중..."):
                        response = qa_chain.run(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                # DB가 아직 없을 때 예외 처리
                st.error("아직 학습된 데이터가 없습니다. 관리자에게 문의하세요.")
                # st.error(f"Error detail: {e}") # 디버깅용

    # --- 기능 2: Wiki ---
    elif menu == "📝 My Wiki":
        st.header("프로젝트 위키 (Wiki)")
        st.markdown("""
        이곳은 팀원들이 공통으로 보는 가이드라인 페이지입니다.
        Markdown 문법을 지원합니다.
        """)
        
        # 탭으로 구분
        tab1, tab2 = st.tabs(["읽기 모드", "수정 모드"])
        
        # 위키 내용도 파일로 저장하면 좋지만, 간단히 세션으로 예시
        if "wiki_content" not in st.session_state:
            st.session_state["wiki_content"] = "### 환영합니다\n이곳은 SAP 프로젝트 위키입니다."

        with tab1:
            st.markdown(st.session_state["wiki_content"])
        
        with tab2:
            new_content = st.text_area("내용 수정", st.session_state["wiki_content"], height=300)
            if st.button("위키 저장"):
                st.session_state["wiki_content"] = new_content
                st.success("저장되었습니다!")
                st.rerun()

# --- 앱 실행 흐름 제어 ---
if st.session_state["logged_in"]:
    main_app()
else:

    login_page()
