# --- [필수] Streamlit Cloud 오류 방지 코드 ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# [추가됨] 인터넷 검색을 위한 도구
from langchain_community.tools import DuckDuckGoSearchRun

# --- 설정 및 초기화 ---
st.set_page_config(page_title="SAP S/4HANA Pro", layout="wide")

# 데이터 저장소 경로
PERSIST_DIRECTORY = "./chroma_db"
INSTRUCTION_FILE = "system_instruction.txt"

# 사용자 계정
USERS = {
    "admin": "admin123",
    "user1": "guest123",
    "client": "sap2024"
}

# --- 세션 초기화 ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "user_role" not in st.session_state:
    st.session_state["user_role"] = ""

# --- 도우미 함수 ---
def load_instruction():
    if os.path.exists(INSTRUCTION_FILE):
        with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "당신은 SAP S/4HANA 전문가입니다. 문서를 바탕으로 친절하게 답변해주세요."

def save_instruction(text):
    with open(INSTRUCTION_FILE, "w", encoding="utf-8") as f:
        f.write(text)

# --- 1. 로그인 화면 ---
def login_page():
    st.title("🔒 SAP Assistant Login")
    col1, col2 = st.columns([1, 2])
    with col1:
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        if st.button("로그인"):
            if username in USERS and USERS[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["user_role"] = "admin" if username == "admin" else "user"
                st.rerun()
            else:
                st.error("잘못된 정보입니다.")

# --- 2. 메인 앱 ---
def main_app():
    with st.sidebar:
        st.write(f"접속자: **{st.session_state['username']}** ({st.session_state['user_role']})")
        if st.button("로그아웃"):
            st.session_state["logged_in"] = False
            st.rerun()
        st.divider()
        
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = st.text_input("OpenAI API Key", type="password")
        
        if st.session_state["user_role"] == "admin":
            menu_options = ["💬 AI Chatbot", "⚙️ Admin Settings"]
        else:
            menu_options = ["💬 AI Chatbot"]
        menu = st.radio("메뉴", menu_options)

    if not api_key:
        st.warning("API Key가 필요합니다.")
        return

    # --- [관리자] 설정 메뉴 ---
    if menu == "⚙️ Admin Settings":
        st.header("🛠️ 관리자 설정")
        
        st.subheader("1. 지식 데이터(PDF) 관리")
        uploaded_files = st.file_uploader("PDF 업로드", type=["pdf"], accept_multiple_files=True)
        if st.button("PDF 학습 및 저장"):
            if uploaded_files:
                with st.spinner("학습 중..."):
                    documents = []
                    for uploaded_file in uploaded_files:
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        loader = PyPDFLoader(uploaded_file.name)
                        documents.extend(loader.load())
                        os.remove(uploaded_file.name)
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    splits = text_splitter.split_documents(documents)
                    Chroma.from_documents(splits, OpenAIEmbeddings(api_key=api_key), persist_directory=PERSIST_DIRECTORY)
                    st.success("학습 완료!")

        st.divider()
        st.subheader("2. AI 페르소나 설정")
        current_instruction = load_instruction()
        new_instruction = st.text_area("System Prompt", value=current_instruction, height=150)
        if st.button("저장"):
            save_instruction(new_instruction)
            st.success("저장되었습니다.")

    # --- [챗봇] 검색 + 웹 검색 기능 ---
    elif menu == "💬 AI Chatbot":
        st.header("S/4HANA Assistant (Hybrid Search)")
        
        system_instruction = load_instruction()
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- [핵심 로직] PDF 검색 + 웹 검색 결합 ---
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🔍 문서를 검색 중입니다...")

                try:
                    # 1. PDF 문서 검색
                    embedding_function = OpenAIEmbeddings(api_key=api_key)
                    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)
                    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.get_relevant_documents(prompt)
                    
                    pdf_context = "\n".join([doc.page_content for doc in docs])
                    
                    # 2. 웹 검색 (SAP 관련 사이트로 한정)
                    # PDF 내용이 너무 적거나, 확실한 답변을 위해 외부 검색을 병행합니다.
                    message_placeholder.markdown("🌐 SAP Community 및 공식 문서를 검색 중입니다...")
                    
                    search = DuckDuckGoSearchRun()
                    # 검색어에 'site:sap.com' 등을 붙여서 전문가 커뮤니티만 찾게 강제합니다.
                    search_query = f"site:sap.com OR site:help.sap.com OR site:community.sap.com {prompt}"
                    try:
                        web_context = search.run(search_query)
                    except:
                        web_context = "웹 검색을 수행할 수 없습니다."

                    # 3. LLM에게 답변 요청 (문맥 결합)
                    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=api_key)
                    
                    full_template = f"""
                    [지시사항]
                    {system_instruction}
                    
                    당신은 SAP 전문가입니다. 아래의 [내부 문서]와 [외부 검색 결과]를 종합하여 답변하세요.
                    
                    1. 우선적으로 [내부 문서]의 내용을 기반으로 답변하세요.
                    2. 만약 내부 문서에 내용이 없거나 부족하다면, [외부 검색 결과]를 사용하여 답변하세요.
                    3. 외부 검색 결과를 사용했다면, 반드시 답변 끝에 "출처: SAP Community/Help"와 같이 명시하세요.
                    4. 두 곳 모두 정보가 없다면 솔직하게 모른다고 답하세요.

                    [내부 문서 (PDF)]
                    {pdf_context}

                    [외부 검색 결과 (Web)]
                    {web_context}

                    [사용자 질문]
                    {prompt}

                    [답변]
                    """
                    
                    response = llm.invoke(full_template).content
                    
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")

# --- 실행 ---
if st.session_state["logged_in"]:
    main_app()
else:
    login_page()
