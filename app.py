import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="AI Knowledge Base | Week 2", page_icon="📚", layout="wide")
st.title("📚 AI Knowledge Base — RAG Chat")
st.caption("Upload PDF documents, then ask questions. Answers are grounded in retrieved document passages.")

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning("Add your Google Gemini API key to a local .env file. See .env.example. Never upload your real .env file to GitHub.")

@st.cache_resource
def get_embeddings(key):
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=key)

@st.cache_resource
def get_llm(key):
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, temperature=0.2)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

with st.sidebar:
    st.header("1. Add knowledge")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    chunk_size = st.slider("Chunk size", 500, 1500, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 150, 25)
    if st.button("Process PDFs", type="primary", use_container_width=True):
        if not api_key:
            st.error("Set GOOGLE_API_KEY in your .env file first.")
        elif not uploaded_files:
            st.error("Upload at least one PDF.")
        else:
            all_docs = []
            progress = st.progress(0, text="Reading PDF documents...")
            try:
                for i, uploaded in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.getvalue())
                        temp_path = tmp.name
                    try:
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source"] = uploaded.name
                        all_docs.extend(docs)
                    finally:
                        os.unlink(temp_path)
                    progress.progress((i + 1) / len(uploaded_files), text=f"Read {uploaded.name}")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    add_start_index=True
                )
                chunks = splitter.split_documents(all_docs)
                if not chunks:
                    st.error("No readable text was found. If the PDF is scanned, run OCR on it first.")
                else:
                    embeddings = get_embeddings(api_key)
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        collection_name="week2_knowledge_base"
                    )
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.session_state.chat_history = []
                    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s).")
            except Exception as exc:
                st.error(f"Could not process the PDFs: {exc}")
            finally:
                progress.empty()

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for filename in st.session_state.indexed_files:
            st.write(f"📄 {filename}")
        if st.button("Clear knowledge base", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.indexed_files = []
            st.session_state.chat_history = []
            st.rerun()

st.header("2. Ask questions")
if st.session_state.vectorstore is None:
    st.info("Upload PDF files in the sidebar and click **Process PDFs** to start chatting.")
else:
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Sources"):
                    for source in item["sources"]:
                        st.markdown(f"- **{source['file']}**, page {source['page']}")
                        st.caption(source["excerpt"])

    question = st.chat_input("Ask a question about your PDFs...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and preparing an answer..."):
                try:
                    docs = st.session_state.vectorstore.similarity_search(question, k=4)
                    if not docs:
                        answer = "I couldn't find relevant information in the uploaded documents."
                        sources = []
                    else:
                        context_parts = []
                        sources = []
                        seen = set()
                        for doc in docs:
                            filename = os.path.basename(doc.metadata.get("source", "Unknown PDF"))
                            page = int(doc.metadata.get("page", 0)) + 1
                            context_parts.append(f"[Source: {filename}, page {page}]\n{doc.page_content}")
                            key = (filename, page)
                            if key not in seen:
                                sources.append({
                                    "file": filename,
                                    "page": page,
                                    "excerpt": doc.page_content[:280].replace("\n", " ") + ("..." if len(doc.page_content) > 280 else "")
                                })
                                seen.add(key)

                        prompt = ChatPromptTemplate.from_messages([
                            ("system", """You are a careful document question-answering assistant.
Answer the user's question using only the supplied context. If the answer is not supported by the context, say you could not find it in the uploaded documents.
Cite claims inline using [filename, page N]. Do not invent citations.
Context:
{context}"""),
                            ("human", "{question}")
                        ])
                        chain = prompt | get_llm(api_key)
                        response = chain.invoke({
                            "context": "\n\n---\n\n".join(context_parts),
                            "question": question
                        })
                        answer = response.content
                        if isinstance(answer, list):
                            answer = "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in answer)
                    st.markdown(answer)
                    if sources:
                        with st.expander("Retrieved sources"):
                            for source in sources:
                                st.markdown(f"- **{source['file']}**, page {source['page']}")
                                st.caption(source["excerpt"])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as exc:
                    st.error(f"Could not answer this question: {exc}")

st.divider()
st.caption("Week 2 Mini Project | LangChain + ChromaDB + Gemini | Answers depend on the quality and text readability of uploaded PDFs.")
