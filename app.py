import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="TechDocs Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #757575;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #1E88E5;
        margin-top: 10px;
        font-size: 0.9em;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .source-box {
            background-color: #262730;
            border-left: 5px solid #4FC3F7;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    # Local HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def get_custom_sources(vs):
    try:
        collection = vs._collection
        res = collection.get(include=['metadatas'])
        sources = set()
        for meta in res.get('metadatas', []):
            if meta and 'source' in meta:
                src = meta['source']
                # Assume official docs are URLs (starting with http)
                if not src.startswith("http"):
                    sources.add(src)
        return sorted(list(sources))
    except Exception:
        return []

def init_chain(vectorstore, temperature=0.2):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=temperature)
    
    prompt_template = """You are a highly skilled Technical Documentation Assistant.
Your sole purpose is to answer the user's question using ONLY the provided context from the documentation.
You must be precise, professional, and directly answer the question.
If the answer cannot be found in the context, you must state: 
"I'm sorry, but I can only answer questions based on the provided documentation, and I couldn't find the answer to that question in my knowledge base."
Do not use your general knowledge.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown source")
            formatted_docs.append(f"Source: {source}\nContent: {doc.page_content}")
        return "\n\n".join(formatted_docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8649/8649603.png", width=120)
    st.title("Settings & Info")
    
    st.markdown("### 📚 Knowledge Base")
    st.info("This RAG Chatbot is currently trained on official documentation from:\n"
            "- **FastAPI** (Framework)\n"
            "- **Python** (Language)\n"
            "- **Pandas** (Library)")
    
    try:
        vs_temp = get_vectorstore()
        user_docs = get_custom_sources(vs_temp)
        if user_docs:
            st.markdown("#### 📁 Your Uploaded Documents")
            for src in user_docs:
                col1, col2 = st.columns([4, 1])
                col1.caption(f"📄 {src}")
                if col2.button("❌", key=f"del_{src}", help=f"Delete {src}"):
                    vs_temp._collection.delete(where={"source": src})
                    st.rerun()
    except Exception:
        pass
    
    with st.expander("📄 Upload New Documents", expanded=False):
        doc_name = st.text_input("Custom Name (Optional)", placeholder="e.g. My Company Data")
        uploaded_files = st.file_uploader("Upload PDF/TXT files to add to knowledge base:", type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Add to Knowledge Base", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    try:
                        vs = get_vectorstore()
                        new_docs = []
                        import tempfile
                        from langchain_community.document_loaders import PyPDFLoader, TextLoader
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        
                        for file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split(".")[-1]) as tmp_file:
                                tmp_file.write(file.getvalue())
                                tmp_path = tmp_file.name
                                
                            if file.name.lower().endswith(".pdf"):
                                loader = PyPDFLoader(tmp_path)
                            else:
                                loader = TextLoader(tmp_path, encoding="utf-8")
                                
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["source"] = doc_name.strip() if doc_name.strip() else file.name
                            new_docs.extend(docs)
                            os.unlink(tmp_path)
                        
                        if new_docs:
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
                            splits = text_splitter.split_documents(new_docs)
                            vs.add_documents(splits)
                            st.success(f"Added {len(uploaded_files)} doc(s) ({len(splits)} chunks)!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please upload files first.")
    
    st.markdown("### 🛠️ Architecture")
    
    with st.expander("👀 Pipeline Visualization"):
        st.markdown("""
**🔵 Basic RAG**
`Question → Search → Top-K Docs → LLM → Answer`
Assumes all retrieved documents are perfectly relevant.

**🟠 Corrective RAG (CRAG)**
`Question → Search → Top-K Docs → 🧠 Evaluate`
If context is poor:
`→ Correct/Filter → Generate Answer`
Checks whether retrieved documents are useful before answering!
        """)
    
    def on_mode_change():
        st.session_state.messages = [
            {"role": "assistant", "content": f"Switched to {st.session_state.rag_mode_radio} Mode. How can I help you?", "sources": []}
        ]

    rag_mode = st.radio("Select RAG Mode", ["Basic RAG", "CRAG", "Compare Both"], key="rag_mode_radio", on_change=on_mode_change)
    
    with st.expander("⚙️ Engine Settings", expanded=False):
        temperature = st.slider("Response Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        st.markdown("- **Embeddings**: `all-MiniLM-L6-v2`\n- **LLM**: `gemini-3.6-flash`\n- **Vector DB**: ChromaDB")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Technical Documentation Assistant. How can I help you with Python, Pandas, or FastAPI today?", "sources": []}
        ]
        st.rerun()
    
    st.caption("Developed for Viva Project Demonstration.")

# --- Main App ---
st.markdown('<p class="main-header">🤖 TechDocs Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask me anything about Python, Pandas, or FastAPI etc.</p>', unsafe_allow_html=True)

# Initialization checks
if not os.path.exists("./chroma_db"):
    st.error("Vector database not found. Please run `python ingest.py` first.")
    st.stop()

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

try:
    vectorstore = get_vectorstore()
    # Read temperature from sidebar or default to 0.2
    temp = locals().get("temperature", 0.2) 
    chain, retriever = init_chain(vectorstore, temperature=temp)
    from crag import get_crag_app
    crag_app = get_crag_app(vectorstore, temperature=temp)
except Exception as e:
    st.error(f"Error initializing the system: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Technical Documentation Assistant. How can I help you with Python, Pandas, or FastAPI today?", "sources": []}
    ]

# Display chat messages
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Display sources if present
        if message.get("sources"):
            with st.expander("📚 View References"):
                for src, snippet in message["sources"]:
                    st.markdown(f"<div class='source-box'><b>🔗 Source:</b> <a href='{src}' target='_blank'>{src}</a><br><i>...{snippet}...</i></div>", unsafe_allow_html=True)

# Suggested Questions
if len(st.session_state.messages) == 1:
    st.markdown("### Test Questions (Try these!)")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🐼 Easy: Read CSV", use_container_width=True, help="Both RAGs will answer this easily."):
            st.session_state.prompt_from_button = "How do I read a CSV file in Pandas?"
    with col2:
        if st.button("🚀 Mixed: Pandas + FastAPI", use_container_width=True, help="Forces the retriever to pull from two domains."):
            st.session_state.prompt_from_button = "How can I read a CSV file in Pandas and then expose the result through FastAPI?"
    with col3:
        if st.button("❓ Ambiguous / Poor Query", use_container_width=True, help="Retrieves irrelevant context to trigger CRAG."):
            st.session_state.prompt_from_button = "How do I load data and create an API?"

# Chat Input
prompt = st.chat_input("E.g., How do I read a CSV file in Pandas?")

if "prompt_from_button" in st.session_state:
    prompt = st.session_state.prompt_from_button
    del st.session_state.prompt_from_button

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documentation..."):
            try:
                # Helper function for sources
                def get_sources_data(docs_list):
                    src_data = []
                    u_urls = set()
                    for d in docs_list:
                        u = d.metadata.get("source", "Unknown")
                        if u not in u_urls:
                            u_urls.add(u)
                            snip = d.page_content.replace('\\n', ' ')[:150]
                            src_data.append((u, snip))
                    return src_data

                if rag_mode == "Basic RAG":
                    docs = retriever.invoke(prompt)
                    sources_data = get_sources_data(docs)
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    
                elif rag_mode == "CRAG":
                    res = crag_app.invoke({"question": prompt})
                    docs = res.get("documents", [])
                    sources_data = get_sources_data(docs)
                    response = res["generation"]
                    st.markdown(response)
                    
                elif rag_mode == "Compare Both":
                    import time
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🔵 Basic RAG")
                        start_time = time.time()
                        
                        docs_and_scores = vectorstore.similarity_search_with_score(prompt, k=4)
                        base_docs = []
                        for d, s in docs_and_scores:
                            d.metadata['vector_similarity'] = max(0.0, 1.0 - (s / 2.0)) * 100
                            base_docs.append(d)
                            
                        base_response = chain.invoke(prompt)
                        base_time = time.time() - start_time
                        
                        st.markdown(f"""
**Retrieval Metrics:**
- 📄 Documents Retrieved: {len(base_docs)}
- 🎯 Context Quality: Unknown (Assumed 100%)
- 🧠 LLM Calls: 1
- ⏱️ Time: {base_time:.2f}s

**Pipeline Status:**
`LLM directly used all {len(base_docs)} retrieved documents without filtering.`
                        """)
                        
                        with st.expander("📚 Retrieved Documents"):
                            for i, d in enumerate(base_docs):
                                src = d.metadata.get('source', 'Unknown')
                                sim = d.metadata.get('vector_similarity', 0.0)
                                st.write(f"**Doc {i+1}:** {src}")
                                st.caption(f"Similarity: {sim:.1f}% | Status: ✅ USED")
                                
                        st.info(base_response)
                        
                    with col2:
                        st.markdown("### 🟠 Corrective RAG")
                        start_time = time.time()
                        crag_res = crag_app.invoke({"question": prompt})
                        crag_time = time.time() - start_time
                        
                        crag_docs = crag_res.get("documents", [])
                        grading_details = crag_res.get("grading_details", [])
                        crag_response = crag_res["generation"]
                        
                        relevant_count = sum(1 for d in grading_details if d['relevant'])
                        total_count = len(grading_details)
                        quality_score = (relevant_count / total_count * 100) if total_count > 0 else 0
                        removed_count = total_count - relevant_count
                        
                        if quality_score == 100:
                            assessment = "🟢 CORRECT"
                        elif quality_score > 0:
                            assessment = "🟡 AMBIGUOUS"
                        else:
                            assessment = "🔴 INCORRECT"
                            
                        st.markdown(f"""
**Retrieval Metrics:**
- 📄 Documents Retrieved: {total_count}
- 🎯 Context Quality: {quality_score:.0f}%
- 🧠 LLM Calls: 2 (Grader + Generator)
- ⏱️ Time: {crag_time:.2f}s

**🔧 Correction Performed:**
`Evaluated {total_count} docs → Rejected {removed_count} irrelevant docs → Generated answer with remaining {relevant_count} docs.`
                        """)
                        
                        with st.expander("🔍 CRAG Retrieval Evaluation"):
                            st.markdown(f"**Overall Assessment:** {assessment}")
                            st.markdown("---")
                            for i, detail in enumerate(grading_details):
                                icon = "✅ KEEP" if detail['relevant'] else "❌ REMOVE"
                                relevance = "HIGH" if detail['relevant'] else "LOW"
                                color = "green" if detail['relevant'] else "red"
                                sim = detail.get('similarity', 0.0)
                                st.write(f"**Doc {i+1}:** {detail['source']}")
                                st.markdown(f"Similarity: {sim:.1f}% | Relevance: :{color}[{relevance}] | Action: **{icon}**")
                                
                        st.success(crag_response)
                        
                    # Calculate token sizes (roughly 4 chars per token)
                    base_context_length = sum(len(d.page_content) for d in base_docs)
                    crag_context_length = sum(len(d.page_content) for d in crag_docs)
                    base_tokens = base_context_length // 4
                    crag_tokens = crag_context_length // 4

                    st.markdown("---")
                    st.markdown("### 📊 Pipeline Comparison Summary")
                    
                    st.markdown(f"""
| Metric | 🔵 Basic RAG | 🟠 Corrective RAG |
| :--- | :--- | :--- |
| **Documents Retrieved** | {len(base_docs)} | {total_count} |
| **Documents Used** | {len(base_docs)} | {relevant_count} |
| **Context Tokens Sent to LLM** | ~{base_tokens:,} tokens | ~{crag_tokens:,} tokens |
| **API Calls Made** | 1 call | 2 calls |
| **Total Latency** | {base_time:.2f}s | {crag_time:.2f}s |
| **Architecture Logic** | Blindly trusts vector database | Actively grades and filters noise |
""")
                    
                    sources_data = get_sources_data(crag_docs)
                    response = f"*(Comparison Mode executed)*"
                
                # Display sources dynamically
                if sources_data and "I'm sorry" not in response and rag_mode != "Compare Both":
                    with st.expander("📚 View References"):
                        for src, snippet in sources_data:
                            st.markdown(f"<div class='source-box'><b>🔗 Source:</b> <a href='{src}' target='_blank'>{src}</a><br><i>...{snippet}...</i></div>", unsafe_allow_html=True)
                elif rag_mode == "Compare Both" and sources_data:
                    with st.expander("📚 View CRAG Filtered References"):
                        for src, snippet in sources_data:
                            st.markdown(f"<div class='source-box'><b>🔗 Source:</b> <a href='{src}' target='_blank'>{src}</a><br><i>...{snippet}...</i></div>", unsafe_allow_html=True)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "sources": sources_data if "I'm sorry" not in response else []
                })
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
