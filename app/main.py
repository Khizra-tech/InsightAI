"""Phase 6: Streamlit interface. Run: streamlit run app/main.py"""
import io

import streamlit as st

from rag_pipeline import RAGPipeline

st.set_page_config(page_title="InsightAI", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📄 InsightAI")
    st.caption("Chat with, summarize and compare your PDFs.")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", type="primary", disabled=not files):
        for f in files:
            try:
                with st.spinner(f"Processing {f.name}..."):
                    n = pipeline.ingest(io.BytesIO(f.getvalue()), f.name)
                st.success(f"{f.name}: {n} chunks")
            except Exception as e:
                st.error(f"{f.name}: {e}")
    st.divider()
    docs = pipeline.store.list_documents()
    st.subheader("Documents")
    for d in docs:
        c1, c2 = st.columns([4, 1])
        c1.write(d)
        if c2.button("🗑", key=f"del-{d}"):
            pipeline.store.delete_document(d)
            st.rerun()
    if not docs:
        st.info("No documents yet.")

tab_chat, tab_sum, tab_cmp = st.tabs(["💬 Chat", "📝 Summarize", "⚖️ Compare"])

# ---------- Chat ----------
with tab_chat:
    scope = st.selectbox("Search in", ["All documents"] + docs)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    for s in m["sources"]:
                        st.markdown(f"**{s['source']}, p.{s['page']}** (score {s['score']})")
                        st.caption(s["text"][:300] + "...")
    if q := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    res = pipeline.answer(q, source=None if scope == "All documents" else scope)
                st.markdown(res["answer"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": res["answer"], "sources": res["sources"]}
                )
            except Exception as e:
                st.error(str(e))

# ---------- Summarize ----------
with tab_sum:
    if docs:
        target = st.selectbox("Document", docs, key="sum_doc")
        if st.button("Summarize"):
            try:
                with st.spinner("Summarizing..."):
                    st.markdown(pipeline.summarize(target))
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Upload a document first.")

# ---------- Compare ----------
with tab_cmp:
    if len(docs) >= 2:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Document A", docs, key="cmp_a")
        b = c2.selectbox("Document B", docs, index=1, key="cmp_b")
        aspect = st.text_input("Compare on", "main topics and conclusions")
        if st.button("Compare"):
            if a == b:
                st.warning("Pick two different documents.")
            else:
                try:
                    with st.spinner("Comparing..."):
                        st.markdown(pipeline.compare(a, b, aspect))
                except Exception as e:
                    st.error(str(e))
    else:
        st.info("Upload at least two documents to compare.")
