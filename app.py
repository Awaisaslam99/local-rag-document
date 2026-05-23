import streamlit as st
import json
from pipeline import DocumentIngestor, DocumentProcessor, LocalSearchEngine, DocumentQA

st.set_page_config(page_title="AI Document Processing Pipeline", layout="wide")

st.title("Intelligent Document Processing")
st.markdown(
    "Upload your CVs, invoices, or utility bills to classify them, "
    "extract structured fields, and query them semantically."
)


# ---------------------------------------------------------------------------
# Cache heavy models so they don't reload on every Streamlit rerun
# ---------------------------------------------------------------------------

@st.cache_resource
def load_processor():
    return DocumentProcessor()

processor = load_processor()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for key, default in [
    ("raw_docs", {}),
    ("processed_results", None),
    ("search_engine", None),
    ("qa_engine", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# SIDEBAR — Upload & Process
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "pdf/txt",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.error("Please select at least one document first.")
        else:
            ingestor = DocumentIngestor()
            new_docs = {}

            # --- Step 1: Extract raw text ---
            with st.spinner("Extracting raw text from files..."):
                for file in uploaded_files:
                    if file.name.lower().endswith(".pdf"):
                        new_docs[file.name] = ingestor.extract_text_from_pdf(file)
                    elif file.name.lower().endswith(".txt"):
                        new_docs[file.name] = " ".join(
                            file.read().decode("utf-8", errors="ignore").split()
                        )

            # --- Step 2: Classify + extract fields (runs ONCE) ---
            with st.spinner("Running classification and field extraction..."):
                results = processor.process_all_documents(new_docs)

                # Auto-save Output.json
                with open("Output.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)

            # --- Step 3: Build semantic search index ---
            with st.spinner("Building FAISS semantic search index..."):
                search_engine = LocalSearchEngine(new_docs)

            qa_engine = DocumentQA(search_engine, results, new_docs)

            st.session_state.qa_engine = qa_engine

            # Commit to session state only after all steps succeed
            st.session_state.raw_docs = new_docs
            st.session_state.processed_results = results
            st.session_state.search_engine = search_engine

            st.success(f"Successfully processed {len(new_docs)} file(s). Output saved to Output.json")


# ---------------------------------------------------------------------------
# MAIN PAGE — Results + Search
# ---------------------------------------------------------------------------

if st.session_state.processed_results:
    left_col, right_col = st.columns([3, 2])

    # --- Left: Extraction results ---
    with left_col:
        st.subheader("Pipeline Results")
        tab1, tab2 = st.tabs(["Structured Table", "JSON Output"])

        with tab1:
            table_data = []
            # Use enumerate with start=1 to create a 1-based serial number
            for idx, (fname, info) in enumerate(st.session_state.processed_results.items(), start=1):
                row = {
                    "#": idx,  # This creates your new 1-indexed column
                    "File Name": fname,
                    "Type": info.get("class", "N/A"),
                }
                extra = [f"{k}: {v}" for k, v in info.items() if k != "class"]
                row["Extracted Structural Metadata"] = " | ".join(extra) if extra else "—"
                table_data.append(row)
            
            import pandas as pd
            df_display = pd.DataFrame(table_data).set_index("#")
            
            # Render the updated dataframe
            st.table(df_display)

        with tab2:
            st.json(st.session_state.processed_results)

    # --- Right: Semantic Search & QA ---
    with right_col:
        tab_search, tab_qa = st.tabs(["Semantic Search", "Ask a Question (AI)"])

        with tab_search:
            st.subheader("Semantic Search")
            search_query = st.text_input(
                "Search documents by meaning:",
                placeholder="e.g. payments due in January"
            )
            if search_query and st.session_state.search_engine:
                hits = st.session_state.search_engine.search(search_query)
                if hits:
                    for hit in hits:
                        with st.container(border=True):
                            st.markdown(f"**{hit['filename']}** — Score: `{hit['score']:.4f}`")
                            st.caption(hit["snippet"])
                else:
                    st.info("No matching documents found.")

        with tab_qa:
            st.subheader("Ask a Question (AI)")
            query_input = st.text_input(
                "Ask something about your documents:",
                placeholder="Ask Questions about uploaded files..."
            )
            if query_input and st.session_state.qa_engine:
                with st.spinner("Generating answer..."):
                    qa_result = st.session_state.qa_engine.answer(query_input)

                if qa_result["sources"]:
                    st.markdown("##### Answer")
                    with st.container(border=True):
                        st.markdown(qa_result["answer"])
                else:
                    st.info("No matching documents found.")

else:
    st.info("Upload documents in the sidebar to and run the pipeline.")

