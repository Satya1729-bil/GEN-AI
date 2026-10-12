"""RxReveal Streamlit app.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import hmac
import html
import re
from pathlib import Path

import streamlit as st

from rxreveal.config import get_settings
from rxreveal.index import Embedder, EmbedderError, VectorIndex, make_embedder
from rxreveal.ingest import load_folder
from rxreveal.llm import GeminiLLM, LLMError
from rxreveal.rag import answer_question

settings = get_settings()

st.set_page_config(
    page_title="RxReveal | Medical Document Assistant",
    page_icon=":material/medical_information:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styles
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --rx-teal: #0E7C7B;
  --rx-teal-soft: #E6F3F2;
  --rx-ink: #10242B;
  --rx-muted: #5B6F76;
  --rx-line: #D9E6E6;
  --rx-card: #FFFFFF;
  --rx-wash: #F5FAFA;
}

html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

.block-container { padding-top: 2.2rem; max-width: 820px; }

.rx-header { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.rx-mark {
  width: 44px; height: 44px; border-radius: 12px; background: var(--rx-teal);
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.rx-title { font-size: 1.65rem; font-weight: 700; color: var(--rx-ink); letter-spacing: -0.02em; line-height: 1.1; }
.rx-title span { color: var(--rx-teal); }
.rx-sub { color: var(--rx-muted); font-size: 0.95rem; margin: 4px 0 18px 0; }

.rx-notice {
  display: flex; gap: 10px; align-items: flex-start;
  background: var(--rx-wash); border: 1px solid var(--rx-line); border-left: 4px solid var(--rx-teal);
  border-radius: 10px; padding: 10px 14px; color: var(--rx-muted); font-size: 0.85rem; margin-bottom: 18px;
}

.rx-empty {
  border: 1px dashed var(--rx-line); border-radius: 14px; padding: 22px; text-align: center;
  color: var(--rx-muted); background: var(--rx-wash); margin: 8px 0 16px 0;
}
.rx-empty b { color: var(--rx-ink); }

.rx-section { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--rx-muted); margin: 6px 0 8px 0; }

.rx-source { border: 1px solid var(--rx-line); border-radius: 10px; padding: 10px 12px; margin-bottom: 8px;
  background: var(--rx-card); }
.rx-source-head { display: flex; justify-content: space-between; gap: 8px; font-size: 0.8rem;
  color: var(--rx-muted); margin-bottom: 6px; }
.rx-source-head b { color: var(--rx-teal); }
.rx-source-text { font-size: 0.85rem; color: var(--rx-ink); white-space: pre-wrap; line-height: 1.45; }

.rx-pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 500;
  background: var(--rx-teal-soft); color: var(--rx-teal); }
.rx-pill.warn { background: #FFF4E5; color: #9A5B00; }

section[data-testid="stSidebar"] { background: var(--rx-wash); border-right: 1px solid var(--rx-line); }
.stButton > button { border-radius: 10px; border: 1px solid var(--rx-line); }
.stButton > button:hover { border-color: var(--rx-teal); color: var(--rx-teal); }
div[data-testid="stChatInput"] textarea { font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)

LOGO_SVG = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z" fill="#FFFFFF"/></svg>'
)


# ---------------------------------------------------------------- access gate
def password_gate() -> None:
    """Optional password screen, turned on by setting APP_PASSWORD."""
    if not settings.app_password or st.session_state.get("authenticated"):
        return
    st.markdown(
        f'<div class="rx-header"><div class="rx-mark">{LOGO_SVG}</div>'
        '<div class="rx-title">Rx<span>Reveal</span></div></div>',
        unsafe_allow_html=True,
    )
    with st.form("login"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if hmac.compare_digest(password, settings.app_password):
                st.session_state.authenticated = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()


password_gate()


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Starting the search engine...")
def _cached_embedder(backend: str, gemini_model: str, dimension: int, api_key: str | None,
                     local_model: str) -> Embedder:
    # The arguments are the cache key; make_embedder reads the same values from settings.
    return make_embedder(settings)


def get_embedder() -> tuple[Embedder | None, str | None]:
    try:
        embedder = _cached_embedder(settings.embed_backend, settings.gemini_embed_model,
                                    settings.embed_dimension, settings.gemini_api_key,
                                    settings.embed_model)
        return embedder, None
    except EmbedderError as exc:
        return None, str(exc)


def _index_version(folder: Path) -> float:
    meta = folder / VectorIndex.META_FILE
    return meta.stat().st_mtime if meta.exists() else 0.0


@st.cache_resource(show_spinner=False)
def get_index(folder: str, version: float) -> VectorIndex | None:
    path = Path(folder)
    return VectorIndex.load(path) if VectorIndex.exists(path) else None


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str | None, model: str) -> GeminiLLM | None:
    if not api_key:
        return None
    try:
        return GeminiLLM(api_key=api_key, model=model)
    except LLMError:
        return None


def rebuild_index(embedder: Embedder) -> VectorIndex:
    chunks = load_folder(settings.data_dir, size=settings.chunk_size, overlap=settings.chunk_overlap)
    index = VectorIndex.build(chunks, embedder)
    index.save(settings.index_dir)
    get_index.clear()
    return index


def safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "document"
    return f"{stem[:80]}.pdf"


# ---------------------------------------------------------------- search engine + index
settings.data_dir.mkdir(parents=True, exist_ok=True)
has_pdfs = any(settings.data_dir.rglob("*.pdf"))
embedder, embedder_error = get_embedder()
index_error = None


def try_rebuild(message: str) -> None:
    global index_error
    with st.spinner(message):
        try:
            rebuild_index(embedder)
        except (ValueError, EmbedderError) as exc:
            index_error = str(exc)


if embedder and has_pdfs and not VectorIndex.exists(settings.index_dir):
    try_rebuild("Indexing your documents for the first time...")

index = get_index(str(settings.index_dir), _index_version(settings.index_dir))

# An index built with a different embedding model lives in a different vector space,
# so it is rebuilt automatically (for example after switching RX_EMBED_BACKEND).
if embedder and index and not index.matches(embedder) and has_pdfs:
    try_rebuild("The search settings changed. Re-indexing your documents...")
    index = get_index(str(settings.index_dir), _index_version(settings.index_dir))
    if index and not index.matches(embedder):
        index = None
llm = get_llm(settings.gemini_api_key, settings.gemini_model)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "top_k" not in st.session_state:
    st.session_state.top_k = settings.top_k

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="rx-section">Document library</div>', unsafe_allow_html=True)
    if index:
        c1, c2 = st.columns(2)
        c1.metric("Files", len(index.meta.get("sources", [])))
        c2.metric("Passages", index.meta.get("chunks", 0))
        for name in index.meta.get("sources", []):
            st.markdown(f":material/description: {html.escape(name)}")
        st.caption(f"Last indexed {index.meta.get('built_at', '')}")
    else:
        st.caption("No documents indexed yet.")

    if settings.allow_upload:
        st.markdown('<div class="rx-section">Add documents</div>', unsafe_allow_html=True)
        uploads = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
        )
        if uploads and embedder and st.button("Add to library", type="primary", use_container_width=True):
            target = settings.data_dir / "uploads"
            target.mkdir(parents=True, exist_ok=True)
            saved = 0
            for file in uploads:
                data = file.getvalue()
                if len(data) > settings.max_upload_mb * 1024 * 1024:
                    st.warning(f"{file.name} is larger than {settings.max_upload_mb} MB and was skipped.")
                    continue
                if not data.startswith(b"%PDF"):
                    st.warning(f"{file.name} is not a valid PDF and was skipped.")
                    continue
                (target / safe_filename(file.name)).write_bytes(data)
                saved += 1
            if saved:
                with st.spinner("Indexing documents..."):
                    try:
                        rebuild_index(embedder)
                        st.rerun()
                    except (ValueError, EmbedderError) as exc:
                        st.error(str(exc))

    if embedder and has_pdfs and st.button("Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding the index..."):
            try:
                rebuild_index(embedder)
                st.rerun()
            except (ValueError, EmbedderError) as exc:
                st.error(str(exc))

    st.markdown('<div class="rx-section">Settings</div>', unsafe_allow_html=True)
    st.session_state.top_k = st.slider(
        "Passages to search", min_value=3, max_value=10, value=st.session_state.top_k,
        help="How many document passages are given to the model for each answer.",
    )
    if llm:
        st.markdown(f'<span class="rx-pill">Answers by {html.escape(settings.gemini_model)}</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="rx-pill warn">No answer model: passages only</span>',
                    unsafe_allow_html=True)
        st.caption("Add GEMINI_API_KEY to your .env file to get written answers.")
    if embedder:
        st.caption(f"Search: {html.escape(embedder.name)}")

    if st.session_state.messages and st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- header
st.markdown(
    f'<div class="rx-header"><div class="rx-mark">{LOGO_SVG}</div>'
    '<div class="rx-title">Rx<span>Reveal</span></div></div>'
    '<div class="rx-sub">Ask questions about your medical documents. '
    'Answers come only from the files in your library, with page references.</div>'
    '<div class="rx-notice"><span>For information only. RxReveal does not give medical advice, '
    'diagnosis or treatment. Always check with a qualified clinician.</span></div>',
    unsafe_allow_html=True,
)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/menu_book:"):
        for s in sources:
            text = s["text"]
            excerpt = text if len(text) <= 700 else text[:700].rstrip() + " ..."
            st.markdown(
                f'<div class="rx-source"><div class="rx-source-head">'
                f'<span><b>[{s["number"]}]</b> {html.escape(s["source"])}, page {s["page"]}</span>'
                f'<span>match {s["score"]:.2f}</span></div>'
                f'<div class="rx-source-text">{html.escape(excerpt)}</div></div>',
                unsafe_allow_html=True,
            )


def render_message(msg: dict) -> None:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/medical_information:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("error"):
            st.caption(f":material/error: {msg['error']}")
        render_sources(msg.get("sources", []))


# ---------------------------------------------------------------- chat
if embedder_error:
    st.error(f"Search is not set up yet. {embedder_error}", icon=":material/key:")
    st.stop()

if index_error:
    st.error(f"Could not index your documents. {index_error}", icon=":material/error:")

if not index:
    st.markdown(
        '<div class="rx-empty"><b>Your library is empty.</b><br>'
        "Upload PDF files from the sidebar, or put them in the <code>data</code> folder "
        "and run <code>python -m rxreveal.build_index</code>.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

EXAMPLES = [
    "What are the common symptoms of tuberculosis?",
    "Which patients were treated with corticosteroids?",
    "List patients with a temperature above 102 °F.",
    "What follow-up periods are most common?",
]

question = st.chat_input("Ask about symptoms, diagnoses, treatments, follow-ups...")
question = question or st.session_state.pop("pending", None)

if not st.session_state.messages and not question:
    st.markdown('<div class="rx-section">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for message in st.session_state.messages:
    render_message(message)

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    history = []
    msgs = st.session_state.messages[:-1]
    for q, a in zip(msgs[::2], msgs[1::2]):
        if q["role"] == "user" and a["role"] == "assistant" and a.get("mode") == "llm":
            history.append((q["content"], a["content"]))

    with st.chat_message("assistant", avatar=":material/medical_information:"):
        with st.spinner("Searching your documents..."):
            result = answer_question(
                question,
                index=index,
                embedder=embedder,
                llm=llm,
                top_k=st.session_state.top_k,
                min_score=settings.min_score,
                history=history,
            )
        sources = [
            {"number": s.number, "source": s.chunk.source, "page": s.chunk.page,
             "score": s.score, "text": s.chunk.text}
            for s in result.sources
        ]
        st.markdown(result.text)
        if result.error:
            st.caption(f":material/error: {result.error}")
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.text, "sources": sources,
         "mode": result.mode, "error": result.error}
    )
