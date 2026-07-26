"""
Korean -> English Neural Machine Translator
--------------------------------------------
A Streamlit front-end for the custom seq2seq + Bahdanau-attention model
trained in the companion notebook. Loads a `tf.saved_model` bundle and
runs real-time inference.

Run with:  streamlit run app.py
"""

import html
import time
from pathlib import Path

import streamlit as st
import tensorflow as tf

# `tensorflow_text` must be imported even though it's never called directly.
# The custom text-normalization op used inside the exported model's graph
# (NFKC Unicode normalization) is registered as a side effect of this
# import. Without it, loading the SavedModel will raise an error.
import tensorflow_text as tf_text  # noqa: F401


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "saved_model"

SAMPLE_SENTENCES = {
    "Easy": "안녕하세요.",
    "Medium": "오늘 날씨가 정말 좋네요.",
    "Hard": "저는 인공지능을 공부하고 있습니다.",
}

st.set_page_config(
    page_title="Korean → English Translator",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 760px;
        padding-top: 2.5rem;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f0f7ff 0%, #eef2ff 100%);
        border: 1px solid #dbe4ff;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
    }
    .result-card h4 {
        margin: 0 0 0.5rem 0;
        color: #4338ca;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .result-text {
        font-size: 1.35rem;
        font-weight: 500;
        color: #1e1b4b;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .source-echo {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.6rem;
        border-top: 1px dashed #dbe4ff;
        padding-top: 0.6rem;
        white-space: pre-wrap;
    }
    .timing-note {
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_DIR.exists():
        return None
    return tf.saved_model.load(str(MODEL_DIR))


def translate_line(model, text: str) -> str:
    result = model.translate(tf.constant([text]))
    return result[0].numpy().decode("utf-8")


def translate_text(model, text: str) -> str:
    """Translate possibly multi-line input, one line at a time.

    The model was trained on single sentences, so long multi-sentence
    blocks translate best when split by line.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return translate_line(model, lines[0])
    return "\n".join(translate_line(model, line) for line in lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.write(
        "This app runs a custom sequence-to-sequence model with "
        "Bahdanau attention, trained from scratch in TensorFlow on a "
        "Korean–English parallel corpus."
    )
    st.markdown("### Try a sample")
    for level, sentence in SAMPLE_SENTENCES.items():
        if st.button(f"{level}: {sentence}", use_container_width=True):
            st.session_state["korean_input"] = sentence
    st.markdown("---")
    st.caption(f"Model directory: `{MODEL_DIR.name}/`")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">🌐 Korean → English Translator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Real-time translation powered by a Bahdanau-attention seq2seq model.</div>',
    unsafe_allow_html=True,
)

model = load_model()

if model is None:
    st.error(
        f"Couldn't find a saved model at `{MODEL_DIR}`.\n\n"
        "Unzip `translator_bundle.zip` from the notebook and make sure the "
        "`saved_model/` folder sits next to `app.py`."
    )
    st.stop()

if "korean_input" not in st.session_state:
    st.session_state["korean_input"] = ""

korean_text = st.text_area(
    "Korean text",
    key="korean_input",
    height=140,
    placeholder="예: 오늘 날씨가 정말 좋네요.",
    label_visibility="collapsed",
)


def clear_input():
    st.session_state["korean_input"] = ""
col1, col2 = st.columns([1, 3])
with col1:
    translate_clicked = st.button("Translate 🔁", type="primary", use_container_width=True)
with col2:
   st.button("Clear", use_container_width=True, on_click=clear_input)


if translate_clicked:
    if not korean_text.strip():
        st.warning("Please enter some Korean text first.")
    else:
        with st.spinner("Translating..."):
            start = time.time()
            translation = translate_text(model, korean_text)
            elapsed = time.time() - start

        safe_translation = html.escape(translation)
        safe_source = html.escape(korean_text.strip())

        st.markdown(
            f"""
            <div class="result-card">
                <h4>English Translation</h4>
                <div class="result-text">{safe_translation}</div>
                <div class="source-echo">🇰🇷 {safe_source}</div>
            </div>
            <div class="timing-note">Translated in {elapsed:.2f}s</div>
            """,
            unsafe_allow_html=True,
        )
