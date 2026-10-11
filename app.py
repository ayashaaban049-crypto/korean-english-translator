"""
Korean <-> English Neural Machine Translator (Version 2)
---------------------------------------------------------
Streamlit front-end for a bidirectionally fine-tuned
facebook/nllb-200-distilled-600M model, hosted on the Hugging Face Hub.

NOTE: NLLB-200 weights are released under CC-BY-NC-4.0 (non-commercial use only).

Run with:  streamlit run app.py
"""

import html
import time

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration -- CHANGE MODEL_NAME to your own Hugging Face repo id
# ---------------------------------------------------------------------------
MODEL_NAME = "ayashaaban/nllb-ko-en-finetuned"

LANGUAGES = {
    "Korean": "kor_Hang",
    "English": "eng_Latn",
}
LANG_FLAGS = {"Korean": "🇰🇷", "English": "🇺🇸"}

MAX_SOURCE_LENGTH = 128
MAX_NEW_TOKENS = 128
NUM_BEAMS = 4

SAMPLES = {
    "Korean": ["안녕하세요.", "오늘 날씨가 정말 좋네요.", "저는 인공지능을 공부하고 있습니다."],
    "English": ["Hello.", "The weather is really nice today.", "I am studying artificial intelligence."],
}

st.set_page_config(page_title="Korean ↔ English Translator", page_icon="🌐", layout="centered")

st.markdown(
    """
    <style>
    .main .block-container { max-width: 760px; padding-top: 2.5rem; }
    .app-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.1rem; }
    .app-subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .result-card {
        background: linear-gradient(135deg, #f0f7ff 0%, #eef2ff 100%);
        border: 1px solid #dbe4ff; border-radius: 14px;
        padding: 1.4rem 1.6rem; margin-top: 1.2rem;
    }
    .result-card h4 {
        margin: 0 0 0.5rem 0; color: #4338ca; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    .result-text { font-size: 1.35rem; font-weight: 500; color: #1e1b4b;
                   line-height: 1.5; white-space: pre-wrap; }
    .source-echo { color: #6b7280; font-size: 0.9rem; margin-top: 0.6rem;
                   border-top: 1px dashed #dbe4ff; padding-top: 0.6rem; white-space: pre-wrap; }
    .timing-note { color: #9ca3af; font-size: 0.8rem; margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading (cached: loads once per server process, not per interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
    # max_new_tokens is the single length control at generation time.
    model.generation_config.max_length = None
    model.eval()
    return tokenizer, model


@torch.no_grad()
def translate_line(tokenizer, model, text, src_name, tgt_name):
    tokenizer.src_lang = LANGUAGES[src_name]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LENGTH)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANGUAGES[tgt_name]),
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def translate_text(tokenizer, model, text, src_name, tgt_name):
    """Translate multi-line input one line at a time (model was trained on sentences)."""
    lines = [line for line in text.split("\n") if line.strip()]
    return "\n".join(translate_line(tokenizer, model, line, src_name, tgt_name) for line in lines)


# ---------------------------------------------------------------------------
# State + callbacks (callbacks run BEFORE widgets are re-created, which avoids
# Streamlit's "cannot modify session_state after widget is instantiated" error)
# ---------------------------------------------------------------------------
st.session_state.setdefault("src_lang", "Korean")
st.session_state.setdefault("tgt_lang", "English")
st.session_state.setdefault("input_text", "")


def swap_languages():
    st.session_state["src_lang"], st.session_state["tgt_lang"] = (
        st.session_state["tgt_lang"], st.session_state["src_lang"])


def clear_input():
    st.session_state["input_text"] = ""


def use_sample(sentence):
    st.session_state["input_text"] = sentence


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Try a sample")
    for sample in SAMPLES[st.session_state["src_lang"]]:
        st.button(sample, use_container_width=True, on_click=use_sample, args=(sample,), key=f"sample_{sample}")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">🌐 Korean ↔ English Translator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Bidirectional translation with a fine-tuned multilingual Transformer.</div>',
    unsafe_allow_html=True,
)

if MODEL_NAME.startswith("YOUR_HF_USERNAME"):
    st.error("Set `MODEL_NAME` at the top of app.py to your Hugging Face model repo id.")
    st.stop()

with st.spinner("Loading translation model (first run can take a few minutes)..."):
    tokenizer, model = load_model()

col_from, col_swap, col_to = st.columns([5, 2, 5])
with col_from:
    st.selectbox("From", list(LANGUAGES), key="src_lang",
                 format_func=lambda x: f"{LANG_FLAGS[x]} {x}")
with col_swap:
    st.write("")
    st.button("⇄ Swap", on_click=swap_languages, use_container_width=True)
with col_to:
    st.selectbox("To", list(LANGUAGES), key="tgt_lang",
                 format_func=lambda x: f"{LANG_FLAGS[x]} {x}")

src_name, tgt_name = st.session_state["src_lang"], st.session_state["tgt_lang"]

text = st.text_area(
    "Text to translate", key="input_text", height=140,
    placeholder="Type or paste text here...", label_visibility="collapsed",
)

b1, b2 = st.columns([1, 3])
with b1:
    translate_clicked = st.button("Translate 🔁", type="primary", use_container_width=True)
with b2:
    st.button("Clear", use_container_width=True, on_click=clear_input)

if translate_clicked:
    if src_name == tgt_name:
        st.warning("Source and target languages are the same. Choose different languages.")
    elif not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Translating..."):
            start = time.time()
            translation = translate_text(tokenizer, model, text, src_name, tgt_name)
            elapsed = time.time() - start

        st.markdown(
            f"""
            <div class="result-card">
                <h4>{LANG_FLAGS[tgt_name]} {html.escape(tgt_name)} translation</h4>
                <div class="result-text">{html.escape(translation)}</div>
                <div class="source-echo">{LANG_FLAGS[src_name]} {html.escape(text.strip())}</div>
            </div>
            <div class="timing-note">Translated in {elapsed:.2f}s</div>
            """,
            unsafe_allow_html=True,
        )

st.caption("Model weights: CC-BY-NC-4.0 (non-commercial use only).")
