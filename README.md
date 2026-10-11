# Korean ↔ English Neural Machine Translation

A portfolio project that evolves from a classical **Seq2Seq + Attention** translator (trained from scratch) to a **bidirectional, fine-tuned multilingual Transformer** (NLLB-200), deployed as a Streamlit web app.

**Live demo:** https://korean-english-translator-jrwuk3bwsihlqv5mtvp5gn.streamlit.app/
**Fine-tuned model:** https://huggingface.co/ayashaaban/nllb-ko-en-finetuned

> **License note:** the model is derived from `facebook/nllb-200-distilled-600M`, released under **CC-BY-NC-4.0**. This project is for **non-commercial, educational/portfolio use only** and must not be presented as suitable for unrestricted commercial deployment.

---

## Project evolution

```
Seq2Seq (GRU) → + Attention → Pretrained Transformer → Transfer Learning
→ Bidirectional NMT → Evaluation & Error Analysis → Deployment
```

### Version 1 — Seq2Seq + Bidirectional GRU + Bahdanau Attention

Adapted from the TensorFlow "Neural machine translation with attention" tutorial (originally Spanish → English) and converted to Korean → English.

- Korean text → Hangul-preserving cleaning → word-level `TextVectorization` (8,000 Korean / 5,000 English tokens)
- Bidirectional GRU encoder → Bahdanau attention → GRU decoder → English
- Trained from scratch on ~20,000 Korean–English sentence pairs from OPUS-100

**Why it was limited:**

| Limitation | Effect |
|---|---|
| Word-level tokenization on an agglutinative language | Korean particles/endings create many surface forms; unseen forms become `[UNK]` |
| Small training set, no pretraining | All linguistic knowledge must be learned from ~20K pairs |
| Single-head attention | Limited capacity on longer/complex sentences |
| One direction only | EN→KO would need a separate model |

### Version 2 — Pretrained multilingual Transformer + fine-tuning

- **Model:** `facebook/nllb-200-distilled-600M` (one model, both directions, selected by language-code tokens `kor_Hang` / `eng_Latn`)
- **Tokenizer:** the model's own SentencePiece subword tokenizer (no hand-built vocabulary)
- **Transfer learning:** continue training the pretrained weights on our corpus instead of learning translation from scratch

The pretrained model was evaluated **before** fine-tuning, so any gain could be measured honestly.

---

## Data

- **Source:** OPUS-100 `en-ko` (via Hugging Face `datasets`), 20,000-pair sample
- **Cleaning (conservative, audited):** Unicode NFKC normalization, HTML-entity residue cleanup, control-character removal, whitespace normalization, removal of rows with no real content (Unicode-aware). Hangul, Hanja, Latin names, numbers and URLs are preserved.
- **Deduplication:** exact duplicate pairs removed *before* splitting
- **Leakage-safe split (90/5/5):** rows sharing a Korean source or an English target are grouped (connected components via union-find) and each group is assigned to a single split, so identical source or target sentences never appear in both training and evaluation. Overlap was verified programmatically.

## Training setup (Version 2)

| Setting | Value |
|---|---|
| Training data | Train split only, each pair used in **both** directions (KO→EN and EN→KO) |
| Max source / target length | 64 / 64 tokens |
| Learning rate | 2e-5 (conservative, to limit catastrophic forgetting) |
| Epochs | up to 3, early stopping (patience 1) |
| Batch size | 4 per device × 8 accumulation steps = 32 effective |
| Precision / memory | FP16, gradient checkpointing (Tesla T4, 14.6 GB) |
| Model selection | Best validation loss; test set never used for tuning |
| Best checkpoint | `checkpoint-3330`, validation loss ≈ 2.2818 |
| Training time | ≈ 102 minutes |

Validation loss per epoch: 2.2977 → 2.2847 → 2.2818.

## Results

Evaluated on the **same frozen 500-example subset** of the held-out test split (never resampled), with SacreBLEU and chrF:

| Model | KO→EN BLEU | KO→EN chrF | EN→KO BLEU | EN→KO chrF |
|---|---:|---:|---:|---:|
| Pretrained NLLB (before fine-tuning) | 15.97 | 33.81 | 3.88 | 17.08 |
| Fine-tuned NLLB | 15.93 | 33.48 | **5.10** | **18.48** |
| Change | −0.05 | −0.33 | **+1.22** | **+1.40** |

**Summary:** fine-tuning improved the weaker EN→KO direction while leaving KO→EN essentially unchanged. A sentence-level error analysis (improvement/degradation distribution, length effects, under/over-translation diagnostics) is the next planned step and is not part of the current notebook.

### Limitations

- EN→KO remains much weaker than KO→EN in absolute terms.
- BLEU and chrF are surface-form metrics and do not directly measure meaning.
- The evaluation set is 500 examples from a single corpus (OPUS-100); results may not generalize to other domains.
- No statistical significance test was run; the deltas are point estimates.
- Version 1 was not scored on this frozen set, so the table compares pretrained vs. fine-tuned NLLB only.

---

## Web app

A Streamlit app with source/target language selectors and a **Swap** button. Model weights are hosted on the Hugging Face Hub (not in this repository) and downloaded at startup.

**Deployment notes**

- Streamlit Community Cloud's free tier has roughly 1 GB RAM, which is tight for a ~2.4 GB model; the app may restart under memory pressure. Hugging Face Spaces is the fallback hosting option.
- The first load after inactivity is slower because the model is downloaded again.

### Run locally

```bash
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model (~2.4 GB).

## Repository structure

```
korean-english-translator/
├── app.py                              # Streamlit app (Version 2)
├── requirements.txt
├── README.md
└── notebooks/
    ├── v1_seq2seq_attention.ipynb      # Version 1 (baseline)
    └── v2_transformer_nmt.ipynb        # Version 2 (fine-tuning, evaluation, error analysis)
```

## Future improvements

- Sentence-level error analysis of pretrained vs. fine-tuned outputs (where and why fine-tuning helped or hurt)
- Improve EN→KO further (data quality filtering, directional sampling, lower learning rate / fewer epochs, evaluated on the same frozen set)
- Larger and more diverse training data
- Human evaluation alongside BLEU/chrF
- Statistical significance testing of metric differences
- Production hosting with more memory (e.g. Hugging Face Spaces or an inference endpoint)

## Acknowledgements

- Version 1 is based on the TensorFlow "Neural machine translation with attention" tutorial.
- Base model: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC-4.0).
- Data: [OPUS-100](https://opus.nlpl.eu/).
