import os
import re
from pathlib import Path

import streamlit as st
import torch
import pandas as pd
import fitz
from transformers import AutoTokenizer, AutoModelForSequenceClassification

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("PAJAK_MODEL_DIR", APP_DIR / "model"))
MAX_LEN = 512
CONTENT_TOKENS = MAX_LEN - 2
LABELS = ["PKB", "BBNKB", "PBBKB", "ROKOK", "PAP", "PAB", "OPSEN_MBLB"]


def clean_text(text):
    # Samakan aturan ini dengan remove_noise() pada notebook p1
    # sebelum pengujian akhir agar preprocessing konsisten.
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.I)
    text = re.sub(r"\b(baca juga|advertisement|iklan)\b\s*:?", " ", text, flags=re.I)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.lower().strip()


def model_files_exist(folder):
    config = (folder / "config.json").exists()
    weights = any((folder / n).exists() for n in (
        "model.safetensors", "pytorch_model.bin",
        "model.safetensors.index.json", "pytorch_model.bin.index.json"
    ))
    tokenizer = any((folder / n).exists() for n in (
        "tokenizer.json", "vocab.txt", "tokenizer_config.json"
    ))
    return config and weights and tokenizer


@st.cache_resource(show_spinner="Memuat IndoBERT-p1...")
def load_model(folder_string):
    folder = Path(folder_string)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True)
    if int(model.config.num_labels) != len(LABELS):
        raise ValueError(
            f"Model memiliki {model.config.num_labels} label, "
            f"sedangkan aplikasi mengharapkan {len(LABELS)}. "
            "Periksa LABELS dan LABEL_MAP pada notebook."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def extract_pdf_text(pdf_bytes):
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"PDF tidak dapat dibuka: {exc}") from exc

    total_pages = pdf.page_count
    if total_pages == 0:
        pdf.close()
        raise ValueError("PDF tidak memiliki halaman.")

    pages, pages_with_text = [], 0
    try:
        for i in range(total_pages):
            page_text = pdf.load_page(i).get_text("text").strip()
            if page_text:
                pages.append(page_text)
                pages_with_text += 1
    finally:
        pdf.close()

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(
            "Teks tidak ditemukan. Dokumen mungkin PDF hasil scan/gambar. "
            "Gunakan PDF digital yang teksnya dapat diseleksi atau pilih Input Teks."
        )
    return text, total_pages, pages_with_text


def classify_text(text, tokenizer, model, device):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Teks kosong setelah pra-pemrosesan.")

    ids = tokenizer.encode(cleaned, add_special_tokens=False)
    if not ids:
        raise ValueError("Teks tidak menghasilkan token.")

    chunks = [ids[i:i + CONTENT_TOKENS] for i in range(0, len(ids), CONTENT_TOKENS)]
    weighted_sum = torch.zeros(len(LABELS), dtype=torch.float64)
    total_weight = 0

    for start in range(0, len(chunks), 8):
        batch_chunks = chunks[start:start + 8]
        features = [
            tokenizer.prepare_for_model(
                chunk, add_special_tokens=True, truncation=True,
                max_length=MAX_LEN, return_attention_mask=True
            )
            for chunk in batch_chunks
        ]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.inference_mode():
            probs = torch.softmax(model(**batch).logits, dim=-1).cpu().to(torch.float64)

        for prob, chunk in zip(probs, batch_chunks):
            weight = len(chunk)
            weighted_sum += prob * weight
            total_weight += weight

    avg = weighted_sum / max(total_weight, 1)
    results = [
        {"label": LABELS[i], "probability": float(avg[i])}
        for i in range(len(LABELS))
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return cleaned, len(chunks), len(ids), results


def display_results(cleaned, chunk_count, token_count, results, source):
    best = results[0]
    st.divider()
    st.subheader("Hasil Klasifikasi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Kategori Pajak Dominan", best["label"])
    c2.metric("Probabilitas Tertinggi", f"{best['probability'] * 100:.2f}%")
    c3.metric("Potongan Teks", str(chunk_count))

    st.success(
        f"Prediksi {source}: **{best['label']}** "
        f"dengan probabilitas agregat **{best['probability'] * 100:.2f}%**."
    )
    df = pd.DataFrame(results)
    df["probability_percent"] = (df["probability"] * 100).round(2)
    shown = df[["label", "probability_percent"]].rename(columns={
        "label": "Kategori Pajak", "probability_percent": "Probabilitas (%)"
    })
    st.subheader("Probabilitas Seluruh Kategori")
    st.dataframe(shown, hide_index=True, use_container_width=True)
    st.bar_chart(shown.set_index("Kategori Pajak")["Probabilitas (%)"])
    st.caption(
        f"Total token isi: {token_count:,}. Setiap potongan memiliki maksimum "
        f"{MAX_LEN} token termasuk token khusus."
    )
    with st.expander("Lihat teks setelah pra-pemrosesan"):
        st.text_area("Teks", cleaned, height=250, disabled=True)
    st.info(
        "Probabilitas merupakan keluaran relatif model, bukan jaminan kebenaran "
        "atau keputusan pajak resmi."
    )


st.set_page_config(page_title="Klasifikasi Pajak Daerah", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1120px;padding-top:2rem;padding-bottom:3rem;}
.hero {padding:1.5rem 1.7rem;border-radius:18px;
background:linear-gradient(115deg,#102a43 0%,#176b62 100%);
color:white;margin-bottom:1.2rem;}
.hero h1 {color:white;margin:0 0 .4rem 0;font-size:2rem;}
.hero p {color:#e4f1ef;margin:0;}
</style>
<div class="hero">
<h1>📊 Klasifikasi Jenis Pajak Daerah</h1>
<p>IndoBERT-p1 Domain-Specific</p>
</div>
""", unsafe_allow_html=True)
st.write("Pilih salah satu metode input untuk mengklasifikasikan teks ke dalam tujuh kategori pajak daerah.")

with st.sidebar:
    st.subheader("Informasi Model")
    st.markdown("**Model:** IndoBERT-p1 domain-specific")
    st.markdown("**Metode:** Continual pre-training (MLM) dan fine-tuning")
    st.markdown("**Panjang input:** Maksimum 512 token per potongan")
    st.divider()
    st.caption("Kategori yang dikenali:")
    for label in LABELS:
        st.write(f"• {label}")

if not model_files_exist(MODEL_DIR):
    st.error("File model IndoBERT-p1 belum ditemukan atau belum lengkap.")
    st.code(str(MODEL_DIR))
    st.info("Ekstrak isi model_domain_p1.zip langsung ke folder model. Pastikan config.json, file bobot, dan file tokenizer berada di dalam folder tersebut.")
else:
    tab_pdf, tab_text = st.tabs(["📄 Upload PDF", "✍️ Input Teks"])

    with tab_pdf:
        st.subheader("Klasifikasi dari PDF Digital")
        st.write("Unggah PDF digital yang teksnya dapat diseleksi atau disalin. PDF hasil scan tidak didukung.")
        uploaded = st.file_uploader("Pilih dokumen PDF", type=["pdf"], key="pdf_digital")
        if uploaded:
            st.success(f"Dokumen dipilih: {uploaded.name}")
            if st.button("Klasifikasikan PDF", type="primary", use_container_width=True):
                try:
                    with st.spinner("Mengekstrak teks dari PDF..."):
                        raw, pages, pages_with_text = extract_pdf_text(uploaded.getvalue())
                    with st.spinner("Memuat IndoBERT-p1 dan mengklasifikasikan dokumen..."):
                        tokenizer, model, device = load_model(str(MODEL_DIR))
                        cleaned, chunks, tokens, results = classify_text(raw, tokenizer, model, device)
                    st.success(f"Berhasil membaca teks pada {pages_with_text} dari {pages} halaman.")
                    display_results(cleaned, chunks, tokens, results, "dokumen PDF")
                except Exception as exc:
                    st.error("PDF belum berhasil diproses.")
                    st.exception(exc)

    with tab_text:
        st.subheader("Klasifikasi dari Teks Manual")
        st.write("Tempelkan teks berita atau dokumen perpajakan ke dalam kolom berikut.")
        manual = st.text_area("Masukkan teks", height=260, placeholder="Tempelkan teks di sini...")
        if st.button("Klasifikasikan Teks", type="primary", use_container_width=True):
            if not manual.strip():
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                try:
                    with st.spinner("Memuat IndoBERT-p1 dan mengklasifikasikan teks..."):
                        tokenizer, model, device = load_model(str(MODEL_DIR))
                        cleaned, chunks, tokens, results = classify_text(manual, tokenizer, model, device)
                    display_results(cleaned, chunks, tokens, results, "teks manual")
                except Exception as exc:
                    st.error("Teks belum berhasil diproses.")
                    st.exception(exc)

st.divider()
st.caption("Prototype penelitian — Klasifikasi Jenis Pajak Daerah Tingkat Provinsi menggunakan IndoBERT-p1 domain-specific.")
