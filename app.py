"""
app.py — Streamlit Web App: Prediksi Harga WTI
Deploy gratis di: https://streamlit.io/cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import gdown  # download model dari Google Drive

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Minyak WTI",
    page_icon="🛢️",
    layout="wide"
)

# ── Konstanta ─────────────────────────────────────────────────────
LOOKBACK = 30
NUMERIC_FEATURES = [
    "brent_price", "dxy_index", "vix", "gpr_index",
    "wti_return", "brent_return",
    "wti_lag_1", "wti_lag_3", "wti_lag_7",
    "wti_volatility_7d", "wti_volatility_30d",
    "brent_wti_spread", "event_severity", "event_flag",
]
SENTIMENT_FEATURES = ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]
HYBRID_FEATURES    = NUMERIC_FEATURES + SENTIMENT_FEATURES

# ── Download model dari Google Drive (hanya sekali) ───────────────
@st.cache_resource
def download_and_load_models():
    """
    Download model dari Google Drive jika belum ada.
    Ganti FILE_IDs di bawah dengan ID file Anda sendiri.
    Cara dapat ID: klik kanan file di Drive → Share → salin ID dari link.
    """
    os.makedirs("models", exist_ok=True)

    files = {
        "models/hybrid_finbert_lstm_final.h5": "1_cdAwU4iGgu0STf-xM4WyvPBnxpuygOw",
        "models/scaler_hybrid.pkl":            "1KBvZr9caP5FoD2eQyvAr0Kq250unreKR",
        "models/target_scaler_hybrid.pkl":     "14pGlfF1P0dVQ5wAPBAJ9lE8K9uDU6hh8",
        "data/dataset_with_sentiment.csv":     "1ZFi1TTUuSGC9Bukz16fnCfJmZLo4GOFX",
    }

    for path, file_id in files.items():
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)

    # Load model
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model      = tf.keras.models.load_model("models/hybrid_finbert_lstm_final.h5")
    scaler     = joblib.load("models/scaler_hybrid.pkl")
    tgt_scaler = joblib.load("models/target_scaler_hybrid.pkl")
    df         = pd.read_csv("data/dataset_with_sentiment.csv", parse_dates=["date"])

    return model, scaler, tgt_scaler, df


@st.cache_resource
def load_finbert():
    """Load FinBERT — di-cache agar tidak reload setiap interaksi."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.nn.functional import softmax

    MODEL_NAME = "ProsusAI/finbert"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    finbert    = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    finbert.eval()
    return tokenizer, finbert, finbert.config.id2label


def get_sentiment(text, tokenizer, finbert, id2label):
    import torch
    from torch.nn.functional import softmax

    if not text.strip():
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    enc = tokenizer(text, max_length=512, truncation=True,
                    padding=True, return_tensors="pt")
    with torch.no_grad():
        probs = softmax(finbert(**enc).logits, dim=-1).numpy()[0]

    labels = [id2label[i].lower() for i in range(len(id2label))]
    return {
        "positive": float(probs[labels.index("positive")]),
        "negative": float(probs[labels.index("negative")]),
        "neutral":  float(probs[labels.index("neutral")]),
    }


def predict_wti(df, event_text, model, scaler, tgt_scaler, tokenizer, finbert, id2label):
    feats = [f for f in HYBRID_FEATURES if f in df.columns]
    window = df[feats].tail(LOOKBACK).values.astype(float)

    if len(window) < LOOKBACK:
        st.error(f"Data historis kurang dari {LOOKBACK} hari.")
        return None, None

    # Update sentimen di baris terakhir
    sent = get_sentiment(event_text, tokenizer, finbert, id2label)
    for col, key in [("sentiment_positive","positive"),
                     ("sentiment_negative","negative"),
                     ("sentiment_neutral","neutral")]:
        if col in feats:
            idx = feats.index(col)
            window[-1, idx] = sent[key]

    # Scale
    n = len(feats)
    dummy = np.zeros((LOOKBACK, 1))
    X_with_dummy = np.hstack([window, dummy])
    try:
        X_scaled = scaler.transform(X_with_dummy)[:, :n]
    except Exception:
        from sklearn.preprocessing import MinMaxScaler
        X_scaled = MinMaxScaler().fit_transform(window)

    X_input   = X_scaled.reshape(1, LOOKBACK, n)
    y_scaled  = model.predict(X_input, verbose=0)[0, 0]
    y_actual  = tgt_scaler.inverse_transform([[y_scaled]])[0, 0]
    return y_actual, sent


# ════════════════════════════════════════════════════════════════
# UI STREAMLIT
# ════════════════════════════════════════════════════════════════
st.title("🛢️ Prediksi Harga Minyak Mentah WTI")
st.caption("Model: Hybrid FinBERT-LSTM | Sentimen Peristiwa Geopolitik")

# Sidebar
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini memprediksi harga minyak WTI hari berikutnya "
    "menggunakan model **Hybrid FinBERT-LSTM** yang menggabungkan "
    "data historis harga dan sentimen peristiwa geopolitik."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** `ProsusAI/finbert` + LSTM Stacked")
st.sidebar.markdown("**Lookback:** 30 hari")
st.sidebar.markdown("**Dataset:** 2010–2026")

# Load model
with st.spinner("Memuat model... (pertama kali mungkin 1–2 menit)"):
    try:
        model, scaler, tgt_scaler, df = download_and_load_models()
        tokenizer, finbert, id2label  = load_finbert()
        st.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# ── Tab layout ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Data Historis", "📈 Performa Model"])

# ─── TAB 1: PREDIKSI ─────────────────────────────────────────────
with tab1:
    st.subheader("Prediksi Harga WTI Hari Berikutnya")

    col1, col2 = st.columns([2, 1])
    with col1:
        event_text = st.text_area(
            "📰 Deskripsi Peristiwa Geopolitik (opsional)",
            placeholder="Contoh: OPEC agrees to cut oil production by 1 million barrels per day",
            height=100
        )
    with col2:
        st.markdown("**Contoh peristiwa:**")
        examples = [
            "Russia cuts oil exports amid new sanctions",
            "OPEC+ increases production quota",
            "US-Iran tensions escalate in Strait of Hormuz",
            "Saudi Arabia announces surprise production cut",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                event_text = ex

    if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
        with st.spinner("Memproses sentimen dan menghitung prediksi..."):
            y_pred, sent = predict_wti(
                df, event_text, model, scaler, tgt_scaler,
                tokenizer, finbert, id2label
            )

        if y_pred is not None:
            last_price = df["wti_price"].iloc[-1]
            delta      = y_pred - last_price
            delta_pct  = (delta / last_price) * 100

            # Metrik utama
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediksi WTI", f"${y_pred:.2f}", f"{delta:+.2f} ({delta_pct:+.1f}%)")
            m2.metric("Harga Terakhir", f"${last_price:.2f}")
            m3.metric("Arah", "📈 Naik" if delta > 0 else "📉 Turun")

            # Sentimen
            st.markdown("---")
            st.markdown("**🧠 Analisis Sentimen FinBERT**")
            s1, s2, s3 = st.columns(3)
            s1.metric("Positif",  f"{sent['positive']:.1%}")
            s2.metric("Negatif",  f"{sent['negative']:.1%}")
            s3.metric("Netral",   f"{sent['neutral']:.1%}")

            dominant = max(sent, key=sent.get)
            color_map = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
            st.info(f"Sentimen dominan: {color_map[dominant]} **{dominant.upper()}**")

            # Plot 30 hari terakhir + prediksi
            st.markdown("---")
            st.markdown("**📈 Pergerakan 30 Hari Terakhir + Prediksi**")
            fig, ax = plt.subplots(figsize=(10, 4))
            hist = df["wti_price"].tail(30)
            ax.plot(range(len(hist)), hist.values, color="#1A5276",
                    lw=2, label="Historis")
            ax.scatter(len(hist), y_pred, color="#E74C3C", s=120,
                       zorder=5, label=f"Prediksi: ${y_pred:.2f}")
            ax.axhline(last_price, color="grey", lw=0.8, linestyle="--")
            ax.set_xlabel("Hari ke-")
            ax.set_ylabel("USD/barel")
            ax.set_title("Harga WTI — 30 Hari Terakhir & Prediksi Hari Berikutnya")
            ax.legend()
            ax.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)

# ─── TAB 2: DATA HISTORIS ─────────────────────────────────────────
with tab2:
    st.subheader("📊 Data Historis WTI")

    col_a, col_b = st.columns(2)
    with col_a:
        n_days = st.slider("Tampilkan N hari terakhir", 30, 500, 180)
    with col_b:
        show_brent = st.checkbox("Tampilkan Brent", value=True)

    df_plot = df.tail(n_days)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df_plot["date"], df_plot["wti_price"],
             label="WTI", color="#1A5276", lw=1.5)
    if show_brent and "brent_price" in df_plot.columns:
        ax2.plot(df_plot["date"], df_plot["brent_price"],
                 label="Brent", color="#E67E22", lw=1.2, alpha=0.8)
    ax2.set_ylabel("USD/barel")
    ax2.legend(); ax2.grid(alpha=0.25)
    plt.xticks(rotation=30); plt.tight_layout()
    st.pyplot(fig2)

    # Statistik ringkas
    st.markdown("**Statistik Deskriptif**")
    st.dataframe(df[["wti_price","brent_price","vix","gpr_index"]].describe().round(2))

# ─── TAB 3: PERFORMA MODEL ────────────────────────────────────────
with tab3:
    st.subheader("📈 Performa Model (Data Uji)")
    st.info("Upload file `model_comparison.csv` dari folder `outputs/results/` untuk melihat perbandingan metrik.")

    uploaded = st.file_uploader("Upload model_comparison.csv", type="csv")
    if uploaded:
        res = pd.read_csv(uploaded)
        st.dataframe(res.set_index("Model").style.highlight_min(
            subset=["MAE","RMSE","MAPE"], color="#d4edda"
        ).highlight_max(subset=["R2"], color="#d4edda"))

        fig3, axes = plt.subplots(1, 4, figsize=(14, 4))
        metrics = ["MAE","RMSE","MAPE","R2"]
        colors  = ["#1E8449","#2980B9","#8E44AD","#E67E22"]
        for ax, metric, color in zip(axes, metrics, colors):
            ax.bar(res["Model"], res[metric], color=color, edgecolor="white")
            ax.set_title(metric); ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
    else:
        st.markdown("""
        Tabel perbandingan akan muncul di sini setelah upload.  
        File tersebut dihasilkan otomatis oleh `03_Model_Training_Evaluation.py`.
        """)
