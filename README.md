# 🇮🇩 Analisis Sentimen & Deteksi Sarkasme: Timnas Indonesia

Proyek ini bertujuan untuk menganalisis opini publik terhadap **Timnas Indonesia** (pemain, pelatih Shin Tae-yong, dan pundit) menggunakan pendekatan *Deep Learning* berbasis Transformer.

Fokus utama penelitian ini dibagi menjadi dua modul terpisah:
1.  **Sentiment Analysis:** Mengklasifikasikan komentar menjadi Positif atau Negatif.
2.  **Sarcasm Detection:** (Coming Soon) Mendeteksi nuansa sarkasme yang sering muncul dalam komentar sepak bola.

---

## 📌 Modul 1: Sentiment Analysis

Pada tahap ini, kami melakukan *benchmarking* mendalam dengan dua strategi data: **Imbalanced (Full Data)** dan **Balanced (Downsampling)**, untuk menemukan model terbaik.

### 🧪 Eksperimen 1: Strategi Data & Model Selection

Kami membandingkan performa 4 model Transformer dengan dua skenario pelatihan.

#### Skenario A: Full Data (Tanpa Downsampling) 🏆
Strategi ini menggunakan seluruh data yang tersedia. Hasil menunjukkan model mampu menangkap pola lebih baik karena jumlah data yang lebih banyak.

| Model | Checkpoint | Accuracy | F1-Macro | Status |
| :--- | :--- | :--- | :--- | :--- |
| **IndoBERT Base P1** | `indobenchmark/indobert-base-p1` | **77.13%** | **63.28%** | ✅ **Selected Best Model** |
| IndoBERT Base IndoLEM | `indolem/indobert-base-uncased` | 76.09% | 65.05% | 🥈 Runner Up |
| XLM-RoBERTa Large | `xlm-roberta-large` | 74.34% | 60.65% | Good but Heavy |
| IndoBERT Large (IndoNLU) | `indobenchmark/indobert-large-p2` | 74.17% | 42.58% | ❌ Failed (Majority Bias) |

#### Skenario B: Balanced Data (Downsampling)
Strategi ini menyamakan jumlah data Positif dan Negatif (Rasio 1:1). Meskipun F1-Score lebih seimbang, terjadi penurunan Akurasi yang signifikan karena berkurangnya jumlah data latih.

| Model | Accuracy | F1-Macro | Catatan Evaluasi |
| :--- | :--- | :--- | :--- |
| **IndoBERT Base P1** | **68.69%** | **68.68%** | Performa paling stabil & seimbang. |
| IndoBERT Base IndoLEM | 64.65% | 64.65% | Konsisten, namun di bawah Base P1. |
| IndoBERT Large (IndoNLU)| 63.97% | 63.91% | Berhasil belajar (tidak bias), tapi akurasi rendah. |
| XLM-RoBERTa Large | 50.17% | 38.58% | **Model Collapse** akibat data terlalu sedikit. |

> **Kesimpulan Riset:**
> Kami memutuskan menggunakan **Skenario A (Full Data)** dengan model **IndoBERT Base P1**. Meskipun data tidak seimbang, model ini menghasilkan generalisasi yang lebih baik (Akurasi 77%) dibandingkan Skenario B (Akurasi 68%) yang kehilangan terlalu banyak informasi akibat pemangkasan data.

---

### ⚙️ Metodologi Final (Best Run)
* **Model:** IndoBERT Base P1
* **Problem Type:** Binary Classification (Positive vs Negative).
* **Handling Netral:** Data kelas 'Netral' dieliminasi untuk ketegasan polaritas.
* **Preprocessing:**
    * Custom Emoji Mapping (e.g., 🤡 -> "badut").
    * Normalization (Slang/Typo correction).
    * Minimal Cleaning (mempertahankan tanda baca `?!` untuk konteks emosi).
* **Hyperparameters:**
    * Learning Rate: `3e-5`
    * Batch Size: `32`
    * Max Sequence Length: `64` tokens
    * Epochs: 10 (Early Stopping Enabled)

---

## 🔜 Modul 2: Sarcasm Detection (Next Phase)

Mengingat tingginya penggunaan bahasa sarkas dalam komentar sepak bola (contoh: *"Mainnya bagus banget, sampe ngantuk nontonnya"*), modul analisis sentimen biasa seringkali terkecoh.

Tahap selanjutnya dari penelitian ini akan membangun model khusus untuk **Sarkasme** menggunakan dataset yang sama, dengan pendekatan:
1.  Fine-tuning **IndoBERT Base** (karena terbukti performanya di tahap sentimen).
2.  Fokus pada fitur linguistik kontekstual (tanda baca berlebih, emoji kontradiktif).

---

## 🛠️ Tech Stack
* **Python**
* **HuggingFace Transformers** (Model & Tokenizer)
* **PyTorch**
* **Scikit-Learn** (Evaluation Metrics)
* **Pandas & NumPy** (Data Processing)

---

## 👨‍💻 Author
Data Science Student | Finalist Gemastik Data Mining 2025
Universitas Negeri Surabaya