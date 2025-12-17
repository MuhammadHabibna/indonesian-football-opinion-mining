# 🇮🇩 Analisis Sentimen & Deteksi Sarkasme: Timnas Indonesia

Proyek ini bertujuan untuk menganalisis opini publik terhadap **Timnas Indonesia** (pemain, pelatih Shin Tae-yong, dan pundit) menggunakan pendekatan *Deep Learning* berbasis Transformer.

Fokus utama penelitian ini dibagi menjadi dua modul terpisah:
1.  **Sentiment Analysis:** Mengklasifikasikan komentar menjadi Positif atau Negatif.
2.  **Sarcasm Detection:** (Coming Soon) Mendeteksi nuansa sarkasme yang sering muncul dalam komentar sepak bola.

---

## 📌 Modul 1: Sentiment Analysis

Pada tahap ini, kami melakukan *benchmarking* terhadap beberapa varian model BERT untuk menyelesaikan masalah *Binary Classification* (Positif vs Negatif) pada dataset komentar media sosial yang *imbalanced* (tidak seimbang).

### 🏆 Model Selection Benchmark
Berikut adalah hasil eksperimen pelatihan model tanpa melakukan *downsampling* data mayoritas:

| Model | Checkpoint | Accuracy (Val) | F1-Macro (Val) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **IndoBERT Base P1** | `indobenchmark/indobert-base-p1` | **77.13%** | 63.28% | ✅ **Selected Best Model** |
| IndoBERT Base IndoLEM | `indolem/indobert-base-uncased` | 76.09% | **65.05%** | 🥈 Runner Up |
| XLM-RoBERTa Large | `xlm-roberta-large` | 74.34% | 60.65% | Underfitting |
| IndoBERT Large (IndoNLU) | `indobenchmark/indobert-large-p2` | 74.17% | 42.58% | ❌ Failed (Majority Bias) |

> **Catatan:**
> * **IndoBERT Base P1** dipilih karena memberikan keseimbangan terbaik antara Akurasi dan stabilitas pelatihan.
> * **IndoBERT Large** gagal konvergensi (F1 stuck di 42%), menunjukkan ketidakmampuan model besar menangani *imbalance* data pada dataset berukuran kecil/sedang tanpa *downsampling*.

### ⚙️ Metodologi Training
* **Problem Type:** Binary Classification (Positive vs Negative).
* **Handling Netral:** Data kelas 'Netral' dieliminasi karena ambiguitas tinggi, fokus pada polaritas tegas.
* **Preprocessing:**
    * Custom Emoji Mapping (e.g., 🤡 -> "badut").
    * Normalization (Slang/Typo correction).
    * Minimal Cleaning (mempertahankan tanda baca `?!` untuk konteks emosi).
* **Hyperparameters (Best Run):**
    * Learning Rate: `3e-5`
    * Batch Size: `32`
    * Max Sequence Length: `64` tokens
    * Epochs: 10 (with Early Stopping)

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
Data Science Student | Google Student Ambassador 2025-2026
Universitas Negeri Surabaya