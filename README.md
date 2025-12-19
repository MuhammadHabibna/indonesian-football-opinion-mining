# 🇮🇩 Analisis Sentimen & Deteksi Sarkasme: Timnas Indonesia

Proyek ini bertujuan untuk menganalisis opini publik terhadap **Timnas Indonesia** (pemain, pelatih Shin Tae-yong, dan pundit) menggunakan pendekatan *Deep Learning* berbasis Transformer.

Fokus utama penelitian ini dibagi menjadi dua modul terpisah:
1.  **Sentiment Analysis:** Mengklasifikasikan komentar menjadi Positif atau Negatif.
2.  **Sarcasm Detection:** (Coming Soon) Mendeteksi nuansa sarkasme yang sering muncul dalam komentar sepak bola.

---

## 📊 Dataset Statistics & Funneling

Dataset yang digunakan mengalami penyusutan melalui tahapan *preprocessing* dan *filtering* untuk menjaga kualitas pelatihan. Berikut adalah alur data dari mentah hingga siap latih:

1.  **Raw Dataset:** `8,117` baris
    * *Kondisi:* Data mentah hasil scraping, masih kotor dan mengandung duplikat.
2.  **Post-Preprocessing:** `7,045` baris
    * *Proses:* Pembersihan emoji, normalisasi teks alay, dan penghapusan duplikat.
3.  **Final Binary Dataset:** `5,732` baris
    * *Proses:* **Drop Neutral**. Menghapus data dengan label 'Netral' (Ambigu) agar model fokus pada polaritas tegas (Positif vs Negatif).
    * *Usage:* Data inilah yang dibagi menjadi Train, Validation, dan Test.

---

## 📌 Modul 1: Sentiment Analysis

Pada tahap ini, dilakukan *benchmarking* terhadap varian model BERT (Base vs Large) dengan dua strategi data yang berbeda: **Imbalanced (Full Data)** dan **Balanced (Downsampling)**.

### 🧪 Eksperimen 1: Model Benchmark

Kami menambahkan kolom **Params** (Parameter) untuk melihat korelasi antara ukuran model dengan performanya.

#### Skenario A: Full Data (Tanpa Downsampling) 🏆
Menggunakan seluruh 5,732 data (Imbalanced). Strategi ini terbukti memberikan hasil terbaik.

| Model | Checkpoint | Size (Params) | Accuracy | F1-Macro |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IndoBERT Base P1** | `indobenchmark/indobert-base-p1` | **~110M** | **76.13%** | **66.11%** |
| IndoBERT Base IndoLEM | `indolem/indobert-base-uncased` | ~110M | 74.69% | 61.19% |
| XLM-RoBERTa Large | `xlm-roberta-large` | ~560M | 74.34% | 60.65% |
| IndoBERT Large (IndoNLU) | `indobenchmark/indobert-large-p2` | ~335M | 74.17% | 42.58% |

#### Skenario B: Balanced Data (Downsampling)
Menyamakan jumlah data Positif dan Negatif (Rasio 1:1). Meskipun F1-Score lebih seimbang, akurasi total turun signifikan karena banyaknya data yang dibuang.

| Model | Size (Params) | Accuracy | F1-Macro | Catatan Evaluasi |
| :--- | :--- | :--- | :--- | :--- |
| **IndoBERT Base P1** | **~110M** | **68.69%** | **68.68%** | Performa paling stabil & seimbang. |
| IndoBERT Base IndoLEM | ~110M | 64.65% | 64.65% | Konsisten, namun di bawah Base P1. |
| IndoBERT Large (IndoNLU)| ~335M | 63.97% | 63.91% | Berhasil belajar (tidak bias), tapi akurasi rendah. |
| XLM-RoBERTa Large | ~560M | 50.17% | 38.58% | **Model Collapse** (Data terlalu sedikit untuk 560M params). |

> **Analisis:**
> Model dengan ukuran raksasa (**XLM-R 560M** dan **IndoNLU 335M**) cenderung kesulitan jika dilatih dengan data sedikit (Skenario B) atau data tidak seimbang (Skenario A). Model **Base (110M)** terbukti lebih efisien dan stabil untuk ukuran dataset ini.

---

### ⚙️ Experimental Setup & Hyperparameters

Berikut adalah detail konfigurasi parameter. Kami membedakan strategi untuk **Model Base** (Ringan) dan **Model Large** (Berat) untuk mencegah *Out-Of-Memory*.

| Parameter | Base Models (P1 & IndoLEM) | Large Models (IndoNLU & XLM-R) | Penjelasan |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | `3e-5` | `1e-5` | LR diperkecil untuk model Large agar bobot tidak rusak. |
| **Batch Size** | `32` | `4` (accumulated to 32) | Model Large butuh batch kecil agar muat di GPU. |
| **Gradient Accum.** | `1` | `8` | Teknik simulasi batch besar pada memori terbatas. |
| **Dropout** | `0.1` | `0.2 - 0.3` | Regularisasi ekstra untuk mencegah overfitting di model Large. |
| **Max Seq Length** | `64` | `64` | Disesuaikan dengan distribusi panjang kata dataset. |

---

## 🔜 Modul 2: Sarcasm Detection (Next Phase)

Tahap selanjutnya akan membangun model khusus untuk **Sarkasme** menggunakan dataset yang sama, dengan pendekatan fine-tuning **IndoBERT Base** dan fitur linguistik kontekstual.

---

## 🛠️ Tech Stack
* **Python**
* **HuggingFace Transformers** (Model & Tokenizer)
* **PyTorch**
* **Scikit-Learn** (Evaluation Metrics)
* **Pandas & NumPy** (Data Processing)

---

## 👨‍💻 Author
Data Science Student | Finalist Gemastik 2025
Universitas Negeri Surabaya
