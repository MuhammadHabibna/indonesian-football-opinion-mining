# 🇮🇩 Analisis Sentimen & Deteksi Sarkasme: Timnas Indonesia

Proyek ini bertujuan untuk menganalisis opini publik terhadap **Timnas Indonesia** (pemain, pelatih Shin Tae-yong, dan pundit) menggunakan pendekatan *Deep Learning* berbasis Transformer.

Fokus utama penelitian ini dibagi menjadi dua modul terpisah:
1.  **Sentiment Analysis:** Mengklasifikasikan komentar menjadi Positif atau Negatif (Binary).
2.  **Sarcasm Detection:** (Coming Soon) Mendeteksi nuansa sarkasme yang sering muncul dalam komentar sepak bola.

---

## 📊 Data Pipeline & Statistics

Dataset yang digunakan mengalami penyusutan melalui tahapan *preprocessing* dan *filtering* untuk menjaga kualitas pelatihan. Berikut adalah "Data Funnel" dari data mentah hingga siap latih:

1.  **Raw Dataset:** `8,117` baris
    * *Kondisi:* Data mentah hasil scraping, mengandung duplikat, noise, dan label yang belum bersih.
2.  **Post-Preprocessing:** `7,045` baris
    * *Proses:* Pembersihan emoji, normalisasi teks alay, penghapusan duplikat, dan cleaning regex.
3.  **Final Binary Dataset:** `5,732` baris
    * *Proses:* **Drop Neutral**. Menghapus data dengan label 'Netral' (Ambigu) untuk memfokuskan model pada polaritas tegas (Positif vs Negatif).
    * *Usage:* Data inilah yang digunakan untuk training dan testing (Split 80:10:10).

---

## 📌 Modul 1: Sentiment Analysis

Pada tahap ini, kami melakukan *benchmarking* mendalam dengan dua strategi data: **Imbalanced (Full Data)** dan **Balanced (Downsampling)**.

### 🏆 Model Selection Benchmark
Tabel di bawah menunjukkan performa model menggunakan **Skenario A (Full Data)** yang terbukti memberikan hasil terbaik dibandingkan *Downsampling*.

| Model | Checkpoint | Accuracy | F1-Macro | Status |
| :--- | :--- | :--- | :--- | :--- |
| **IndoBERT Base P1** | `indobenchmark/indobert-base-p1` | **77.13%** | **63.28%** | ✅ **Selected Best Model** |
| IndoBERT Base IndoLEM | `indolem/indobert-base-uncased` | 76.09% | 65.05% | 🥈 Runner Up |
| XLM-RoBERTa Large | `xlm-roberta-large` | 74.34% | 60.65% | Good but Heavy |
| IndoBERT Large (IndoNLU) | `indobenchmark/indobert-large-p2` | 74.17% | 42.58% | ❌ Failed (Majority Bias) |

---

### ⚙️ Experimental Setup & Hyperparameters

Untuk tujuan pembelajaran, berikut adalah detail konfigurasi parameter yang digunakan. Kami menerapkan strategi yang berbeda untuk **Model Base** (Ringan) dan **Model Large** (Berat).

#### 1. Configuration for Base Models (IndoBERT P1 & IndoLEM)
Model tipe *Base* (~110M params) dilatih dengan konfigurasi standar yang agresif karena lebih stabil pada data sedikit.

| Parameter | Value | Alasan / Catatan |
| :--- | :--- | :--- |
| **Batch Size** | `32` | Ukuran batch optimal untuk kestabilan gradien. |
| **Learning Rate** | `3e-5` | *Sweet spot* untuk fine-tuning BERT Base. |
| **Gradient Accumulation** | `1` | Tidak diperlukan karena memori GPU cukup. |
| **Dropout** | `0.1` (Default) | Regularisasi standar. |
| **Max Sequence Length** | `64` | Cukup untuk menampung rata-rata panjang komentar. |

#### 2. Configuration for Large Models (IndoNLU Large & XLM-R)
Model tipe *Large* (~300M - 500M params) memerlukan teknik khusus untuk menghindari *Out-Of-Memory (OOM)* dan *Overfitting* parah.

| Parameter | Value | Alasan / Catatan |
| :--- | :--- | :--- |
| **Batch Size** | `4` (per device) | Dikecilkan ekstrem agar muat di VRAM (Colab T4). |
| **Gradient Accumulation** | `8` | Mengakumulasi gradien agar total batch size setara `32`. |
| **Learning Rate** | `1e-5` | LR lebih kecil agar bobot model besar tidak "rusak" drastis. |
| **Dropout** | `0.2` - `0.3` | **Ditingkatkan.** Mencegah model menghafal data (overfitting). |
| **Gradient Checkpointing** | `True` | Menghemat memori dengan komputasi ulang saat backward pass. |

---

## 🔜 Modul 2: Sarcasm Detection (Next Phase)

Mengingat tingginya penggunaan bahasa sarkas dalam komentar sepak bola, tahap selanjutnya akan membangun model khusus untuk **Sarkasme** menggunakan dataset yang sama.

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