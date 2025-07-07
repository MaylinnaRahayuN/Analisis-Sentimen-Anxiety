# 🎓 Analisis Sentimen di Media Sosial Menggunakan TF-IDF dan H2O Gradient Boosting untuk Deteksi Kecemasan (Anxiety) Siswa

Sebuah aplikasi berbasis web yang dibangun menggunakan framework **Streamlit** dan bahasa pemrograman **Python**. Sistem ini bertujuan untuk mendeteksi tingkat kecemasan siswa dari unggahan media sosial menggunakan teknik **TF-IDF Vectorization** dan algoritma **H2O Gradient Boosting**.

📄 Artikel ini telah dipublikasikan pada:
👉 [Scientific Journal of Informatics - Universitas Negeri Semarang](https://journal.unnes.ac.id/journals/sji/article/view/20582/3318)

---

## 🚀 Fitur Aplikasi

Aplikasi ini memiliki beberapa menu yang dapat diakses melalui sidebar:

### 1. Home
Menampilkan informasi umum dan pengantar mengenai sistem analisis sentimen yang dikembangkan.

---

### 2. Upload Dataset
- Mendukung file dengan format `.xlsx` dan `.csv`.
- Setelah diunggah, isi dataset akan langsung ditampilkan.
- Dataset yang diunggah akan disimpan untuk tahap-tahap berikutnya.

---

### 3. Preprocess Data
Menampilkan hasil preprocessing data:
- **Text cleaning** (lowercase, remove punctuation)
- **Tokenization**
- **Stopword removal dan stemming**
- **Visualisasi WordCloud**
- **Top 20 Bigrams**
- Menampilkan dataframe yang berisi kolom: original text, label, cleaned text, tokens, token processed.

---

### 4. EDA (Exploratory Data Analysis)
Menampilkan:
- Statistik dasar dari dataset
- Visualisasi distribusi label (misal: batang/grafik pie)

---

### 5. TF-IDF
- Menampilkan hasil vektorisasi TF-IDF dalam bentuk tabel.
- Tabel dapat diunduh langsung oleh pengguna.

---

### 6. Modelling
- Melakukan pelatihan model menggunakan algoritma **H2O Gradient Boosting** dengan **5-Fold Cross Validation**.
- Menampilkan metrik evaluasi dari setiap fold.
- Menampilkan:
  - **Confusion Matrix**
  - **Akurasi, Precision, Recall, F1-Score**
  - **Error Rate, MSE, RMSE**

---

## 🗂️ Struktur Folder/Modul
```
📁 pages/
├── 📄 1_Home.py
├── 📄 2_Upload_Dataset.py
├── 📄 3_Preprocessing.py
├── 📄 4_EDA.py
├── 📄 5_TF_IDF.py
├── 📄 6_Modelling.py

📄 app.py
```

## ▶️ Cara Menjalankan Aplikasi

1. Pastikan kamu sudah menginstall semua dependensi:
    ```bash
    pip install -r requirements.txt
    ```

2. Jalankan aplikasi dengan Streamlit:
    ```bash
    streamlit run app.py
    ```

---

## 🧠 Teknologi yang Digunakan

- Python
- Streamlit
- Pandas
- Scikit-learn
- H2O.ai
- Matplotlib / Seaborn / WordCloud

---

## 📌 Catatan

- Proyek ini merupakan bagian dari penelitian deteksi kecemasan siswa menggunakan teknik NLP dan machine learning.
- Semua proses dilakukan secara end-to-end, mulai dari upload dataset hingga evaluasi model.

---

