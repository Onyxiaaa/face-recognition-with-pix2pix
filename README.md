# 🎭 Sistem Rekonstruksi & Pengenalan Wajah dengan Masker

Aplikasi web interaktif yang dibangun dengan Streamlit untuk merekonstruksi bagian wajah yang tertutup masker dan melakukan verifikasi identitas berdasarkan hasil rekonstruksi tersebut.



## ✨ Fitur Utama

-   **Rekonstruksi Wajah**: Menggunakan model Generative Adversarial Network (GAN) berbasis arsitektur Pix2Pix untuk "menggambar ulang" bagian wajah yang tertutup masker.
-   **Dua Skenario**:
    1.  **Masker Sintetis**: Mensimulasikan pemakaian masker (berbagai jenis dan warna) pada wajah yang tidak bermasker untuk pengujian terkontrol.
    2.  **Masker Asli**: Menerima input gambar wajah yang benar-benar memakai masker.
-   **Pengenalan Wajah**: Memanfaatkan model FaceNet untuk mengekstrak *embedding* (fitur wajah) dari hasil rekonstruksi dan membandingkannya dengan database menggunakan *cosine similarity*.
-   **Manajemen Database**: Antarmuka sederhana untuk mendaftarkan wajah baru (sebagai *ground truth*), mengubah nama, dan menghapus data dari database.

---

## 🛠️ Teknologi yang Digunakan

-   **Framework Aplikasi Web**: [Streamlit](https://streamlit.io/)
-   **Deep Learning**: [TensorFlow (Keras)](https://www.tensorflow.org/) & [PyTorch](https://pytorch.org/)
-   **Pengenalan Wajah**: [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch)
-   **Deteksi & Manipulasi Wajah**: [OpenCV](https://opencv.org/), [Dlib](http://dlib.net/)
-   **Manipulasi Data**: [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/)

---

## 🚀 Instalasi dan Pengaturan

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut.

### 1. Prasyarat

-   Python 3.8+
-   `pip` (Python package installer)
-   Git

### 2. Clone Repository

```bash
git clone https://github.com/Onyxiaaa/face-recognition-with-pix2pix.git
cd face-recognition-with-pix2pix
```

### 3. Buat Virtual Environment (Sangat Direkomendasikan)

-   **Windows**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
-   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 4. Instal Dependensi

Instal semua pustaka yang diperlukan menggunakan file `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 5. Siapkan Model dan Direktori

Pastikan Anda memiliki struktur direktori dan file model seperti berikut di dalam folder proyek Anda. Aplikasi ini tidak akan berjalan tanpa file-file ini.

```
.
├── model pix2pix/
│   ├── Model Pix2pix rekonstruksi wajah.h5
│   └── masker asli.h5
│
├── dlib_models/
│   └── shape_predictor_68_face_landmarks.dat  <-- (Akan diunduh otomatis saat pertama kali dijalankan jika tidak ada)
│
├── utils/
│   └── aux_functions.py                       <-- (Pastikan file dari MaskTheFace ada di sini)
│
├── app.py                                     <-- (File kode utama Anda)
├── simulasi.py                                <-- (File kode utama Anda)
├── requirements.txt
└── README.md
```

**Catatan Penting**: Anda harus menyediakan file model `.h5` dan meletakkannya di dalam direktori `model pix2pix/`.

---

## 🏃 Cara Menjalankan Aplikasi

Setelah semua dependensi terinstal dan model sudah disiapkan, jalankan aplikasi Streamlit dengan perintah berikut di terminal:

```bash
streamlit run app.py
```

```bash
streamlit run simulasi.py
```

Aplikasi akan terbuka secara otomatis di browser default Anda.

---

## 📖 Cara Penggunaan

1.  **Pendaftaran Wajah**: Buka halaman **"Pendaftaran Wajah"**. Masukkan nama Anda dan ambil foto **tanpa masker** menggunakan webcam. Foto ini akan menjadi referensi di database.
2.  **Manajemen Database**: Halaman ini memungkinkan Anda untuk melihat semua data yang terdaftar, mengubah nama, atau menghapus data.
3.  **Pengujian Rekonstruksi**:
    -   Pilih skenario di sidebar: **"Simulasi Masker Sintetis"** atau **"Uji Masker Asli"**.
    -   Ambil foto sesuai instruksi (dengan atau tanpa masker).
    -   Sistem akan memproses gambar, menampilkan hasil rekonstruksi, dan mencoba mengenali wajah Anda berdasarkan data di database.