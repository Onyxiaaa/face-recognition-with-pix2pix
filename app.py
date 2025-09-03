import streamlit as st
from PIL import Image
import numpy as np
import os
import pickle
from types import SimpleNamespace

# Impor library Deep Learning
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.nn.functional import cosine_similarity
from tensorflow import keras

# --- PERUBAHAN UTAMA 1: Definisikan path untuk semua model Anda ---
# Ganti path ini dengan lokasi file model Anda yang sebenarnya.
MODEL_PATHS = {
    "Pix2Pix Masker Sintetis": "model pix2pix/Model Pix2pix rekonstruksi wajah.h5",
    "Pix2Pix Masker Asli": "model pix2pix/masker asli.h5"  # <-- GANTI PATH INI
}
DB_PATH = "face_database.pkl"

# --- Konfigurasi Halaman Web ---
st.set_page_config(
    page_title="Sistem Rekonstruksi & Pengenalan Wajah",
    page_icon="🎭",
    layout="wide"
)


# --- Fungsi Caching untuk Memuat Model ---
# Fungsi ini sekarang menerima path sebagai argumen
@st.cache_resource
def load_generator_model(model_path):
    """Memuat model Generator Pix2Pix dari path file yang diberikan."""
    st.info(f"Memuat model Generator dari: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        st.success(f"Model '{os.path.basename(model_path)}' berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Error saat memuat model Generator: {e}")
        return None


@st.cache_resource
def load_face_models():
    """Memuat model pendukung (FaceNet dan MTCNN)."""
    st.info("Memuat model FaceNet & MTCNN...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        mtcnn = MTCNN(image_size=160, margin=25, keep_all=False, post_process=True, device=device)
        st.success("Model FaceNet & MTCNN berhasil dimuat!")
        return facenet, mtcnn, device
    except Exception as e:
        st.error(f"Error memuat model pendukung: {e}")
        return None, None, None


# --- Fungsi Utilitas ---
def preprocess_for_generator(pil_image):
    """Mempersiapkan gambar untuk input model Generator (Pix2Pix)."""
    import tensorflow as tf
    image_array = np.array(pil_image.convert('RGB'))
    image = tf.convert_to_tensor(image_array)
    image = tf.image.resize(image, [256, 256])
    image_normalized = (tf.cast(image, tf.float32) / 127.5) - 1
    return tf.expand_dims(image_normalized, 0)


def tensor2pil(image_tensor):
    """Mengubah tensor output kembali menjadi gambar PIL."""
    image = (image_tensor + 1) / 2
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def get_facenet_embedding(pil_image, facenet_model, device):
    """Mendapatkan embedding dari tensor wajah yang sudah di-crop oleh MTCNN."""
    try:
        face_tensor = mtcnn(pil_image)
        if face_tensor is not None:
            with torch.no_grad():
                embedding = facenet_model(face_tensor.unsqueeze(0).to(device))
                return embedding[0].cpu()
        return None
    except Exception:
        return None


# --- Manajemen Database ---
def load_database():
    """Memuat database wajah dari file."""
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_database(db):
    """Menyimpan database wajah ke file."""
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)


# --- Muat Model Pendukung dan Database di Awal ---
facenet, mtcnn, device = load_face_models()

# Gunakan st.session_state untuk menyimpan database agar bisa dimodifikasi
if 'face_db' not in st.session_state:
    st.session_state.face_db = load_database()

# --- Navigasi Halaman ---
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.radio("Pilih Halaman", ["Pengujian Rekonstruksi", "Pendaftaran Wajah", "Manajemen Database"])

# ==============================================================================
# --- HALAMAN PENDAFTARAN (CREATE) ---
# ==============================================================================
if page == "Pendaftaran Wajah":
    st.header("👤 Pendaftaran Wajah Baru")
    st.write("Daftarkan wajah Anda (tanpa masker) untuk dijadikan sebagai ground truth.")

    if not all([facenet, mtcnn]):
        st.error("Model pendukung (FaceNet/MTCNN) gagal dimuat. Pendaftaran tidak dapat dilakukan.")
        st.stop()

    nama_pengguna = st.text_input("Masukkan Nama Anda:")
    uploaded_photo = st.file_uploader("Unggah Foto Wajah **TANPA MASKER**", type=['jpg', 'jpeg', 'png'])

    if st.button("Daftarkan Wajah") and uploaded_photo and nama_pengguna:
        if nama_pengguna in st.session_state.face_db:
            st.error(f"Nama '{nama_pengguna}' sudah terdaftar!")
        else:
            with st.spinner("Memproses pendaftaran..."):
                pil_image = Image.open(uploaded_photo).convert('RGB')
                embedding = get_facenet_embedding(pil_image, facenet, device)

                if embedding is None:
                    st.error("Wajah tidak terdeteksi pada foto. Silakan gunakan foto lain yang lebih jelas.")
                else:
                    st.session_state.face_db[nama_pengguna] = {
                        'embedding': embedding,
                        'image': np.array(pil_image)
                    }
                    save_database(st.session_state.face_db)
                    st.success(f"Wajah '{nama_pengguna}' berhasil didaftarkan!")
                    st.image(pil_image, caption=f"Foto Terdaftar untuk {nama_pengguna}", width=256)

# ==============================================================================
# --- HALAMAN MANAJEMEN DATABASE (READ, UPDATE, DELETE) ---
# ==============================================================================
elif page == "Manajemen Database":
    st.header("🗃️ Manajemen Database Wajah")

    if not st.session_state.face_db:
        st.info("Database wajah kosong. Silakan daftarkan wajah terlebih dahulu.")
    else:
        st.write(f"Total data terdaftar: **{len(st.session_state.face_db)}**")

        for name, data in list(st.session_state.face_db.items()):
            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(data['image'], caption=f"Foto: {name}", width=150)

            with col2:
                new_name = st.text_input("Ubah Nama:", value=name, key=f"update_{name}")
                if st.button("Simpan Perubahan", key=f"save_{name}"):
                    if new_name != name:
                        if new_name in st.session_state.face_db:
                            st.error(f"Nama '{new_name}' sudah ada. Harap gunakan nama lain.", icon="⚠️")
                        else:
                            st.session_state.face_db[new_name] = st.session_state.face_db.pop(name)
                            save_database(st.session_state.face_db)
                            st.success(f"Nama berhasil diubah dari '{name}' menjadi '{new_name}'.")
                            st.experimental_rerun()

                if st.button("Hapus Data", key=f"delete_{name}", type="primary"):
                    del st.session_state.face_db[name]
                    save_database(st.session_state.face_db)
                    st.success(f"Data untuk '{name}' berhasil dihapus.")
                    st.experimental_rerun()

# ==============================================================================
# --- HALAMAN PENGUJIAN ---
# ==============================================================================
elif page == "Pengujian Rekonstruksi":
    st.header("🎭 Pengujian Rekonstruksi dan Pengenalan Wajah")
    st.write(
        "Unggah foto wajah **DENGAN MASKER**. Sistem akan merekonstruksi, mengenali, dan membandingkannya dengan ground truth dari database.")

    # --- PERUBAHAN UTAMA: Tambahkan pilihan model kembali di sidebar ---
    with st.sidebar:
        st.markdown("---")
        st.header("Pengaturan Pengujian")
        selected_model_name = st.selectbox("Pilih Model Rekonstruksi", list(MODEL_PATHS.keys()))

    selected_model_path = MODEL_PATHS[selected_model_name]

    # Muat model generator yang dipilih
    generator = None
    if os.path.exists(selected_model_path):
        generator = load_generator_model(selected_model_path)
    else:
        st.error(f"File model '{selected_model_path}' tidak ditemukan. Harap periksa path di dalam kode.")

    if not all([facenet, mtcnn]):
        st.error("Model pendukung (FaceNet/MTCNN) gagal dimuat. Pengujian tidak dapat dilakukan.")
        st.stop()

    masked_file = st.file_uploader("Unggah Gambar Wajah Bermasker", type=['jpg', 'jpeg', 'png'])

    if st.button("Proses Gambar") and masked_file:
        if not st.session_state.face_db:
            st.error("Database wajah kosong. Harap daftarkan wajah terlebih dahulu.")
        elif generator is None:
            st.error("Model Generator gagal dimuat. Proses tidak dapat dilanjutkan.")
        else:
            with st.spinner(f"Memproses dengan model '{selected_model_name}'..."):
                pil_masked = Image.open(masked_file).convert('RGB')

                processed_input = preprocess_for_generator(pil_masked)
                prediction = generator(processed_input, training=False)
                pil_predicted = tensor2pil(prediction[0].numpy())

                reconstructed_embedding = get_facenet_embedding(pil_predicted, facenet, device)

                if reconstructed_embedding is None:
                    st.error("Wajah tidak terdeteksi pada hasil rekonstruksi.")
                else:
                    best_match_name = None
                    best_match_score = 0.5

                    for name, data in st.session_state.face_db.items():
                        db_embedding = data['embedding']
                        similarity = cosine_similarity(reconstructed_embedding.unsqueeze(0),
                                                       db_embedding.unsqueeze(0)).item()
                        if similarity > best_match_score:
                            best_match_score = similarity
                            best_match_name = name

                    if best_match_name is None:
                        st.error("Wajah tidak dikenali. Tidak ada yang cocok di database.")
                    else:
                        st.success(f"Wajah Dikenali Sebagai: **{best_match_name}**")

                        pil_gt = Image.fromarray(st.session_state.face_db[best_match_name]['image'])
                        emb_gt = st.session_state.face_db[best_match_name]['embedding']

                        emb_masked = get_facenet_embedding(pil_masked, facenet, device)
                        sim_masked_vs_gt = None
                        if emb_masked is not None:
                            sim_masked_vs_gt = cosine_similarity(emb_masked.unsqueeze(0), emb_gt.unsqueeze(0)).item()

                        st.subheader("Hasil Perbandingan")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(pil_masked, caption="Input (Bermasker)", use_column_width=True)
                            st.metric("Similarity vs GT",
                                      f"{sim_masked_vs_gt:.4f}" if sim_masked_vs_gt is not None else "N/A")
                        with col2:
                            st.image(pil_predicted, caption=f"Rekonstruksi ({selected_model_name})",
                                     use_column_width=True)
                            delta = best_match_score - sim_masked_vs_gt if sim_masked_vs_gt is not None else None
                            st.metric("Similarity vs GT", f"{best_match_score:.4f}",
                                      delta=f"{delta:.4f}" if delta is not None else None)
                        with col3:
                            st.image(pil_gt, caption=f"Ground Truth ({best_match_name})", use_column_width=True)
                            st.metric("Similarity vs GT", "1.0000 (Target)")
