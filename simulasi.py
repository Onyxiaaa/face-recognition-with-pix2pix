import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2
import pickle
from types import SimpleNamespace

# Impor library Deep Learning
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.nn.functional import cosine_similarity
from tensorflow import keras

# Impor Dlib dan fungsi dari MaskTheFace
import dlib
from utils.aux_functions import mask_image, download_dlib_model

# --- Definisikan path untuk semua model Anda ---
MODEL_PATHS = {
    "Model Masker Sintetis": "model pix2pix/Model Pix2pix rekonstruksi wajah.h5",
    "Model Masker Asli": "model pix2pix/masker asli.h5"
}
DB_PATH = "face_database.pkl"

# --- Konfigurasi Halaman Web ---
st.set_page_config(
    page_title="Sistem Rekonstruksi & Pengenalan Wajah",
    page_icon="🎭",
    layout="wide"
)


# --- Fungsi Caching untuk Memuat Model ---
@st.cache_resource
def load_generator_model(model_path):
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
    st.info("Memuat model FaceNet, MTCNN, & Dlib...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        mtcnn = MTCNN(image_size=160, margin=25, keep_all=False, post_process=True, device=device)

        dlib_model_path = "dlib_models/shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(dlib_model_path):
            st.info("Model Dlib tidak ditemukan, mengunduh...")
            download_dlib_model()
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(dlib_model_path)

        st.success("Semua model pendukung berhasil dimuat!")
        return facenet, mtcnn, detector, predictor, device
    except Exception as e:
        st.error(f"Error memuat model pendukung: {e}")
        return None, None, None, None, None


# --- Fungsi Utilitas ---
def preprocess_for_generator(pil_image):
    import tensorflow as tf
    image_array = np.array(pil_image.convert('RGB'))
    image = tf.convert_to_tensor(image_array)
    image = tf.image.resize(image, [256, 256])
    image_normalized = (tf.cast(image, tf.float32) / 127.5) - 1
    return tf.expand_dims(image_normalized, 0)


def tensor2pil(image_tensor):
    image = (image_tensor + 1) / 2
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def get_facenet_embedding(pil_image, facenet_model, device):
    try:
        # Di sini MTCNN digunakan untuk preprocessing FaceNet
        face_tensor = mtcnn(pil_image)
        if face_tensor is not None:
            with torch.no_grad():
                embedding = facenet_model(face_tensor.unsqueeze(0).to(device))
                return embedding[0].cpu()
        return None
    except Exception:
        return None


def pad_to_square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def crop_and_square_face(pil_image, face_detector):
    # Fungsi ini sekarang bisa menerima detector Dlib atau MTCNN

    # Jika menggunakan Dlib
    if isinstance(face_detector, dlib.fhog_object_detector):
        cv2_img = np.array(pil_image.convert('RGB'))
        faces = face_detector(cv2_img, 1)
        if len(faces) == 0: return None
        d = faces[0]
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

    # Jika menggunakan MTCNN
    else:
        boxes, _ = face_detector.detect(pil_image)
        if boxes is None: return None
        x1, y1, x2, y2 = boxes[0]

    face_width, face_height = x2 - x1, y2 - y1
    center_x, original_center_y = x1 + face_width / 2, y1 + face_height / 2
    y_shift = int(face_height * 0.15)
    center_y = original_center_y - y_shift

    max_dim = max(face_width, face_height)
    box_size = int(max_dim * 2.0)

    img_w, img_h = pil_image.size
    new_x1 = int(center_x - box_size / 2)
    new_y1 = int(center_y - box_size / 2)
    new_x2 = new_x1 + box_size
    new_y2 = new_y1 + box_size

    if new_x1 < 0: new_x1 = 0
    if new_y1 < 0: new_y1 = 0
    if new_x2 > img_w: new_x2 = img_w
    if new_y2 > img_h: new_y2 = img_h

    cropped_pil = pil_image.crop((new_x1, new_y1, new_x2, new_y2))
    return pad_to_square(cropped_pil)


# --- Manajemen Database ---
def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_database(db):
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)


# --- Muat Model Pendukung dan Database di Awal ---
facenet, mtcnn, detector, predictor, device = load_face_models()

if 'face_db' not in st.session_state:
    st.session_state.face_db = load_database()

# --- Navigasi Halaman ---
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Pengujian Rekonstruksi", "Pendaftaran Wajah", "Manajemen Database"]
)

# ==============================================================================
# --- HALAMAN PENDAFTARAN (CREATE) ---
# ==============================================================================
if page == "Pendaftaran Wajah":
    st.header("👤 Pendaftaran Wajah Baru")
    st.write("Daftarkan wajah Anda (tanpa masker) untuk dijadikan sebagai ground truth.")

    if not all([facenet, mtcnn]):
        st.error("Model pendukung gagal dimuat. Pendaftaran tidak dapat dilakukan.")
        st.stop()

    nama_pengguna = st.text_input("Masukkan Nama Anda:")
    webcam_photo = st.camera_input("Ambil Foto Wajah **TANPA MASKER**")

    if webcam_photo is not None and nama_pengguna:
        if nama_pengguna in st.session_state.face_db:
            st.error(f"Nama '{nama_pengguna}' sudah terdaftar!")
        else:
            with st.spinner("Memproses pendaftaran..."):
                pil_image = Image.open(webcam_photo).convert('RGB')
                embedding = get_facenet_embedding(pil_image, facenet, device)

                if embedding is None:
                    st.error("Wajah tidak terdeteksi pada foto. Silakan coba lagi.")
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
                            st.error(f"Nama '{new_name}' sudah ada.", icon="⚠️")
                        else:
                            st.session_state.face_db[new_name] = st.session_state.face_db.pop(name)
                            save_database(st.session_state.face_db)
                            st.success(f"Nama diubah dari '{name}' menjadi '{new_name}'.")
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

    # --- PERUBAHAN UTAMA: Pilihan skenario di sidebar ---
    with st.sidebar:
        st.markdown("---")
        st.header("Pengaturan Pengujian")
        scenario = st.radio("Pilih Skenario Pengujian", ["Simulasi Masker Sintetis", "Uji Masker Asli"])

    # Tentukan model yang akan digunakan berdasarkan skenario
    if scenario == "Simulasi Masker Sintetis":
        selected_model_name = "Model Masker Sintetis"
    else:
        selected_model_name = "Model Masker Asli"

    selected_model_path = MODEL_PATHS[selected_model_name]
    generator = None
    if os.path.exists(selected_model_path):
        generator = load_generator_model(selected_model_path)
    else:
        st.error(f"File model '{selected_model_path}' tidak ditemukan.")

    if not all([facenet, mtcnn, detector, predictor]):
        st.error("Satu atau lebih model pendukung gagal dimuat. Pengujian tidak dapat dilakukan.")
        st.stop()

    # --- Logika Kondisional berdasarkan Skenario ---
    pil_masked = None
    pil_gt = None

    if scenario == "Simulasi Masker Sintetis":
        st.write("Ambil foto wajah Anda **TANPA MASKER**. Sistem akan mensimulasikan Anda memakai masker.")
        with st.sidebar:
            st.subheader("Pengaturan Masker Sintetis")
            mask_type = st.selectbox("Pilih Jenis Masker", ["surgical", "N95", "KN95", "cloth"])
            mask_color = st.color_picker("Pilih Warna Masker", "#FFFFFF")

        webcam_photo = st.camera_input("Ambil Foto Wajah TANPA MASKER untuk Simulasi")

        if webcam_photo is not None:
            if not st.session_state.face_db:
                st.error("Database wajah kosong. Harap daftarkan wajah terlebih dahulu.")
            elif generator is None:
                st.error("Model Generator gagal dimuat.")
            else:
                with st.spinner("Memproses simulasi..."):
                    pil_image = Image.open(webcam_photo).convert('RGB')
                    unmasked_face = crop_and_square_face(pil_image, detector)

                    if unmasked_face is None:
                        st.error("Wajah tidak terdeteksi.")
                    else:
                        temp_crop_path = "temp_unmasked_face.jpg"
                        unmasked_face.save(temp_crop_path)
                        args = SimpleNamespace(
                            mask_type=mask_type, detector=detector, predictor=predictor, verbose=False, code="",
                            pattern="", pattern_weight=0.5, color=mask_color, color_weight=0.5
                        )
                        masked_images_cv2, _, _, _ = mask_image(temp_crop_path, args)

                        if not masked_images_cv2:
                            st.error("Gagal menerapkan masker.")
                            st.stop()

                        pil_masked = Image.fromarray(cv2.cvtColor(masked_images_cv2[0], cv2.COLOR_BGR2RGB))
                        pil_gt = unmasked_face

    else:  # Skenario Masker Asli
        st.write("Ambil foto wajah Anda **DENGAN MASKER ASLI**.")
        webcam_photo = st.camera_input("Ambil Foto Wajah DENGAN MASKER")

        if webcam_photo is not None:
            if not st.session_state.face_db:
                st.error("Database wajah kosong. Harap daftarkan wajah terlebih dahulu.")
            elif generator is None:
                st.error("Model Generator gagal dimuat.")
            else:
                pil_masked = crop_and_square_face(Image.open(webcam_photo).convert('RGB'), mtcnn)
                if pil_masked is None:
                    st.error("Wajah tidak terdeteksi. Pastikan wajah terlihat jelas oleh kamera.")
                    st.stop()
                pil_gt = None

                # --- Proses Lanjutan (setelah input disiapkan) ---
    if pil_masked is not None:
        with st.spinner(f"Merekonstruksi dengan model '{selected_model_name}'..."):
            processed_input = preprocess_for_generator(pil_masked)
            prediction = generator(processed_input, training=False)
            pil_predicted = tensor2pil(prediction[0].numpy())

            reconstructed_embedding = get_facenet_embedding(pil_predicted, facenet, device)

            st.subheader("Hasil Perbandingan Visual")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(pil_masked, caption="Input (Bermasker)", use_column_width=True)
            with col2:
                st.image(pil_predicted, caption=f"Hasil Rekonstruksi ({selected_model_name})", use_column_width=True)

            if reconstructed_embedding is None:
                st.error("Wajah tidak terdeteksi pada hasil rekonstruksi. Tidak dapat melakukan pengenalan.")
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

                st.subheader("Hasil Pengenalan")
                if best_match_name is None:
                    st.error("Wajah tidak dikenali. Tidak ada yang cocok di database.")
                    with col3:
                        st.image("https://placehold.co/256x256/222/FFF?text=Ground+Truth\nTidak+Ditemukan",
                                 caption="Ground Truth Tidak Ditemukan", use_column_width=True)
                else:
                    st.success(f"Wajah Dikenali Sebagai: **{best_match_name}**")

                    if pil_gt is None:
                        pil_gt = Image.fromarray(st.session_state.face_db[best_match_name]['image'])

                    with col3:
                        st.image(pil_gt, caption=f"Ground Truth ({best_match_name})", use_column_width=True)

                    emb_gt = st.session_state.face_db[best_match_name]['embedding']
                    emb_masked = get_facenet_embedding(pil_masked, facenet, device)
                    sim_masked_vs_gt = None
                    if emb_masked is not None:
                        sim_masked_vs_gt = cosine_similarity(emb_masked.unsqueeze(0), emb_gt.unsqueeze(0)).item()

                    col1.metric("Similarity vs GT",
                                f"{sim_masked_vs_gt:.4f}" if sim_masked_vs_gt is not None else "N/A")
                    delta = best_match_score - sim_masked_vs_gt if sim_masked_vs_gt is not None else None
                    col2.metric("Similarity vs GT", f"{best_match_score:.4f}",
                                delta=f"{delta:.4f}" if delta is not None else None)
                    col3.metric("Similarity vs GT", "1.0000 (Target)")
