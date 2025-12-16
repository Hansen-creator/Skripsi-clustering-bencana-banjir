import streamlit as st
from io import BytesIO
from PIL import Image
import base64

try:
    from db_helpers import get_user_profile, update_user
    from sidebar import show_sidebar
except ImportError:
    def get_user_profile(username):
        st.error("Gagal memuat db_helpers.py")
        return {"username": username, "bio": "Contoh bio", "avatar": None, "avatar_filename": None}

    def update_user(old_username, new_username, new_password, new_bio, avatar_bytes, avatar_filename):
        st.error("Gagal memuat db_helpers.py")
        return False, "Database helper tidak ditemukan."

    def show_sidebar():
        st.sidebar.error("Gagal memuat sidebar.py")
        st.sidebar.button("Logout (dummy)")


st.set_page_config(page_title="Profile")

# Sembunyikan navigasi sidebar default
hide_default_format = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {padding-top: 0rem;}
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

# CEK LOGIN
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    # Jika menggunakan versi Streamlit 1.33+
    if hasattr(st, 'switch_page'):
        st.switch_page("main.py")
    else:
        # Fallback untuk versi lama
        st.warning("Anda harus login untuk mengakses halaman ini.")
        st.stop()

# Panggil sidebar setelah cek login
try:
    show_sidebar()
except Exception as e:
    st.sidebar.error(f"Gagal memuat sidebar: {e}")

# Pastikan current_user ada di session_state
if "current_user" not in st.session_state:
    st.error("Sesi pengguna tidak ditemukan. Silakan login kembali.")
    if hasattr(st, 'switch_page'):
        st.switch_page("main.py")
    st.stop()

username = st.session_state.current_user
profile = get_user_profile(username)

if profile is None:
    st.error("User tidak ditemukan di database.")
    st.stop()

st.title("Profile")

# Tampilkan pesan sukses, error, atau info jika ada dari session_state
if "update_success" in st.session_state:
    st.success(st.session_state.update_success)
    del st.session_state.update_success

if "update_error" in st.session_state:
    st.error(st.session_state.update_error)
    del st.session_state.update_error

if "update_info" in st.session_state:
    st.info(st.session_state.update_info)
    del st.session_state.update_info

initial = username[0].upper()

avatar_bytes = profile.get("avatar")
avatar_filename = profile.get("avatar_filename")
bio_text = profile.get("bio") or ""

# Update session state dengan data terbaru dari database
st.session_state.bio = bio_text
if avatar_bytes:
    st.session_state.avatar = avatar_bytes
    st.session_state.avatar_filename = avatar_filename

col1, col2 = st.columns([1, 3])
with col1:
    if avatar_bytes:
        try:
            b64_img = base64.b64encode(avatar_bytes).decode()
            st.markdown(
                f"""
                <img src="data:image/png;base64,{b64_img}"
                     style="width:120px;height:120px;border-radius:50%;object-fit:cover;">
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Gagal memuat avatar: {e}")
            # Tampilkan initial sebagai fallback
            st.markdown(
                f"""
                <div style="width:120px;height:120px;border-radius:50%;
                            background:#4a90e2;color:white;display:flex;
                            align-items:center;justify-content:center;
                            font-size:48px;font-weight:bold;">
                    {initial}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"""
            <div style="width:120px;height:120px;border-radius:50%;
                        background:#4a90e2;color:white;display:flex;
                        align-items:center;justify-content:center;
                        font-size:48px;font-weight:bold;">
                {initial}
            </div>
            """,
            unsafe_allow_html=True,
        )

with col2:
    st.subheader(username)
    st.write("**Bio:**")
    if bio_text and bio_text.strip() != "":
        st.write(bio_text)
    else:
        st.write("_(Bio kosong)_")

st.markdown("---")
st.write("Ubah username, password, bio, atau upload avatar baru (jpg/jpeg/png).")

# FORM UPDATE PROFILE
with st.form("update_form"):
    new_username = st.text_input(
        "Username Baru",
        value=username,
        help="Username ini akan digunakan untuk login dan akan dilihat oleh pengguna lain."
    )
    new_password = st.text_input(
        "Password Baru (kosong = tidak diubah)",
        type="password",
        value="",
        help="Kosongkan bidang ini jika Anda tidak ingin mengubah password."
    )
    new_bio = st.text_area(
        "Bio (opsional, bisa sama dengan user lain)",
        value=bio_text,
        placeholder="Tulis bio tentang diri Anda...",
        help="Ceritakan sedikit tentang diri Anda. Ini akan muncul di halaman profil Anda."
    )

    uploaded_file = st.file_uploader(
        "Upload Avatar (jpg/jpeg/png) — maksimal 2MB",
        type=["jpg", "jpeg", "png"],
        help="Ukuran file tidak boleh melebihi 2MB."
    )

    submitted = st.form_submit_button("Simpan Perubahan")

    if submitted:

        if "update_success" in st.session_state: del st.session_state.update_success
        if "update_error" in st.session_state: del st.session_state.update_error
        if "update_info" in st.session_state: del st.session_state.update_info

        update_avatar_bytes = None
        update_avatar_filename = None
        validation_error = False
        error_message = None

        # Lakukan validasi
        if not new_username or new_username.strip() == "":
            error_message = "❌ Username tidak boleh kosong."
            validation_error = True

        if not validation_error and uploaded_file is not None:

            data = uploaded_file.getvalue()
            size = len(data)

            if size > 2 * 1024 * 1024: # 2MB
                error_message = "❌ File terlalu besar. Maksimal 2MB."
                validation_error = True
            else:
                try:
                    # Coba buka gambar untuk memvalidasi
                    img = Image.open(BytesIO(data))
                    img.verify() # Verifikasi format
                    
                    img_check = Image.open(BytesIO(data))
                    if img_check.format not in ("JPEG", "PNG"):
                        error_message = "❌ Tipe file tidak valid. Hanya jpg/jpeg/png."
                        validation_error = True
                    else:
                        update_avatar_bytes = data
                        update_avatar_filename = uploaded_file.name
                except Exception:
                    error_message = "❌ File bukan gambar valid."
                    validation_error = True

        if not validation_error:
            pw_param = new_password if new_password and new_password.strip() != "" else None
            
            bio_param = new_bio.strip()

            is_username_changed = new_username != username
            is_password_changed = pw_param is not None
            is_bio_changed = bio_param != bio_text
            is_avatar_changed = update_avatar_bytes is not None

            if not any([is_username_changed, is_password_changed, is_bio_changed, is_avatar_changed]):
                st.session_state.update_info = "Tidak ada perubahan yang dilakukan."
            else:
                # Hanya kirim data yang berubah ke fungsi update
                success, msg = update_user(
                    old_username=username,
                    new_username=new_username if is_username_changed else None,
                    new_password=pw_param if is_password_changed else None,
                    new_bio=bio_param if is_bio_changed else None,
                    avatar_bytes=update_avatar_bytes if is_avatar_changed else None,
                    avatar_filename=update_avatar_filename if is_avatar_changed else None,
                )

                if success:
                    # Update session state jika ada perubahan
                    if is_username_changed:
                        st.session_state.current_user = new_username
                    if is_bio_changed:
                        st.session_state.bio = bio_param
                    if is_avatar_changed:
                        st.session_state.avatar = update_avatar_bytes
                        st.session_state.avatar_filename = update_avatar_filename

                    st.session_state.update_success = "Profile berhasil diperbarui!"
                else:
                    st.session_state.update_error = msg 

        # Jika ada error validasi, set pesannya
        elif error_message:
            st.session_state.update_error = error_message

        # untuk menampilkan notifikasi di atas
        st.rerun()