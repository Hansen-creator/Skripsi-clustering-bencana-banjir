import streamlit as st
from utils_db import get_connection, user_exists, check_user, save_dataset # Pastikan save_dataset di-import
import pandas as pd
import io
from PIL import Image
import base64

# FUNGSI BANTUAN
def make_circle_avatar(image_bytes):
    """Mengubah gambar menjadi bentuk lingkaran"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        
        # Buat mask lingkaran
        mask = Image.new("L", img.size, 0)
        mask_draw = Image.new("L", img.size, 0)
        # Gambar lingkaran
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask_draw)
        draw.ellipse((0, 0) + img.size, fill=255)
        
        # Terapkan mask
        img.putalpha(mask_draw)
        
        # Convert ke bytes
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        st.error(f"Error processing avatar: {e}")
        return image_bytes

def display_avatar(image_bytes, size=100):
    """Menampilkan avatar dalam bentuk lingkaran"""
    if image_bytes:
        try:
            # Buat avatar circle
            circle_avatar = make_circle_avatar(image_bytes)
            
            # Convert ke base64 untuk display
            encoded = base64.b64encode(circle_avatar).decode()
            st.markdown(
                f'<div style="display: flex; justify-content: center;">'
                f'<img src="data:image/png;base64,{encoded}" width="{size}" height="{size}" style="border-radius: 50%; object-fit: cover;">'
                f'</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error displaying avatar: {e}")
    else:
        # Default avatar dengan initial
        initial = st.session_state.current_user[0].upper() if "current_user" in st.session_state else "A"
        st.markdown(
            f'<div style="display: flex; justify-content: center;">'
            f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:#4a90e2;color:white;display:flex;align-items:center;justify-content:center;font-size:{size*0.4}px;font-weight:bold;">{initial}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# FUNGSI LOGOUT REUSABLE
def logout():
    """Reset session state untuk logout tanpa error"""
    keys_to_del = ["logged_in", "current_user", "user_role", "admin_menu"] # Hapus juga admin_menu
    for key in keys_to_del:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Logout berhasil! Anda akan diarahkan ke halaman login...")
    st.switch_page("main.py")

# PAGE CONFIG
st.set_page_config(
    page_title="Admin Dashboard", 
    page_icon="🛠️",
    layout="wide"
)

# Sembunyikan navigasi sidebar default
hide_default_format = """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stSidebar"] {
            padding-top: 0rem;
        }
        .circle-avatar {
            border-radius: 50%;
            object-fit: cover;
        }
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

# AUTH CHECK
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("❌ Anda harus login terlebih dahulu!")
    if st.button("Kembali ke Login"):
        st.switch_page("main.py")
    st.stop()

if st.session_state.user_role != "admin":
    st.error("Akses ditolak! Halaman ini hanya untuk admin.")
    if st.button("Kembali ke Home"):
        st.switch_page("pages/1_Home.py")
    st.stop()

st.sidebar.markdown("<h1 style='text-align: center; font-size: 1.85rem;'>Dashboard Admin</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

if "admin_menu" not in st.session_state:
    st.session_state.admin_menu = "admin_dashboard" 

if st.sidebar.button("Dashboard", use_container_width=True):
    st.session_state.admin_menu = "admin_dashboard"
if st.sidebar.button("User Management", use_container_width=True):
    st.session_state.admin_menu = "user_management"
if st.sidebar.button("Dataset Management", use_container_width=True):
    st.session_state.admin_menu = "dataset_management"
if st.sidebar.button("Profile Admin", use_container_width=True):
    st.session_state.admin_menu = "profile_admin"
if st.sidebar.button("About", use_container_width=True):
    st.session_state.admin_menu = "about"

st.sidebar.markdown("---")

if st.sidebar.button("Logout", use_container_width=True):
    logout()

if st.session_state.admin_menu == "admin_dashboard":
    st.title("Admin Dashboard")
    st.success(f"Selamat datang, Admin {st.session_state.current_user}!")
    st.warning("**Perhatian:** Sebagai Admin, Anda hanya dapat mengakses halaman admin. Akses ke halaman user biasa dinonaktifkan.")

    # Statistik
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            conn.close()
            st.metric("Total Users", total_users)
        except:
            st.metric("Total Users", "Error")
    with col2:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
            total_admins = cursor.fetchone()[0]
            conn.close()
            st.metric("Total Admin", total_admins)
        except:
            st.metric("Total Admin", "Error")
    with col3:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active=1")
            total_active = cursor.fetchone()[0]
            conn.close()
            st.metric("🟢 Akun Aktif", total_active)
        except:
            st.metric("🟢 Akun Aktif", "Error")
    with col4:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active=0")
            total_nonactive = cursor.fetchone()[0]
            conn.close()
            st.metric("🔴 Akun Nonaktif", total_nonactive)
        except:
            st.metric("🔴 Akun Nonaktif", "Error")

    st.subheader("📜 Log Aktivitas Login")

    try:
        conn = get_connection()
        query = """
            SELECT TOP 1000 username, role, login_time
            FROM login_logs
            ORDER BY login_time DESC
        """
        df_logs = pd.read_sql(query, conn)
        conn.close()

        if not df_logs.empty:
            rows_per_page = 10
            total_rows = len(df_logs)
            total_pages = (total_rows - 1) // rows_per_page + 1

            page = st.number_input(
                "Halaman", min_value=1, max_value=total_pages, step=1, value=1
            )
            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            df_page = df_logs.iloc[start_idx:end_idx]

            st.dataframe(df_page, use_container_width=True)
            st.caption(f"Menampilkan {start_idx+1}–{min(end_idx, total_rows)} dari {total_rows} log")

            # Download ke Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_logs.to_excel(writer, index=False, sheet_name="Login Logs")
            excel_data = output.getvalue()

            st.download_button(
                label="📥 Download Semua Log (Excel)",
                data=excel_data,
                file_name="login_logs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.info("Belum ada aktivitas login yang tercatat.")
    except Exception as e:
        st.error(f"❌ Error mengambil data log: {e}")

elif st.session_state.admin_menu == "user_management":
    st.title("User Management")
    st.info("**Akses Terbatas:** Anda hanya dapat mengelola user dari halaman ini.")

    try:
        conn = get_connection()
        query = """
            SELECT 
                id, 
                username, 
                role, 
                bio, 
                avatar_filename, 
                is_active,
                created_at, 
                last_updated
            FROM users
            ORDER BY created_at DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if not df.empty:
            st.subheader("Daftar Semua User")

            # PAGINATION
            rows_per_page = 10
            total_rows = len(df)
            total_pages = (total_rows - 1) // rows_per_page + 1

            page = st.number_input(
                "Halaman", min_value=1, max_value=total_pages, step=1, value=1, key="user_page"
            )
            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            df_page = df.iloc[start_idx:end_idx]

            # TABEL DENGAN AKSI (DIKEMBALIKAN KE 7 KOLOM)
            for _, row in df_page.iterrows():
                # DIKEMBALIKAN KE 7 KOLOM
                col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2, 2, 2])
                with col1:
                    st.write(f" **{row['username']}**")
                with col2:
                    st.write(f"Role: {row['role']}")
                with col3:
                    status_text = "🟢 Aktif" if row["is_active"] == 1 else "🔴 Nonaktif"
                    st.write(f"Status: {status_text}")
                with col4:
                    st.write(f"Dibuat: {row['created_at']}")
                with col5:
                    if row["is_active"] == 1:
                        if st.button("🚫 Nonaktifkan", key=f"deact_{row['id']}"):
                            try:
                                conn = get_connection()
                                cursor = conn.cursor()
                                cursor.execute(
                                    "UPDATE users SET is_active = 0, last_updated = GETDATE() WHERE id = ?", 
                                    (row['id'],)
                                )
                                conn.commit()
                                conn.close()
                                st.success(f" User {row['username']} dinonaktifkan!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                    else:
                        if st.button(" Aktifkan", key=f"act_{row['id']}"):
                            try:
                                conn = get_connection()
                                cursor = conn.cursor()
                                cursor.execute(
                                    "UPDATE users SET is_active = 1, last_updated = GETDATE() WHERE id = ?", 
                                    (row['id'],)
                                )
                                conn.commit()
                                conn.close()
                                st.success(f"✅ User {row['username']} diaktifkan!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                
                # ==========================================================
                # LOGIKA EDIT (COL6) YANG DIKEMBALIKAN
                # ==========================================================
                with col6:
                    if st.button(" Edit", key=f"edit_{row['id']}"):
                        st.session_state.editing_user_id = row['id']
                        st.session_state.editing_username = row['username']
                        st.rerun()
                # ==========================================================
                
                # ==========================================================
                # LOGIKA DELETE (COL7) YANG DIPINDAHKAN
                # ==========================================================
                with col7:
                    if st.button(" Hapus", key=f"del_{row['id']}"):
                        try:
                            conn = get_connection()
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM users WHERE id = ?", (row['id'],))
                            conn.commit()
                            conn.close()
                            st.success(f"✅ User {row['username']} berhasil dihapus!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error menghapus user: {e}")

            st.caption(f"Menampilkan {start_idx+1}–{min(end_idx, total_rows)} dari {total_rows} user")

            if "editing_user_id" in st.session_state:
                st.markdown("---")
                st.subheader(f"✏️ Edit User: {st.session_state.editing_username}")
                
                try:
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT username, role, password, avatar, avatar_filename 
                        FROM users WHERE id=?
                    """, (st.session_state.editing_user_id,))
                    user_data = cursor.fetchone()
                    conn.close()

                    if user_data:
                        with st.form(f"edit_user_form_{st.session_state.editing_user_id}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                new_username = st.text_input("Username", value=user_data[0])
                                st.write(f"**Role:** {user_data[1]} (tidak dapat diubah)")
                                
                                st.subheader("🔐 Ubah Password")
                                new_password = st.text_input("Password Baru", type="password", 
                                                            placeholder="Kosongkan jika tidak ingin mengubah password")
                                confirm_password = st.text_input("Konfirmasi Password Baru", type="password",
                                                                placeholder="Kosongkan jika tidak ingin mengubah password")
                            
                            with col2:
                                st.write("**Avatar Saat Ini:**")
                                # Tampilkan avatar saat ini
                                if user_data[3]: # Jika ada avatar
                                    display_avatar(user_data[3], size=100)
                                else:
                                    st.image("https://via.placeholder.com/100", width=100)
                                
                                new_avatar = st.file_uploader("Upload Avatar Baru", 
                                                                type=["jpg", "jpeg", "png"],
                                                                key=f"avatar_upload_{st.session_state.editing_user_id}")

                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.form_submit_button("💾 Simpan Perubahan"):
                                    # Validasi password
                                    if new_password and new_password != confirm_password:
                                        st.error("❌ Password tidak sama!")
                                    else:
                                        try:
                                            conn = get_connection()
                                            cursor = conn.cursor()
                                            
                                            # Build dynamic update query
                                            update_fields = []
                                            params = []
                                            
                                            if new_username != user_data[0]:
                                                update_fields.append("username = ?")
                                                params.append(new_username)
                                            
                                            if new_password:
                                                update_fields.append("password = ?")
                                                params.append(new_password)
                                            
                                            if new_avatar:
                                                avatar_bytes = new_avatar.read()
                                                update_fields.append("avatar = ?")
                                                params.append(avatar_bytes)
                                                update_fields.append("avatar_filename = ?")
                                                params.append(new_avatar.name)
                                            
                                            if update_fields:
                                                update_fields.append("last_updated = GETDATE()")
                                                sql = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
                                                params.append(st.session_state.editing_user_id)
                                                
                                                cursor.execute(sql, params)
                                                conn.commit()
                                                st.success("✅ Data user berhasil diperbarui!")
                                                
                                                del st.session_state.editing_user_id
                                                del st.session_state.editing_username
                                                st.rerun()
                                            else:
                                                st.info(" Tidak ada perubahan yang dilakukan.")
                                                
                                        except Exception as e:
                                            st.error(f"❌ Error memperbarui user: {e}")
                                        finally:
                                            conn.close()
                            
                            with col_btn2:
                                if st.form_submit_button("❌ Batal"):
                                    del st.session_state.editing_user_id
                                    del st.session_state.editing_username
                                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error mengambil data user: {e}")

        else:
            st.warning("⚠️ Belum ada user terdaftar di database.")
    except Exception as e:
        st.error(f"❌ Error mengambil data user: {e}")

    st.markdown("---")

    # FORM TAMBAH USER
    st.subheader("➕ Tambah User Baru")
    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username", placeholder="Masukkan username")
            new_password = st.text_input("Password", type="password", placeholder="Masukkan password")
            confirm_password = st.text_input("Konfirmasi Password", type="password", placeholder="Ulangi password")
            new_role = st.selectbox("Role", ["user", "admin"])
        
        with col2:
            avatar_file = st.file_uploader("Upload Avatar (opsional)", type=["jpg", "jpeg", "png"])

        if st.form_submit_button("✅ Simpan User Baru"):
            if not new_username or not new_password:
                st.error("❌ Username dan password wajib diisi!")
            elif user_exists(new_username):
                st.error("❌ Username sudah ada!")
            elif new_password != confirm_password:
                st.error("❌ Password tidak sama!")
            else:
                try:
                    conn = get_connection()
                    cursor = conn.cursor()

                    avatar_bytes, avatar_filename = None, None
                    if avatar_file:
                        avatar_bytes = avatar_file.read()
                        avatar_filename = avatar_file.name

                    cursor.execute("""
                        INSERT INTO users (username, password, role, avatar, avatar_filename, is_active) 
                        VALUES (?, ?, ?, ?, ?, 1)
                    """, (new_username, new_password, new_role, avatar_bytes, avatar_filename))
                    
                    conn.commit()
                    st.success(f"✅ User {new_username} berhasil ditambahkan!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error menambahkan user: {e}")
                finally:
                    conn.close()

elif st.session_state.admin_menu == "dataset_management":
    st.title(" Dataset Management")
    st.info("**Akses Terbatas:** Dataset management hanya tersedia di halaman admin.")

    # Tab diubah (tanpa emoji)
    tab1, tab2, tab3 = st.tabs(["Dataset List", "Pilih Dataset untuk User", "Upload Dataset"])

    with tab1:
        st.subheader("Managed Datasets")

        try:
            conn = get_connection()
            # MODIFIKASI: Hapus kolom description dari query
            query = """
                SELECT id, dataset_name, file_name, file_type, uploaded_by, 
                       uploaded_at, is_selected 
                FROM datasets
                ORDER BY uploaded_at DESC
            """
            df = pd.read_sql(query, conn)
            conn.close()

            if not df.empty:
                st.caption(f"Total dataset ditemukan: {len(df)}")

                for _, row in df.iterrows():
                    # Expander diubah (tanpa emoji)
                    status_label = "[Aktif Dilihat User]" if row['is_selected'] else "[Tidak Aktif]"
                    with st.expander(f"📄 {row['dataset_name']} {status_label}"):
                        # MODIFIKASI: Layout kolom disesuaikan karena deskripsi hilang
                        col1, col2, col3 = st.columns(3) # Jadi 3 kolom saja
                        # col1.write(f"**Deskripsi:** {row['description'] or '-'}") # Deskripsi dihapus
                        col1.write(f"**File Name:** {row['file_name']}")
                        col2.write(f"**Uploader:** {row['uploaded_by']}")
                        # Status diubah (tanpa emoji)
                        col3.write(f"**Status:** {'Tampil ke User' if row['is_selected'] else 'Tidak Tampil'}")
                        col3.write(f"**Uploaded At:** {row['uploaded_at']}")

                        # MODIFIKASI: Layout kolom tombol disesuaikan, tombol Edit dihapus
                        c1, c2, c3 = st.columns(3) # Jadi 3 kolom
                        with c1:
                            # Tombol Download tetap ada
                            # ... (kode download tidak berubah) ...
                            if st.button("📥 Download", key=f"dl_{row['id']}"):
                                try:
                                    conn = get_connection()
                                    cur = conn.cursor()
                                    cur.execute("SELECT file_content, file_name FROM datasets WHERE id=?", (row['id'],))
                                    file_data = cur.fetchone()
                                    conn.close()
                                    if file_data:
                                        file_bytes, file_name = file_data
                                        st.download_button(
                                            label="⬇️ Click to Download",
                                            data=file_bytes,
                                            file_name=file_name,
                                            mime="text/csv", # Asumsi CSV
                                            key=f"dl_btn_{row['id']}"
                                        )
                                except Exception as e:
                                    st.error(f"❌ Error download: {e}")

                        with c2: # Tombol Delete dipindah ke c2
                            if st.button("🗑️ Delete", key=f"del_ds_{row['id']}"):
                                try:
                                    conn = get_connection()
                                    cur = conn.cursor()
                                    cur.execute("DELETE FROM datasets WHERE id=?", (row['id'],))
                                    conn.commit()
                                    conn.close()
                                    st.success(f"✅ Dataset {row['dataset_name']} berhasil dihapus!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error delete: {e}")
                        with c3: # Tombol Tampilkan/Sembunyikan dipindah ke c3
                            if row['is_selected']:
                                if st.button("❌ Sembunyikan", key=f"hide_{row['id']}"):
                                    try:
                                        conn = get_connection()
                                        cur = conn.cursor()
                                        cur.execute("UPDATE datasets SET is_selected = 0 WHERE id = ?", (row['id'],))
                                        conn.commit()
                                        conn.close()
                                        st.success(f"✅ Dataset {row['dataset_name']} disembunyikan dari user!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error: {e}")
                            else:
                                if st.button("✅ Tampilkan", key=f"show_{row['id']}"):
                                    try:
                                        conn = get_connection()
                                        cur = conn.cursor()
                                        cur.execute("UPDATE datasets SET is_selected = 0 WHERE id != ?", (row['id'],))
                                        cur.execute("UPDATE datasets SET is_selected = 1 WHERE id = ?", (row['id'],))
                                        conn.commit()
                                        conn.close()
                                        st.success(f"✅ Dataset {row['dataset_name']} ditampilkan ke user! (Dataset lain otomatis disembunyikan)")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Belum ada dataset tersimpan.")
        except Exception as e:
            st.error(f"❌ Error mengambil dataset: {e}")

    with tab2:
        st.subheader("🎯 Kelola Dataset untuk User")
        st.info("Pilih dataset mana yang akan ditampilkan ke halaman user untuk analisis. Hanya satu dataset yang bisa aktif dalam satu waktu.")
        
        try:
            conn = get_connection()
            # MODIFIKASI: Kolom description dihapus dari query
            query = """
                SELECT id, dataset_name, is_selected 
                FROM datasets 
                ORDER BY is_selected DESC, uploaded_at DESC
            """
            df = pd.read_sql(query, conn)
            conn.close()

            if not df.empty:
                selected_count = len(df[df['is_selected'] == 1])
                st.metric("📊 Dataset yang Ditampilkan ke User", selected_count)
                
                for _, row in df.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1]) # Layout tetap
                    
                    with col1:
                        status_icon = "🎯 [Aktif]" if row['is_selected'] else "❌ [Nonaktif]"
                        st.write(f"{status_icon} **{row['dataset_name']}**")
                        # Baris deskripsi dihapus
                        # if row['description']:
                        #     st.caption(f"{row['description']}")
                    
                    with col2:
                        if row['is_selected']:
                            if st.button("❌ Sembunyikan", key=f"tab2_hide_{row['id']}", use_container_width=True):
                                # ... (logika sembunyikan tidak berubah) ...
                                try:
                                    conn = get_connection()
                                    cur = conn.cursor()
                                    cur.execute("UPDATE datasets SET is_selected = 0 WHERE id = ?", (row['id'],))
                                    conn.commit()
                                    conn.close()
                                    st.success(f"✅ Dataset {row['dataset_name']} disembunyikan!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                        else:
                            if st.button("✅ Tampilkan", key=f"tab2_show_{row['id']}", use_container_width=True):
                                # ... (logika tampilkan tidak berubah) ...
                                try:
                                    conn = get_connection()
                                    cur = conn.cursor()
                                    cur.execute("UPDATE datasets SET is_selected = 0 WHERE id != ?", (row['id'],))
                                    cur.execute("UPDATE datasets SET is_selected = 1 WHERE id = ?", (row['id'],))
                                    conn.commit()
                                    conn.close()
                                    st.success(f"✅ Dataset {row['dataset_name']} ditampilkan! (Dataset lain otomatis disembunyikan)")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                    
                    with col3:
                        if st.button("📋 Detail", key=f"tab2_detail_{row['id']}", use_container_width=True):
                            # Detail sekarang hanya menampilkan nama dataset
                            st.info(f"**Nama Dataset:** {row['dataset_name']}")
            else:
                st.warning("⚠️ Belum ada dataset tersimpan.")
        except Exception as e:
            st.error(f"❌ Error mengambil dataset: {e}")

    with tab3:
        st.subheader("Upload New Dataset")
        # Hanya menerima CSV
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            st.success(f"File {uploaded_file.name} uploaded successfully!")

            dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])
            # MODIFIKASI: Input deskripsi dihapus
            # dataset_description = st.text_area("Description") 
            
            col1, col2 = st.columns(2)
            with col1:
                auto_select = st.checkbox("Tampilkan ke user setelah upload", value=True)
            with col2:
                st.info("✅ Jika dicentang, dataset ini akan langsung aktif dan menonaktifkan dataset lain.")

            if st.button("💾 Save Dataset"):
                file_content = uploaded_file.read()

                # Logika Simpan (MODIFIED FOR SINGLE-SELECT)
                if auto_select:
                    try:
                        conn_update = get_connection()
                        cur_update = conn_update.cursor()
                        cur_update.execute("UPDATE datasets SET is_selected = 0")
                        conn_update.commit()
                        conn_update.close()
                    except Exception as e:
                        st.error(f"❌ Error menonaktifkan dataset lain: {e}")
                        st.stop() # Hentikan proses jika gagal

                # MODIFIKASI: Mengirim string kosong "" untuk deskripsi
                success = save_dataset(
                    dataset_name, "", # Deskripsi dikosongkan 
                    "Keseluruhan",
                    file_content, uploaded_file.name, uploaded_file.type,
                    st.session_state.current_user,
                    is_selected=1 if auto_select else 0 # Simpan status
                )
                if success:
                    st.success("✅ Dataset saved successfully!")
                    st.rerun()
                else:
                    st.error("❌ Gagal menyimpan dataset.")

elif st.session_state.admin_menu == "profile_admin":
    # ... (kode profile admin tidak berubah) ...
    st.title(" Profile Admin")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT username, role, bio, avatar, avatar_filename, last_updated
            FROM users WHERE username=?
        """, (st.session_state.current_user,))
        admin_info = cursor.fetchone()
        conn.close()
    except Exception as e:
        st.error(f"❌ Error retrieving admin info: {e}")
        st.stop()

    if not admin_info:
        st.error("❌ Data admin tidak ditemukan.")
        st.stop()

    username, role, bio, avatar_bytes, avatar_filename, last_updated = admin_info

    st.session_state.bio = bio or ""
    if avatar_bytes:
        st.session_state.avatar = avatar_bytes
        st.session_state.avatar_filename = avatar_filename

    initial = username[0].upper()

    col1, col2 = st.columns([1, 3])
    with col1:
        if avatar_bytes:
            b64_img = base64.b64encode(avatar_bytes).decode()
            st.markdown(
                f"""
                <img src="data:image/png;base64,{b64_img}" 
                     style="width:120px;height:120px;border-radius:50%;object-fit:cover;">
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
        st.write(f"**Role:** {role}")
        st.write("**Bio:**")
        if bio and bio.strip() != "":
            st.write(bio)
        else:
            st.write("_(Bio kosong)_")

    st.markdown("---")
    st.write("Ubah username, password, atau upload avatar baru (jpg/jpeg/png).")

    with st.form("update_form"):
        new_username = st.text_input("Username Baru", value=username)
        new_password = st.text_input("Password Baru (kosong = tidak diubah)", type="password", value="")
        confirm_password = st.text_input("Konfirmasi Password Baru", type="password", value="")
        
        uploaded_file = st.file_uploader(
            "Upload Avatar (jpg/jpeg/png) — maksimal 2MB", type=["jpg", "jpeg", "png"]
        )

        submitted = st.form_submit_button("💾 Simpan Perubahan")

        if submitted:
            if new_username.strip() == "":
                st.error("❌ Username tidak boleh kosong.")
            elif new_password and new_password != confirm_password:
                st.error("❌ Password tidak sama!")
            else:
                update_avatar_bytes = None
                update_avatar_filename = None

                if uploaded_file is not None:
                    uploaded_file.seek(0, 2)
                    size = uploaded_file.tell()
                    uploaded_file.seek(0)
                    if size > 2 * 1024 * 1024:
                        st.error("❌ File terlalu besar. Maksimal 2MB.")
                    else:
                        data = uploaded_file.read()
                        try:
                            img = Image.open(io.BytesIO(data))
                            if img.format not in ("JPEG", "PNG"):
                                st.error("❌ Tipe file tidak valid. Hanya jpg/jpeg/png.")
                            else:
                                update_avatar_bytes = data
                                update_avatar_filename = uploaded_file.name
                        except Exception:
                            st.error("❌ File bukan gambar valid.")

                if (new_username == username and 
                    not new_password.strip() and 
                    uploaded_file is None):
                    st.info("ℹ️ Tidak ada perubahan yang dilakukan.")
                else:
                    try:
                        conn = get_connection()
                        cursor = conn.cursor()
                        
                        update_fields = []
                        params = []
                        
                        if new_username != username:
                            update_fields.append("username = ?")
                            params.append(new_username)
                        
                        if new_password.strip():
                            update_fields.append("password = ?")
                            params.append(new_password)
                        
                        if update_avatar_bytes:
                            update_fields.append("avatar = ?")
                            params.append(update_avatar_bytes)
                            update_fields.append("avatar_filename = ?")
                            params.append(update_avatar_filename)
                        
                        if update_fields:
                            update_fields.append("last_updated = GETDATE()")
                            sql = f"UPDATE users SET {', '.join(update_fields)} WHERE username = ?"
                            params.append(username)
                            
                            cursor.execute(sql, params)
                            conn.commit()
                            
                            st.session_state.current_user = new_username
                            if update_avatar_bytes:
                                st.session_state.avatar = update_avatar_bytes
                                st.session_state.avatar_filename = update_avatar_filename
                            
                            st.success("✅ Profile berhasil diperbarui!")
                            st.rerun()
                        else:
                            st.info("ℹ️ Tidak ada perubahan yang dilakukan.")
                            
                    except Exception as e:
                        st.error(f"❌ Error memperbarui profile: {e}")
                    finally:
                        conn.close()


elif st.session_state.admin_menu == "about":
    st.title("About This System")
    st.markdown("---")

    st.subheader("Pencipta Sistem")
    st.write("Sistem ini dikembangkan oleh **Hansen Pratama** sebagai bagian dari proyek pembelajaran dan penerapan ilmu di bidang **Data Science** dan **Pengembangan Aplikasi**.")

    st.subheader("Pembimbing & Ucapan Terima Kasih")
    st.write("""
    Pengembangan sistem ini tidak lepas dari bimbingan dan bantuan dari:
    - **Ibu Teny Handhayani, S.Kom., M.Kom., Ph.D.**
    - **Bapak Janson Hendryli, S.Kom., M.Kom.**

    Terima kasih atas arahan dan masukan yang telah diberikan.
    """)

    st.subheader("Latar Belakang")
    st.write("""
    Dalam era digital, analisis data menjadi salah satu faktor kunci untuk pengambilan keputusan.
    Sistem ini dibuat untuk mempermudah pengguna dalam **mengunggah dataset, melakukan analisis clustering, 
    dan menampilkan hasilnya secara interaktif** melalui antarmuka yang sederhana.
    """)

    st.subheader("Tujuan Utama")
    st.write("""
    - Memberikan platform yang mudah digunakan untuk analisis data.  
    - Membantu pengguna memahami pola tersembunyi dalam dataset melalui clustering.  
    - Menjadi media pembelajaran bagi pengembang maupun pengguna.  
    """)

    st.subheader("Metode yang Digunakan")
    st.write("""
    Dalam pengembangan sistem ini digunakan:  
    - **Python & Streamlit** untuk antarmuka pengguna.  
    - **SQL Server** sebagai database.  
    - **Metode Clustering Density Based Clustering (DENCLUE)** untuk analisis data.  
    """)