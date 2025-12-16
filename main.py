import streamlit as st
from utils_db import check_user, add_user, user_exists, log_login
import time

# CONFIG
st.set_page_config(
    page_title="Login System",
    page_icon="🔐",
    layout="centered"
)

# CSS & HIDE SIDEBAR
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebarCollapsedControl"] {display: none;}
        
        .main .block-container {padding-top: 2rem;}
        .login-container {
            border-radius: 12px;
            padding: 2.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin: 0 auto;
            max-width: 420px;
            border: 1px solid #e2e8f0;
            /* background-color: #fafafa; */ /* Dihapus untuk menghilangkan background putih */
        }
        .title {
            text-align: center;
            color: #1a202c;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #718096;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-weight: 500;
            border: none;
            transition: all 0.2s ease;
            font-size: 0.95rem;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .stButton button[kind="primary"] {
            background: #2d3748;
            color: white;
        }
        .stButton button[kind="primary"]:hover {
            background: #4a5568;
        }
        .stButton button[kind="secondary"] {
            background: #f7fafc;
            color: #4a5568;
            border: 1px solid #e2e8f0 !important;
        }
        .stButton button[kind="secondary"]:hover {
            background: #edf2f7;
        }
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            padding: 0.75rem;
            font-size: 0.95rem;
        }
        .stTextInput input:focus {
            border-color: #4a5568;
            box-shadow: 0 0 0 2px rgba(74,85,104,0.1);
        }
        .divider {
            text-align: center;
            margin: 1.5rem 0;
            color: #a0aec0;
            position: relative;
        }
        .divider::before {
            content: "";
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
            background: #e2e8f0;
        }
        .divider span {
            padding: 0 1rem;
            position: relative;
            background-color: #fafafa; /* Jika ingin konsisten, ubah ini juga jika perlu */
        }
        .error-box {
            background: #fed7d7;
            color: #c53030;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border-left: 4px solid #f56565;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        .success-box {
            background: #c6f6d5;
            color: #2f855a;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border-left: 4px solid #48bb78;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# SESSION STATE
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "show_register" not in st.session_state:
    st.session_state.show_register = False
if "login_error" not in st.session_state:
    st.session_state.login_error = None
if "register_error" not in st.session_state:
    st.session_state.register_error = None
if "register_success" not in st.session_state:
    st.session_state.register_success = False


# LOGOUT FUNCTION
def logout():
    keys_to_clear = [
        "logged_in", "current_user", "user_role", "login_username",
        "login_password", "reg_username", "reg_password", "reg_confirm",
        "login_error", "register_error", "register_success"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.logged_in = False
    st.session_state.show_register = False


# FUNGSI FOKUS OTOMATIS
def autofocus_input(input_id):
    """Inject JavaScript untuk auto-focus field tertentu"""
    js = f"""
    <script>
        setTimeout(function() {{
            var input = window.parent.document.querySelector('input[id*="{input_id}"]');
            if (input) input.focus();
        }}, 100);
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)


# LOGIN PAGE
def login_page():
    st.markdown('<div class="title">Login Sistem</div>', unsafe_allow_html=True)

    if st.session_state.get("login_error"):
        st.markdown(f'<div class="error-box">{st.session_state.login_error}</div>', unsafe_allow_html=True)
        st.session_state.login_error = None  

    if st.session_state.get("register_success"):
        st.markdown('<div class="success-box">✅ Registrasi berhasil! Silakan login dengan akun baru Anda.</div>', unsafe_allow_html=True)
        st.session_state.register_success = False

    with st.form("login_form"):
        username = st.text_input("**Username**", key="login_username", placeholder="Masukkan username Anda")
        password = st.text_input("**Password**", type="password", key="login_password", placeholder="Masukkan password Anda")
        st.write("") 
        login_submitted = st.form_submit_button("**Login**", use_container_width=True, type="primary")

    if login_submitted:
        try:
            if not username:
                st.session_state.login_error = "⚠️ Username jangan kosong"
                autofocus_input("login_username")
            elif not password:
                st.session_state.login_error = "⚠️ Password jangan kosong"
                autofocus_input("login_password")
            elif not user_exists(username):
                st.session_state.login_error = f"❌ Username '{username}' belum terdaftar"
            else:
                user = check_user(username, password)
                if user:
                    if len(user) > 2 and user[2] == 0:
                        st.session_state.login_error = "❌ Akun Anda dinonaktifkan. Hubungi administrator."
                    else:
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        st.session_state.user_role = user[1]
                        log_login(username, user[1])
                        st.success(f"✅ Selamat datang, {username}!")
                        time.sleep(1)
                        if user[1] == "admin":
                            st.switch_page("pages/admin_dashboard.py")
                        else:
                            st.switch_page("pages/1_Home.py")
                else:
                    st.session_state.login_error = "❌ Password salah"

            if st.session_state.get("login_error"):
                st.rerun()

        except Exception as e:
            st.session_state.login_error = f"❌ Terjadi kesalahan sistem: {str(e)}"
            st.rerun()

    st.markdown('<div class="divider"><span>atau</span></div>', unsafe_allow_html=True)
    if st.button("**Buat Akun Baru**", key="goto_register", use_container_width=True, type="secondary"):
        st.session_state.show_register = True
        st.session_state.login_error = None 
        st.rerun()


# REGISTER PAGE
def register_page():
    st.markdown('<div class="title">Buat Akun Baru</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Isi form berikut untuk membuat akun baru</div>', unsafe_allow_html=True)

    if st.session_state.get("register_error"):
        st.markdown(f'<div class="error-box">{st.session_state.register_error}</div>', unsafe_allow_html=True)
        st.session_state.register_error = None

    with st.form("register_form"):
        new_username = st.text_input("**Username Baru**", key="reg_username", placeholder="Buat username")
        new_password = st.text_input("**Password Baru**", type="password", key="reg_password", placeholder="Buat password")
        confirm_password = st.text_input("**Konfirmasi Password**", type="password", key="reg_confirm", placeholder="Konfirmasi password")
        st.write("")
        register_submitted = st.form_submit_button("**Daftar**", use_container_width=True, type="primary")

    if register_submitted:
        try:
            if not new_username:
                st.session_state.register_error = "❌ Username jangan kosong"
                autofocus_input("reg_username")
            elif not new_password:
                st.session_state.register_error = "❌ Password Baru jangan kosong"
                autofocus_input("reg_password")
            elif not confirm_password:
                st.session_state.register_error = "❌ Konfirmasi Password jangan kosong"
                autofocus_input("reg_confirm")
            elif user_exists(new_username):
                st.session_state.register_error = f"❌ Username '{new_username}' sudah terdaftar"
            elif new_password != confirm_password:
                st.session_state.register_error = "❌ Password tidak cocok"
                autofocus_input("reg_password")
            elif len(new_password) < 3:
                st.session_state.register_error = "❌ Password terlalu pendek (minimal 3 karakter)"
                autofocus_input("reg_password")
            else:
                if add_user(new_username, new_password): 
                    st.session_state.register_success = True
                    st.session_state.register_error = None
                    st.session_state.show_register = False 
                    st.rerun()
                else:
                    st.session_state.register_error = "❌ Gagal menyimpan user baru"

            if st.session_state.get("register_error"):
                st.rerun()

        except Exception as e:
            st.session_state.register_error = f"❌ Terjadi kesalahan sistem: {str(e)}"
            st.rerun()

    st.markdown('<div class="divider"><span></span></div>', unsafe_allow_html=True)
    if st.button("**Kembali ke Login**", key="back_to_login", use_container_width=True, type="secondary"):
        st.session_state.show_register = False
        st.session_state.register_error = None 
        st.rerun()


# MAIN ENTRY
def main_entry():
    try:
        if not st.session_state.logged_in:
            if st.session_state.get("show_register", False):
                register_page()
            else:
                login_page()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.session_state.user_role == "admin":
                st.switch_page("pages/admin_dashboard.py")
            else:
                st.switch_page("pages/1_Home.py")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")
        if st.button("Logout dan Kembali"):
            logout()
            st.rerun()


if __name__ == "__main__":
    main_entry()
