import streamlit as st
from sidebar import show_sidebar

st.set_page_config(page_title="FAQ")

hide_default_format = """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stSidebar"] {
            padding-top: 0rem;
        }
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

# CEK LOGIN
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.switch_page("main.py")

# SIDEBAR
show_sidebar()

# KONTEN FAQ
st.title("❓ FAQ - Pertanyaan yang Sering Diajukan")
st.write("Berikut adalah beberapa pertanyaan yang sering diajukan beserta jawabannya. Jika ada pertanyaan lain, silakan hubungi admin.")

with st.expander("Bagaimana cara melakukan clustering?"):
    st.write("""
    Untuk melakukan **clustering**, silakan **download dataset** yang sudah disediakan pada halaman Dataset.  
    Anda dapat menggunakan dataset tersebut untuk melakukan eksperimen clustering.  
    **Jika Anda tidak menggunakan dataset yang sudah disediakan, maka proses clustering tidak akan dapat dilakukan.**
    """)

with st.expander("Bagaimana cara mendownload dataset?"):
    st.write("""
    Silakan masuk ke **halaman Dataset**, lalu klik tombol **Download file** untuk mengunduh dataset dalam format **CSV**.
    """)

with st.expander("Bagaimana cara mengubah profil?"):
    st.write("""
    Silakan masuk ke **halaman Profile**, lalu **upload foto profil** Anda dalam format **JPG**, **JPEG**, atau **PNG** yang sudah disediakan.
    """)

st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)
