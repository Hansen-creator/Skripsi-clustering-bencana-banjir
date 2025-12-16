import streamlit as st
from sidebar import show_sidebar

st.set_page_config(page_title="About")

hide_default_format = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {padding-top: 0rem;}
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

# SHOW SIDEBAR
show_sidebar()

# ABOUT CONTENT
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