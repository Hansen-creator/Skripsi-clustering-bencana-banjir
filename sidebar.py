import streamlit as st

def show_sidebar():
    """
    Menampilkan sidebar navigasi minimalis (gaya tombol) 
    untuk user yang sudah login.
    """
    
    # Judul sidebar (Diganti ke markdown agar bisa center)
    # Original: st.sidebar.title("Dashboard User")
    
    # MODIFIKASI: Judul sidebar di-center (disamakan dengan style admin)
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 1.85rem;'>Dashboard", unsafe_allow_html=True)
    
    if st.sidebar.button("Home", use_container_width=True):
        st.switch_page("pages/1_Home.py")
        
    if st.sidebar.button("Dataset", use_container_width=True):
        st.switch_page("pages/4_Dataset.py")

    if st.sidebar.button("Clustering", use_container_width=True):
        st.switch_page("pages/2_Clustering.py")

    if st.sidebar.button("Profile", use_container_width=True):
        st.switch_page("pages/3_Profile.py")

    if st.sidebar.button("FAQ", use_container_width=True):
        st.switch_page("pages/5_FAQ.py")

    if st.sidebar.button("About", use_container_width=True):
        st.switch_page("pages/6_About.py")

    st.sidebar.markdown("---")
    
    # Tombol Logout (minimalis)
    if st.sidebar.button("Logout", use_container_width=True):
        # Reset session state
        keys_to_del = ["logged_in", "current_user", "user_role"]
        for key in keys_to_del:
            if key in st.session_state:
                del st.session_state[key]
        
        # Arahkan kembali ke halaman login utama
        st.switch_page("main.py")