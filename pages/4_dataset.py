import streamlit as st
import pandas as pd
from sidebar import show_sidebar
from utils_db import get_connection
import io

st.set_page_config(page_title="Dataset", layout="wide")

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
    st.switch_page("main.py")

# SIDEBAR
show_sidebar()

st.title("Dataset")
st.write("Berikut adalah template dataset yang tersedia untuk melakukan clustering.")

# FUNGSI GET DATASET YANG DIPILIH
def get_selected_datasets():
    """Mengambil dataset yang dipilih untuk ditampilkan ke user"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT dataset_name, file_name, uploaded_by, uploaded_at, file_content
            FROM datasets 
            WHERE is_selected = 1
            ORDER BY uploaded_at DESC
        """)
        rows = cur.fetchall()
        cols = [column[0] for column in cur.description]
        return pd.DataFrame.from_records(rows, columns=cols)
    except Exception as e:
        st.error(f"Error mengambil dataset: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def display_dataset_preview(file_content, dataset_name):
    """Menampilkan preview dataset dalam bentuk tabel"""
    try:
        # Convert bytes to DataFrame
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Tampilkan preview dataset
        st.subheader(f"📊 Preview Data: {dataset_name}")
        
        with st.container(border=True):
            st.write("**Statistik Data:**")
            # Statistik dataset
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Jumlah Baris", df.shape[0])
            with col2:
                st.metric("Jumlah Kolom", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Tampilkan dataframe
            st.dataframe(df, use_container_width=True, height=400)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error membaca dataset: {e}")
        return None

# TAMPILKAN DATASET YANG DIPILIH
datasets_df = get_selected_datasets()

if datasets_df.empty:
    st.info("📭 Belum ada dataset yang tersedia untuk analisis. Silakan hubungi admin untuk mengaktifkan dataset.")
else:
    st.success(f"Tersedia {len(datasets_df)} dataset untuk analisis")
    
    dataset_names = [row['dataset_name'] for _, row in datasets_df.iterrows()]
    tabs = st.tabs([f"📄 {name}" for name in dataset_names])
    
    for i, (tab, (_, row)) in enumerate(zip(tabs, datasets_df.iterrows())):
        with tab:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Tampilkan metadata 
                st.caption(f"File: {row['file_name']} | Diunggah oleh: {row['uploaded_by']} | Tanggal Upload: {row['uploaded_at']}")
            
            with col2:
                # Download button
                st.download_button(
                    label="Download CSV",
                    data=row["file_content"],
                    file_name=row["file_name"],
                    mime="text/csv",
                    key=f"download_{i}",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Tampilkan preview dataset
            df = display_dataset_preview(row["file_content"], row['dataset_name'])

            # TOMBOL MULAI CLUSTERING
            st.write("") 
            if df is not None:
                if st.button("Mulai Clustering", key=f"cluster_{i}", type="primary", use_container_width=True):
                    st.session_state["selected_dataset"] = row['dataset_name']
                    st.switch_page("pages/2_Clustering.py")