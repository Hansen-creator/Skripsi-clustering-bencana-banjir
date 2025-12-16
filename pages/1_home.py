import streamlit as st
import requests
from sidebar import show_sidebar

st.set_page_config(page_title="Home", layout="wide")

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

# Cek login
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.switch_page("main.py")

show_sidebar()

# Halaman Home
st.title("Selamat Datang di Dashboard Banjir")

st.markdown(f"👋 Halo, **{st.session_state.current_user}**! Selamat datang di aplikasi ini.")

st.markdown(
    """
    **Banjir** adalah salah satu bencana alam yang sering terjadi di Indonesia, terutama saat musim hujan. 
    Dampak banjir dapat meliputi kerusakan rumah, infrastruktur, serta korban jiwa. 
    Dashboard ini bertujuan untuk memberikan informasi terkini serta edukasi mengenai banjir. 
    """
)

# Carousel Gambar Banjir
st.subheader("Dokumentasi Banjir")

images = [
    {
        "title": "Banjir Jakarta",
        "desc": "Banjir besar yang melanda Jakarta pada awal tahun.",
        "url": "https://stmikkomputama.ac.id/wp-content/uploads/2025/09/Ilustrasi-banjir.jpeg"
    },
    {
        "title": "Evakuasi Warga",
        "desc": "Tim SAR mengevakuasi warga terdampak banjir.",
        "url": "https://jatengprov.go.id/wp-content/uploads/2025/01/WhatsApp-Image-2025-01-20-at-15.56.55-2.jpeg"
    },
    {
        "title": "Rumah Terendam",
        "desc": "Banyak rumah warga yang terendam banjir akibat curah hujan tinggi.",
        "url": "https://sigap.sidoarjokab.go.id/images/foto_panduan/banjir.jpg"
    }
]

if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0

def prev_img():
    st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(images)

def next_img():
    st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(images)

current = images[st.session_state.carousel_index]
st.image(current["url"], caption=f"{current['title']} - {current['desc']}", use_container_width=True)

col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.button("Prev"):
        prev_img()
with col3:
    if st.button("Next"):
        next_img()

# Informasi Tambahan tentang Banjir
st.subheader("Informasi Mengenai Banjir")

with st.expander("Penyebab Banjir"):
    st.markdown(
        """
        - Curah hujan tinggi dalam waktu lama. 
        - Drainase atau saluran air yang tersumbat. 
        - Alih fungsi lahan (hutan berubah jadi pemukiman/industri). 
        - Pembuangan sampah sembarangan ke sungai. 
        - Permukaan tanah yang semakin kedap air akibat pembangunan. 
        """
    )

with st.expander("Dampak Banjir"):
    st.markdown(
        """
        - Kerusakan rumah dan fasilitas umum. 
        - Terhambatnya aktivitas ekonomi dan transportasi. 
        - Penyebaran penyakit seperti diare, leptospirosis, dan ISPA. 
        - Menyebabkan trauma psikologis bagi korban. 
        """
    )

with st.expander("Upaya Pencegahan Banjir"):
    st.markdown(
        """
        - Tidak membuang sampah sembarangan ke sungai. 
        - Membangun sistem drainase yang baik. 
        - Menjaga kawasan resapan air dengan menanam pohon. 
        - Membuat sumur resapan dan biopori. 
        - Melakukan normalisasi sungai dan saluran air. 
        """
    )

with st.expander("Langkah Saat Menghadapi Banjir"):
    st.markdown(
        """
        - Segera matikan listrik dan cabut peralatan elektronik. 
        - Pindahkan barang penting ke tempat yang lebih tinggi. 
        - Siapkan tas darurat berisi dokumen penting, obat, makanan, dan pakaian. 
        - Ikuti arahan dari petugas jika dilakukan evakuasi. 
        - Jangan memaksakan diri melewati arus banjir yang deras. 
        """
    )

# Informasi Metode Clustering DENCLUE
st.markdown("---")
st.subheader("Metode Clustering: DENCLUE")

st.markdown(
    """
    Sistem ini menggunakan algoritma **DENCLUE (Density-Based Clustering)** untuk menganalisis dan mengelompokkan data bencana banjir. 
    DENCLUE adalah metode untuk menemukan cluster (kelompok) berdasarkan kepadatan (density) data, yang sangat cocok 
    untuk menemukan area dengan tingkat risiko yang berbeda-beda.
    """
)

with st.expander("Apa itu DENCLUE? (Penjelasan)"):
    st.markdown(
        """
        **DENCLUE (Density-Based Clustering)** adalah algoritma clustering yang bekerja dengan cara memodelkan data sebagai sekumpulan "bukit" dan "lembah" berdasarkan kepadatan (density).
        
        Tujuan utamanya adalah untuk menemukan **"Density Attractors"** (Penarik Kepadatan), 
        yang dapat dianggap sebagai **puncak bukit** virtual dari kepadatan data. Cluster kemudian 
        didefinisikan sebagai area di sekitar attractor ini.
        
        ---
        
        DENCLUE memiliki dua parameter utama yang sangat penting:
        
        1.  **Sigma (σ) / Bandwidth:** Ini adalah parameter yang paling krusial. Sigma menentukan seberapa "luas" pengaruh satu titik data terhadap area di sekitarnya.
            * **Sigma kecil** akan menghasilkan "bukit" kepadatan yang tajam dan sempit, yang mungkin memecah data menjadi banyak cluster kecil.
            * **Sigma besar** akan menghasilkan "bukit" yang lebih landai dan lebar, yang mungkin menggabungkan beberapa cluster yang seharusnya terpisah.
            
        2.  **Threshold (ρ_min) / Ambang Batas Kepadatan:** Ini adalah nilai kepadatan minimum yang harus dimiliki oleh sebuah "puncak bukit" (Attractor) agar dapat dianggap sebagai pusat cluster yang valid. 
            * Jika sebuah puncak bukit memiliki nilai kepadatan di bawah threshold ini, ia (dan titik-titik data yang mengikutinya) akan dianggap sebagai **noise** (data pengganggu).
        """
    )

with st.expander("Apa yang Dimaksud Kepadatan (Density)?"):
    st.markdown(
        """
        Secara umum dalam analisis data, **Kepadatan (Density)** adalah ukuran seberapa 'ramai' atau 
        'padat' titik-titik data berkumpul dalam suatu ruang (area) tertentu.
        
        Ini adalah konsep inti dalam clustering berbasis kepadatan:
        * Area di mana banyak titik data berkumpul berdekatan satu sama lain dianggap sebagai area dengan **Kepadatan Tinggi**.
        * Area di mana titik-titik data tersebar dan letaknya berjauhan satu sama lain dianggap sebagai area dengan **Kepadatan Rendah**.
        """
    )

with st.expander("Bagaimana Tahapan Kerja DENCLUE?"):
    st.markdown(
        """
        Dibawah ini merupakan langkah-langkah utama dalam
        menggunakan metode clustering DENCLUE:
        
        1.  **Preprocessing (normalisasi dan pembersihan):** Data diproses terlebih dahulu agar setiap fitur memiliki
            skala yang sama dan adil dalam perhitungan, seringkali menggunakan standarisasi (z-score).
        
        2.  **Fungsi Gaussian Kernel:** Memilih fungsi matematis (dalam kasus ini, Gaussian) untuk menghitung
            seberapa besar kontribusi atau pengaruh tiap titik data terhadap kepadatan di sekitarnya.
        
        3.  **Perhitungan Density Function:** Menghitung nilai kepadatan total di setiap titik data dengan menjumlahkan semua kontribusi
            dari titik-titik data lain di sekitarnya.
        
        4.  **Gradien Gaussian Density:** Menghitung arah "pendakian" (gradien) atau arah kenaikan kepadatan tercepat 
            dari setiap titik data.
        
        5.  **Penarikan Densitas (Density Attractor Search):** Setiap titik data "bergerak" mengikuti arah gradiennya (mendaki bukit kepadatan) 
            secara berulang (iterasi) hingga mencapai puncak kepadatan lokal. Puncak ini disebut **Density Attractor**.
        
        6.  **Nilai Attractor Densitas (Cluster Assignment):** Titik-titik data yang bergerak ke *Attractor* yang sama (atau attractor yang terhubung) 
            dikelompokkan ke dalam satu cluster. Nilai kepadatan di attractor ini juga menjadi dasar 
            untuk mengkategorikan cluster dan memisahkannya dari noise.
        """
    )
    
# Tombol Mengarahkan ke halaman Clustering
st.markdown("---")
st.subheader("Mulai Eksperimen")

st.markdown(
    """
    User dapat melakukan eksperimen clustering menggunakan dataset yang telah disediakan. 
    Silakan menuju halaman Dataset untuk mengunduh file data yang akan digunakan 
    pada halaman **Clustering & Analisis**.
    """
)

if st.button("Lihat Halaman Dataset"):
    st.switch_page("pages/4_Dataset.py")