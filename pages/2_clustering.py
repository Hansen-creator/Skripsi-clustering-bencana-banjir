import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from geopy.geocoders import Nominatim, GoogleV3
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import time

# Streamlit Config
st.set_page_config(page_title="Clustering - DENCLUE", layout="wide")
hide_default_format = """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
        [data-testid="stSidebar"] { padding-top: 0rem; }
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    try:
        st.switch_page("main.py")
    except Exception:
        pass

try:
    from sidebar import show_sidebar
    show_sidebar()
except ImportError:
    st.sidebar.warning("File 'sidebar.py' tidak ditemukan. Sidebar kustom tidak akan ditampilkan.")

st.title("Clustering")

uploaded_file = st.file_uploader(
    "Upload File Data",
    type=None,
    help="""
    Unggah file data bencana. Sistem akan memvalidasi apakah file ini berformat .csv
    """
)

# Parameter DENCLUE 
SIGMA = 0.6      # Nilai sigma untuk kernel Gaussian
THRESHOLD = 0.35 # Nilai threshold untuk density

st.info(f"**Parameter DENCLUE yang digunakan:** Sigma (σ) = {SIGMA}, Threshold (ρ_min) = {THRESHOLD}")

# Sidebar hanya untuk navigasi
CACHE_FILE = "cache_geocoding.json"
GEOCODING_LOCK = threading.Lock()

# Fungsi Validasi File dan Data
def validate_file_extension(filename):
    """Validasi ekstensi file"""
    allowed_extensions = {'.csv'}
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in allowed_extensions

def validate_csv_structure(df):
    """Validasi struktur CSV memiliki kolom yang diperlukan"""
    required_columns = ['Tanggal Kejadian', 'Kejadian', 'Kabupaten', 'Provinsi']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Kolom wajib tidak ditemukan: {', '.join(missing_columns)}"
    
    return True, "Struktur CSV valid"

def validate_data_quality(df):
    """Validasi kualitas data"""
    issues = []
    
    # Cek jumlah data
    if len(df) < 5:
        issues.append(f"Data terlalu sedikit: hanya {len(df)} baris. Minimal diperlukan 5 baris data.")
    
    # Cek missing values pada kolom kritis
    critical_columns = ['Kabupaten', 'Provinsi']
    for col in critical_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues.append(f"Kolom '{col}' memiliki {missing_count} nilai kosong")
    
    # Cek duplikasi data
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Ditemukan {duplicate_count} data duplikat")
    
    # Cek variasi data
    if 'Kejadian' in df.columns:
        unique_events = df['Kejadian'].nunique()
        if unique_events < 2:
            issues.append("Variasi kejadian sangat terbatas")
    
    return issues

def validate_numeric_columns(df):
    """Validasi kolom numerik untuk clustering"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Tahun' in numeric_cols:
        numeric_cols.remove('Tahun')
    
    if len(numeric_cols) < 2:
        return False, numeric_cols, "Minimal diperlukan 2 kolom numerik untuk clustering"
    
    # Cek apakah kolom numerik memiliki variasi data
    for col in numeric_cols:
        if df[col].nunique() < 2:
            return False, numeric_cols, f"Kolom numerik '{col}' tidak memiliki variasi data yang cukup"
    
    return True, numeric_cols, "Kolom numerik valid"

# Fungsi Preprocessing (MEMOIZATION)
@st.cache_data(show_spinner="Pre-processing data...", ttl=600)
def advanced_preprocessing(df, selected_cols=None):
    """Preprocessing data yang lebih advanced dengan StandardScaler (Cepat karena di-cache)"""
    
    try:
        # Buat copy dataframe
        df_processed = df.copy()
        
        # Handle Missing Values 
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median') 
            df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
        
        # Untuk kolom kategorikal
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        # Normalisasi/Scaling dengan Standar 
        scaler = StandardScaler()
        
        # Validasi kolom yang dipilih masih ada
        available_cols = [col for col in selected_cols if col in df_processed.columns]
        if len(available_cols) < 2:
            raise ValueError(f"Hanya {len(available_cols)} kolom yang tersedia untuk scaling. Minimal 2 kolom diperlukan.")
        
        X = scaler.fit_transform(df_processed[available_cols])
        
        return X, available_cols
    
    except Exception as e:
        raise Exception(f"Error dalam preprocessing: {str(e)}")

def normalize_kabupaten_names(df):
    """Normalisasi nama kabupaten untuk konsistensi"""
    try:
        df_clean = df.copy()
        
        if 'Kabupaten' in df_clean.columns:
            df_clean['Kabupaten'] = (
                df_clean['Kabupaten']
                .astype(str)
                .str.upper()
                .str.strip()
                .str.replace(r'[^\w\s]', '', regex=True)  # Hapus karakter khusus
                .str.replace(r'\s+', ' ', regex=True)  # Normalisasi spasi
            )
        
        if 'Provinsi' in df_clean.columns:
            df_clean['Provinsi'] = (
                df_clean['Provinsi']
                .astype(str)
                .str.upper()
                .str.strip()
                .str.replace(r'[^\w\s]', '', regex=True)
                .str.replace(r'\s+', ' ', regex=True)
            )
        
        return df_clean
    except Exception as e:
        raise Exception(f"Error dalam normalisasi nama: {str(e)}")

# Fungsi untuk Ekstraksi Tahun dari Tanggal
def extract_year_from_date(df, date_column='Tanggal Kejadian'):
    """Mengekstrak tahun dari kolom tanggal dan menambahkannya ke dataframe"""
    try:
        df_processed = df.copy()
        
        if date_column in df_processed.columns:
            try:
                # Coba konversi ke datetime
                df_processed[date_column] = pd.to_datetime(df_processed[date_column], errors='coerce')
                
                # Ekstrak tahun
                df_processed['Tahun'] = df_processed[date_column].dt.year
                
                # Filter out tahun yang tidak valid
                valid_years = df_processed['Tahun'].dropna()
                if len(valid_years) > 0:
                    st.write(f"Rentang tahun: {int(valid_years.min())} - {int(valid_years.max())}")
                else:
                    st.warning("⚠️ Tidak dapat mengekstrak tahun dari kolom tanggal")
                    df_processed['Tahun'] = None
                    
            except Exception as e:
                st.warning(f"⚠️ Tidak dapat memproses kolom tanggal: {e}")
                df_processed['Tahun'] = None
        else:
            st.warning(f"⚠️ Kolom '{date_column}' tidak ditemukan dalam dataset")
            df_processed['Tahun'] = None
        
        return df_processed
    except Exception as e:
        raise Exception(f"Error dalam ekstraksi tahun: {str(e)}")

# Fungsi DENCLUE (core function, tidak di-cache)
def gaussian_kernel(distance, sigma):
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

def density_function(X, sigma):
    n = X.shape[0]
    density = np.zeros(n)
    for i in range(n):
        dist = np.linalg.norm(X - X[i], axis=1)
        density[i] = np.sum(gaussian_kernel(dist, sigma))
    return density

def gradient_density(X, sigma):
    n, d = X.shape
    gradients = np.zeros((n, d))
    for i in range(n):
        diff = X - X[i]
        dist = np.linalg.norm(diff, axis=1)
        weights = gaussian_kernel(dist, sigma)
        gradients[i] = np.sum((weights[:, None] * diff), axis=0)
    return gradients

# Fungsi DENCLUE untuk Clustering (MEMOIZATION)
@st.cache_data(show_spinner="Menghitung density dan cluster DENCLUE...", ttl=600)
def denclue_clustering_cached(X, sigma, threshold, df_length):
    """
    Menjalankan algoritma DENCLUE (cached).
    df_length digunakan hanya sebagai parameter untuk reset cache jika data berubah.
    """
    
    st.info("💡 DENCLUE (Density-based Clustering) mengidentifikasi cluster berdasarkan puncak kepadatan data. Cluster 0 adalah kepadatan tertinggi (Severity Ringan), Cluster 2 adalah kepadatan terendah (Severity Berat)."
    " ")
    
    try:
        if X.shape[0] == 0:
            raise ValueError("Data matrix kosong, tidak dapat melakukan clustering")
        
        density = density_function(X, sigma)
        
        # Validasi density values
        if np.all(density == 0):
            raise ValueError("Semua density values nol, kemungkinan parameter sigma tidak sesuai")
        
        if (density.max() - density.min()) == 0:
            st.warning("Semua data memiliki nilai density yang identik. Hasil clustering mungkin tidak signifikan.")
            clusters = ["Cluster 1"] * len(density) # Default ke Cluster 1
            return clusters, density

        # Normalisasi density
        density_normalized = (density - density.min()) / (density.max() - density.min())
        
        clusters = []
        
        # Definisi Threshold untuk Cluster Numerik
        low_thresh = threshold   
        med_thresh = 2 * threshold
        
        for d in density_normalized:
            if d < low_thresh:
                clusters.append("Cluster 2") # Severity Berat (Density Rendah)
            elif d < med_thresh:
                clusters.append("Cluster 1") # Severity Sedang
            else:
                clusters.append("Cluster 0") # Severity Ringan (Density Tinggi)

        return clusters, density
    
    except Exception as e:
        raise Exception(f"Error dalam clustering DENCLUE: {str(e)}")

# Fungsi Geocoding (MEMOIZATION)

def load_cache():
    """Load cache dengan thread safety"""
    try:
        with GEOCODING_LOCK:
            if os.path.exists(CACHE_FILE):
                try:
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        return json.load(f)
                except:
                    return {}
            return {}
    except Exception as e:
        st.warning(f"Error loading cache: {e}")
        return {}

def save_cache(cache):
    """Save cache dengan thread safety"""
    try:
        with GEOCODING_LOCK:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Error saving cache: {e}")

def geocode_nominatim(kabupaten, provinsi):
    """Geocoding menggunakan Nominatim dengan 'provinsi'"""
    try:
        geolocator = Nominatim(user_agent=f"denclue_app_{np.random.randint(1000)}")
        query = f"{kabupaten}, {provinsi}, Indonesia"
        location = geolocator.geocode(query, timeout=10)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        pass
    return (None, None)

def geocode_opencage(kabupaten, provinsi):
    """Geocoding menggunakan OpenCage (free tier available) dengan 'provinsi'"""
    try:
        API_KEY = st.secrets.get("OPENCAGE_API_KEY", "")
        if not API_KEY:
            return (None, None)
            
        url = f"https://api.opencagedata.com/geocode/v1/json"
        params = {
            'q': f'{kabupaten}, {provinsi}, Indonesia',
            'key': API_KEY,
            'limit': 1
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                lat = data['results'][0]['geometry']['lat']
                lng = data['results'][0]['geometry']['lng']
                return (lat, lng)
    except Exception as e:
        pass
    return (None, None)

def geocode_location(kabupaten, provinsi):
    """Mencoba multiple geocoding services dengan fallback"""
    
    if not kabupaten or pd.isna(kabupaten) or kabupaten == 'Unknown':
        return (None, None)
    
    provinsi_str = str(provinsi).strip()
    
    # Coba OpenCage pertama
    lat, lon = geocode_opencage(kabupaten, provinsi_str)
    if lat is not None and lon is not None:
        return (lat, lon)

    lat, lon = geocode_nominatim(kabupaten, provinsi_str)
    if lat is not None and lon is not None:
        return (lat, lon)
    
    return (None, None)

def process_single_location(args):
    """Process single location untuk threading"""
    idx, kabupaten, provinsi, cache = args
    kabupaten_str = str(kabupaten).strip()
    provinsi_str = str(provinsi).strip()
    
    if not kabupaten_str or kabupaten_str == 'Unknown':
        return idx, (None, None)
    
    cache_key = f"{kabupaten_str}|{provinsi_str}"
    
    # Cek cache dulu
    if cache_key in cache:
        cached_data = cache[cache_key]
        return idx, (cached_data["lat"], cached_data["lon"])    
    
    # Geocode jika tidak ada di cache
    lat, lon = geocode_location(kabupaten_str, provinsi_str)
    
    # Update cache jika berhasil
    if lat is not None and lon is not None:
        with GEOCODING_LOCK:
            cache[cache_key] = {"lat": lat, "lon": lon}
    
    return idx, (lat, lon)

def geocode_kabupaten_fast(df, max_workers=10):
    """Geocoding dengan threading untuk performa lebih cepat"""
    st.info("Memulai proses geocoding. Koordinat akan di-cache untuk penggunaan berikutnya. ")
    try:
        cache = load_cache()
        
        if 'Kabupaten' not in df.columns or 'Provinsi' not in df.columns:
            raise ValueError("Kolom 'Kabupaten' dan 'Provinsi' tidak ditemukan dalam dataset")
        
        kabupaten_list = df.get("Kabupaten", pd.Series([""] * len(df)))
        provinsi_list = df.get("Provinsi", pd.Series([""] * len(df))) # Ambil list provinsi
        
        st.info(f"📍 Mecari titik koordinasi...")
        unique_locations = set()
        for kab, prov in zip(kabupaten_list, provinsi_list):
            kab_str = str(kab).strip()
            prov_str = str(prov).strip()
            if not kab_str or kab_str == 'Unknown':
                continue
            unique_locations.add(f"{kab_str}|{prov_str}")
        
        total_unique = len(unique_locations)
        locations_to_geocode = 0
        for loc in unique_locations:
            if loc not in cache:
                locations_to_geocode += 1
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        args_list = [(idx, kab, prov, cache) for idx, (kab, prov) in enumerate(zip(kabupaten_list, provinsi_list))]
        
        lats = [None] * len(df)
        lons = [None] * len(df)
        
        start_time = time.time()
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single_location, args): args[0] 
                for args in args_list
            }
            
            for future in as_completed(future_to_idx):
                idx, (lat, lon) = future.result()
                lats[idx] = lat
                lons[idx] = lon
                completed += 1
                
                # Update progress
                progress = completed / len(df)
                progress_bar.progress(progress)
                
                elapsed = time.time() - start_time
                status_text.text(f"Progress: {completed}/{len(df)} ({progress:.1%}) - {elapsed:.1f} detik")
        
        # Save cache sekali di akhir
        save_cache(cache)
        
        elapsed = time.time() - start_time
        success_count = sum(1 for lat in lats if lat is not None)
        
        return lats, lons
    
    except Exception as e:
        raise Exception(f"Error dalam proses geocoding: {str(e)}")

# Fallback Geocoding untuk data yang gagal
def get_province_coordinates(provinsi):
    """Koordinat default berdasarkan provinsi untuk fallback"""
    province_coords = {
        "ACEH": (4.695135, 96.749399), "SUMATERA UTARA": (2.115354, 99.545097),
        "SUMATERA BARAT": (-0.739939, 100.800005), "RIAU": (0.293347, 101.706829),
        "JAMBI": (-1.485183, 102.438058), "SUMATERA SELATAN": (-2.990934, 104.756554),
        "BENGKULU": (-3.577847, 102.346387), "LAMPUNG": (-4.558584, 105.406807),
        "KEPULAUAN BANGKA BELITUNG": (-2.496068, 106.428529), "KEPULAUAN RIAU": (3.945651, 108.142866),
        "DKI JAKARTA": (-6.208763, 106.845599), "JAWA BARAT": (-6.914744, 107.609810),
        "JAWA TENGAH": (-7.150975, 110.140259), "DI YOGYAKARTA": (-7.795580, 110.369490),
        "JAWA TIMUR": (-7.250445, 112.768845), "BANTEN": (-6.120000, 106.150276),
        "BALI": (-8.409518, 115.188919), "NUSA TENGGARA BARAT": (-8.652933, 117.361647),
        "NUSA TENGGARA TIMUR": (-8.657382, 121.079370), "KALIMANTAN BARAT": (-0.278781, 111.475285),
        "KALIMANTAN TENGAH": (-1.681488, 113.382354), "KALIMANTAN SELATAN": (-3.092642, 115.283758),
        "KALIMANTAN TIMUR": (0.538659, 116.419389), "KALIMANTAN UTARA": (3.725737, 116.646965),
        "SULAWESI UTARA": (0.624693, 123.975001), "SULAWESI TENGAH": (-1.430025, 121.445617),
        "SULAWESI SELATAN": (-3.668799, 119.974053), "SULAWESI TENGGARA": (-3.644522, 121.894852),
        "GORONTALO": (0.699937, 122.446723), "SULAWESI BARAT": (-2.497451, 119.288494),
        "MALUKU": (-3.238462, 130.145273), "MALUKU UTARA": (1.570999, 127.808769),
        "PAPUA BARAT": (-1.336115, 133.174716), "PAPUA": (-4.269928, 138.080353)
    }
    
    if not provinsi or pd.isna(provinsi):
        return (-2.548926, 118.014863)  
    
    provinsi_upper = str(provinsi).upper()
    for key, coords in province_coords.items():
        if key in provinsi_upper or provinsi_upper in key:
            return coords
    
    return (-2.548926, 118.014863)  # Default ke tengah Indonesia

def apply_fallback_coordinates(df):
    """Apply fallback coordinates untuk data yang gagal di-geocode"""
    try:
        
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            raise ValueError("Kolom Latitude/Longitude tidak ditemukan")
        
        fallback_applied = 0
        for idx, row in df.iterrows():
            if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
                provinsi = row.get('Provinsi', '')
                lat, lon = get_province_coordinates(provinsi)
                df.at[idx, 'Latitude'] = lat
                df.at[idx, 'Longitude'] = lon
                fallback_applied += 1
        
        if fallback_applied > 0:
            st.info(f"Selesai mencari titik koordinat ✅ ")
        
        return df
    except Exception as e:
        raise Exception(f"Error dalam penerapan fallback coordinates: {str(e)}")

# FUNGSI ANALISIS TREND BANJIR
def detect_banjir_data(df):
    """Mendeteksi dan memfilter data banjir secara spesifik"""
    try:
        banjir_keywords = [
            'banjir', 'BANJIR', 'flood', 'FLOOD', 'genangan', 'GENANGAN',
            'banjir bandang', 'BANJIR BANDANG', 'rob', 'ROB'
        ]
        
        # Cari kolom yang mungkin berisi jenis bencana
        disaster_col_candidates = [
            'Jenis Bencana', 'Jenis_Bencana', 'jenis_bencana', 'Bencana', 'bencana',
            'Kategori', 'kategori', 'Jenis', 'jenis', 'Kejadian', 'kejadian'
        ]
        
        banjir_data = df.copy()
        banjir_detected = False
        
        for col in disaster_col_candidates:
            if col in df.columns:
                mask = df[col].astype(str).str.lower().str.contains('|'.join([k.lower() for k in banjir_keywords]))
                if mask.any():
                    banjir_data = df[mask].copy()
                    banjir_detected = True
                    st.success(f"✅ Data banjir terdeteksi di kolom: {col}")
                    st.write(f"📊 Jumlah data banjir: {len(banjir_data)} dari {len(df)} total data")
                    break
        
        if not banjir_detected:
            st.warning("⚠️ Tidak terdeteksi data banjir spesifik. Menggunakan semua data.")
            banjir_data = df.copy()
        
        return banjir_data
    except Exception as e:
        st.error(f"Error dalam deteksi data banjir: {e}")
    return df

def analyze_banjir_trends(df):
    """Analisis trend banjir yang fokus pada distribusi spasial"""
    
    try:
        st.header("🌊 ANALISIS DISTRIBUSI DAMPAK BANJIR")
        
        # Validasi data 
        if len(df) < 5:
            st.error("❌ Data terlalu sedikit untuk analisis trend. Minimal diperlukan 5 baris data.")
            return df
        
        if 'Provinsi' not in df.columns:
            st.error("❌ Kolom 'Provinsi' tidak ditemukan dalam dataset.")
            return df
        
        banjir_df = detect_banjir_data(df)
        
        st.subheader("⚙️ Pengaturan Analisis Banjir")
        
        all_provinsi = sorted(banjir_df['Provinsi'].dropna().unique().tolist())
        if not all_provinsi:
            st.error("❌ Tidak ada data Provinsi yang tersedia")
            return banjir_df
            
        selected_provinsi_trend = st.selectbox(
            "Pilih Provinsi untuk Analisis:",
            all_provinsi,
            help="Pilih provinsi untuk melihat distribusi banjir di provinsi tersebut"
        )
        
        provinsi_data = banjir_df[banjir_df['Provinsi'] == selected_provinsi_trend]
        
        available_kabupaten_in_prov = sorted(provinsi_data['Kabupaten'].dropna().unique().tolist())

        kabupaten_counts_all = pd.DataFrame()
        top_5_kabupaten_default = []
        if len(provinsi_data) > 0:
            kabupaten_counts_all = provinsi_data['Kabupaten'].value_counts().reset_index()
            kabupaten_counts_all.columns = ['Kabupaten', 'Jumlah_Kejadian']
            top_5_kabupaten_default = kabupaten_counts_all.head(5)['Kabupaten'].tolist()

        col_setting1, col_setting2 = st.columns(2)
        
        with col_setting1:
            top_n = st.slider(
                "Jumlah Kabupaten Teratas (untuk Insight):", 
                min_value=5,
                max_value=30,
                value=10,
                help="Angka ini digunakan untuk menentukan kabupaten prioritas di Insight."
            )

        with col_setting2:
            if available_kabupaten_in_prov:
                selected_kabupaten_trend = st.multiselect(
                    "Pilih Kabupaten (untuk Line Chart):",
                    options=available_kabupaten_in_prov,
                    default=top_5_kabupaten_default,
                    help="Pilih kabupaten untuk ditampilkan di grafik tren tahunan."
                )
            else:
                st.info("Tidak ada data kabupaten di provinsi ini.")
                selected_kabupaten_trend = []
        
        st.subheader(f"📈 Tren Kejadian Banjir per Kabupaten di {selected_provinsi_trend}")
        
        if len(provinsi_data) > 0 and 'Tahun' in provinsi_data.columns:
            # Validasi data tahun
            provinsi_data['Tahun'] = pd.to_numeric(provinsi_data['Tahun'], errors='coerce')
            provinsi_data_trend = provinsi_data.dropna(subset=['Tahun'])
            
            # Filter berdasarkan kabupaten yang dipilih 
            if selected_kabupaten_trend:
                provinsi_data_trend_filtered = provinsi_data_trend[
                    provinsi_data_trend['Kabupaten'].isin(selected_kabupaten_trend)
                ]
            else:
                st.warning("Silakan pilih minimal satu kabupaten di filter 'Pilih Kabupaten (untuk Line Chart)' untuk melihat tren.")
                provinsi_data_trend_filtered = pd.DataFrame() 
            
            if not provinsi_data_trend_filtered.empty:
                trend_data = provinsi_data_trend_filtered.groupby(['Tahun', 'Kabupaten']).size().reset_index(name='Jumlah_Kejadian')
                
                if not trend_data.empty:
                    fig_trend, ax_trend = plt.subplots(figsize=(14, 8))
                    
                    sns.lineplot(
                        data=trend_data, 
                        x='Tahun', 
                        y='Jumlah_Kejadian', 
                        hue='Kabupaten', 
                        marker='o', 
                        ax=ax_trend
                    )
                    
                    ax_trend.set_title(f"Tren Kejadian Banjir Tahunan per Kabupaten di {selected_provinsi_trend}")
                    ax_trend.set_xlabel("Tahun")
                    ax_trend.set_ylabel("Jumlah Kejadian Banjir")
                    ax_trend.grid(True, alpha=0.3)
                    
                    ax_trend.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Kabupaten')
                    
                    plt.tight_layout()
                    st.pyplot(fig_trend)
                else:
                    st.warning("Tidak ada data agregat untuk kabupaten yang dipilih pada rentang tahun ini.")
                    
            elif selected_kabupaten_trend: 
                st.warning("Tidak ada data kejadian yang valid (dengan data tahun) untuk kabupaten yang dipilih.")
            
        st.subheader("🗺️ Distribusi Banjir di Seluruh Provinsi")
        
        province_heatmap = banjir_df['Provinsi'].value_counts().reset_index()
        province_heatmap.columns = ['Provinsi', 'Jumlah_Kejadian']
        province_heatmap = province_heatmap.sort_values('Jumlah_Kejadian', ascending=False)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            total_all_incidents = province_heatmap['Jumlah_Kejadian'].sum()
            st.metric("Total Kejadian Semua Provinsi", total_all_incidents)
        
        with col_stat2:
            avg_all_incidents = province_heatmap['Jumlah_Kejadian'].mean()
            st.metric("Rata-rata per Provinsi", f"{avg_all_incidents:.1f}")
        
        with col_stat3:
            if len(province_heatmap) > 0:
                max_province = province_heatmap.iloc[0]['Provinsi']
                max_incidents = province_heatmap.iloc[0]['Jumlah_Kejadian']
                st.metric("Provinsi Terbanyak", f"{max_province} ({max_incidents})")
        
        with col_stat4:
            total_provinces = len(province_heatmap)
            st.metric("Total Provinsi", total_provinces)
        
        # Visualisasi Bar Chart (Sesuai permintaan)
        if not province_heatmap.empty:
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            y_pos = np.arange(len(province_heatmap))
            
            bars = ax2.barh(y_pos, province_heatmap['Jumlah_Kejadian'], color='#3498db')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(province_heatmap['Provinsi'])
            ax2.invert_yaxis()
            ax2.set_xlabel('Jumlah Kejadian Banjir')
            ax2.set_title('Distribusi Kejadian Banjir per Provinsi (Semua Indonesia)')
            ax2.grid(True, alpha=0.3)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                         f'{int(width)}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.warning("Tidak ada data banjir untuk membuat visualisasi distribusi provinsi.")

        st.subheader("💡 Rekomendasi Berdasarkan Analisis (Insight Cepat)")
        
        insights = []
        if len(provinsi_data) > 0:
            current_province_data = province_heatmap[province_heatmap['Provinsi'] == selected_provinsi_trend]
            if len(current_province_data) > 0:
                current_incidents = current_province_data['Jumlah_Kejadian'].iloc[0]
                rank = current_province_data.index[0] + 1
                total_provinces = len(province_heatmap)
                
                insights.append(f"**{selected_provinsi_trend}** berada di **peringkat {rank}** dari {total_provinces} provinsi dengan **{current_incidents}** kejadian banjir")

        # Insight untuk kabupaten teratas (menggunakan top_n yang dipilih pengguna)
        if len(provinsi_data) > 0 and not kabupaten_counts_all.empty:
            top_kabupaten_list = kabupaten_counts_all.head(min(top_n, len(kabupaten_counts_all)))['Kabupaten'].tolist()
            insights.append(f"**{min(top_n, len(kabupaten_counts_all))} Kabupaten prioritas** di {selected_provinsi_trend}: {', '.join(top_kabupaten_list)}")
        
        # Insight umum
        if len(province_heatmap) > 0:
            top_3_provinces = province_heatmap.head(3)['Provinsi'].tolist()
            insights.append(f"**3 provinsi dengan frekuensi banjir tertinggi** secara nasional: {', '.join(top_3_provinces)}")
        
        # Tampilkan insights
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")
        
        return banjir_df
    
    except Exception as e:
        st.error(f"❌ Error dalam analisis trend banjir: {str(e)}")
        return df

# Fungsi untuk Analisis dan Interpretasi
def analyze_cluster_causes(df, numeric_cols):
    """Menganalisis penyebab clustering berdasarkan fitur numerik"""
    try:
        causes = {}
        
        # Daftar Cluster Numerik
        for cluster in ["Cluster 0", "Cluster 1", "Cluster 2"]: 
            cluster_data = df[df["Cluster"] == cluster]
            if len(cluster_data) == 0:
                causes[cluster] = "Tidak ada data"
                continue
            
            feature_means = cluster_data[numeric_cols].mean()
            overall_means = df[numeric_cols].mean()
            
            # Hitung perbedaan dari rata-rata
            differences = feature_means - overall_means
            
            abs_differences = differences.abs().nlargest(2).index
            
            cause_desc = []
            for feature in abs_differences:
                std_dev = df[numeric_cols].std()[feature]
                if abs(differences[feature]) > 0.1 * std_dev: 
                    if differences[feature] > 0:
                        cause_desc.append(f"nilai **{feature}** cenderung **lebih tinggi**")
                    else:
                        cause_desc.append(f"nilai **{feature}** cenderung **lebih rendah**")
            
            if not cause_desc:
                causes[cluster] = "Karakteristik umum, tidak menonjol"
            else:
                causes[cluster] = " & ".join(cause_desc)
            
        return causes
    except Exception as e:
        st.warning(f"Tidak dapat menganalisis penyebab cluster: {e}")
        return {"Cluster 0": "Analisis gagal", "Cluster 1": "Analisis gagal", "Cluster 2": "Analisis gagal"}

# Fungsi untuk menghilangkan duplikasi dan memastikan konsistensi
def ensure_unique_clusters(df):
    """Memastikan setiap kabupaten hanya ada di satu cluster"""
    try:
        if 'Kabupaten' not in df.columns:
            return df
        kabupaten_counts = df['Kabupaten'].value_counts()
        duplicates = kabupaten_counts[kabupaten_counts > 1]
        
        if len(duplicates) > 0:
            df_unique = df.sort_values('Density', ascending=False).drop_duplicates('Kabupaten', keep='first')
            return df_unique
        
        return df
    except Exception as e:
        st.warning(f"Error dalam menghilangkan duplikasi: {e}")
        return df

# Fungsi untuk Filter dan Navigasi Peta dengan LEGEND
def create_filtered_map(df, selected_provinsi=None, selected_kabupaten=None):
    """
    Membuat peta dengan filter dan navigasi ke lokasi tertentu.
    Data 'df' yang diterima DIASUMSIKAN SUDAH UNIK per Kabupaten.
    """
    
    try:
        # Filter data berdasarkan pilihan
        df_filtered = apply_filters(df, selected_provinsi, selected_kabupaten)
        
        # Tentukan pusat peta
        if len(df_filtered) == 0:
            df_filtered = df.copy() 

        if selected_kabupaten and selected_kabupaten != "Semua" and len(df_filtered[df_filtered['Kabupaten'] == selected_kabupaten]) > 0:
            kab_data = df_filtered[df_filtered['Kabupaten'] == selected_kabupaten].iloc[0]
            map_center = [kab_data["Latitude"], kab_data["Longitude"]]
            zoom_start = 10
        elif selected_provinsi and selected_provinsi != "Semua" and len(df_filtered) > 0:
            map_center = [df_filtered["Latitude"].mean(), df_filtered["Longitude"].mean()]
            zoom_start = 8
        else:
            if len(df_filtered) > 0:
                map_center = [df_filtered["Latitude"].mean(), df_filtered["Longitude"].mean()]
            else:
                map_center = [-2.548926, 118.014863] # Default Indonesia
            zoom_start = 5

        folium_map = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron")

        # Feature Group sesuai Cluster Numerik (Cluster 0, 1, 2)
        fg_c2 = folium.FeatureGroup(name='🔴 Cluster 2', show=True).add_to(folium_map)
        fg_c1 = folium.FeatureGroup(name='🟠 Cluster 1', show=True).add_to(folium_map)
        fg_c0 = folium.FeatureGroup(name='🟢 Cluster 0', show=True).add_to(folium_map)
        
        colors = {"Cluster 0": "green", "Cluster 1": "orange", "Cluster 2": "red", "Noise": "gray"}
        conclusions = {
            "Cluster 0": "Sedikit membutuhkan penanganan (Ringan)",
            "Cluster 1": "Cukup membutuhkan penanganan (Sedang)",
            "Cluster 2": "Sangat membutuhkan penanganan (Berat)",
            "Noise": "Tidak terklaster"
        }
        
        # Loop ini sekarang MENGULANGI DATA KABUPATEN UNIK
        for _, row in df_filtered.iterrows():
            cluster = row.get("Cluster", "Noise")
            
            conclusion = conclusions.get(cluster, "N/A")
            
            popup_text = f"""
            <b>{row.get('Kabupaten', 'N/A')}, {row.get('Provinsi', 'N/A')}</b><br>
            <hr style='margin: 3px 0;'>
            Cluster: <b>{cluster}</b><br>
            Kesimpulan: {conclusion}<br>
            <i>(Berdasarkan density tertinggi pada tahun {int(row.get('Tahun', 'N/A'))})</i>
            """

            try:
                # Beri radius lebih besar untuk cluster 2
                radius = 8 if cluster == "Cluster 2" else 6 
            except Exception:
                radius = 6
            
            marker = folium.CircleMarker(
                location=[row.get("Latitude", 0), row.get("Longitude", 0)],
                radius=radius,
                popup=popup_text,
                color=colors.get(cluster, 'blue'),
                fill=True,
                fill_color=colors.get(cluster, 'blue'),
                fill_opacity=0.8,
                tooltip=f"{row.get('Kabupaten', 'N/A')} - {cluster}"
            )
            
            if cluster == "Cluster 2": 
                marker.add_to(fg_c2) 
            elif cluster == "Cluster 1":
                marker.add_to(fg_c1)
            else:
                marker.add_to(fg_c0)

        if selected_kabupaten and selected_kabupaten != "Semua" and len(df_filtered[df_filtered['Kabupaten'] == selected_kabupaten]) > 0:
            folium.Marker(
                location=map_center,
                popup=f"<b>{selected_kabupaten}</b>",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(folium_map)
        
        folium.LayerControl().add_to(folium_map)

        # LEGEND KE PETA
        legend_html = '''
        <div style="
            position: fixed; 
            top: 10px; left: 50px; 
            width: 250px; 
            background-color: rgba(0, 0, 0, 0.8); 
            border: 1px solid #555; 
            z-index: 9999; 
            font-size: 14px; 
            color: white; 
            padding: 12px 15px; 
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
        ">
            <p style="margin: 0 0 8px 0; font-weight: bold; text-align: center; font-size: 15px; color: #FFD700;">
                🎯 KATEGORI CLUSTER
            </p>
            <p style="margin: 4px 0;">
                <i class="fa fa-circle" style="color:red; font-size:12px"></i> <b>Cluster 2 </b>
            </p>
            <p style="margin: 4px 0;">
                <i class="fa fa-circle" style="color:orange; font-size:12px"></i> <b>Cluster 1</b>
            </p>
            <p style="margin: 4px 0;">
                <i class="fa fa-circle" style="color:limegreen; font-size:12px"></i> <b>Cluster 0</b>
            </p>
        </div>
        '''
        folium_map.get_root().html.add_child(folium.Element(legend_html))
        
        return folium_map, df_filtered
    
    except Exception as e:
        st.error(f"❌ Error dalam membuat peta: {str(e)}")
        default_map = folium.Map(location=[-2.548926, 118.014863], zoom_start=5)
        return default_map, df

def apply_filters(df, selected_provinsi=None, selected_kabupaten=None):
    """Menerapkan filter pada dataframe"""
    try:
        df_filtered = df.copy()
        
        if selected_provinsi and selected_provinsi != "Semua":
            df_filtered = df_filtered[df_filtered['Provinsi'] == selected_provinsi]
        
        if selected_kabupaten and selected_kabupaten != "Semua":
            df_filtered = df_filtered[df_filtered['Kabupaten'] == selected_kabupaten]
        
        return df_filtered
    except Exception as e:
        st.warning(f"Error dalam menerapkan filter: {e}")
        return df

def get_dynamic_colors_and_labels(df_sample):
    """Helper untuk mendapatkan warna dan label dinamis untuk plot"""
    cluster_names_sorted = sorted([c for c in df_sample['Cluster'].unique() if c.startswith('Cluster')], key=lambda x: int(x.split(' ')[1]))
    
    all_names = cluster_names_sorted + ['Noise']
    
    palette = {
        'Cluster 0': 'green', 
        'Cluster 1': 'orange', 
        'Cluster 2': 'red',
        'Noise': '#333333'
    }
    
    name_to_id = {name: i for i, name in enumerate(cluster_names_sorted)}
    
    plot_colors = [palette[name] for name in cluster_names_sorted]
    
    return cluster_names_sorted, all_names, palette, name_to_id, plot_colors

def create_silhouette_plot(df, numeric_cols):
    """
    Menghitung dan menampilkan Silhouette Score beserta interpretasi dan waktu komputasi.
    Noise dikecualikan dari skor.
    """
    try:
        start_time = time.time() # Mulai timer

        #  Filter noise untuk silhouette score 
        df_sample_no_noise = df[df['Cluster'].str.startswith('Cluster')].copy()
        
        if len(df_sample_no_noise) < 2:
            st.warning("⚠️ Tidak cukup data (non-noise) untuk silhouette score.")
            return

        cluster_names, _, _, name_to_id, _ = get_dynamic_colors_and_labels(df_sample_no_noise)
        
        if not name_to_id: # Jika tidak ada cluster selain Noise
              st.warning("⚠️ Hanya ada data 'Noise', silhouette score tidak dapat dihitung.")
              return
            
        cluster_labels_no_noise = df_sample_no_noise['Cluster'].map(name_to_id).astype(int)

        if len(np.unique(cluster_labels_no_noise)) < 2:
            st.warning("⚠️ Silhouette score tidak dapat dihitung karena hanya ada 1 cluster (non-noise) dalam data yang difilter.")
            return
            
        # Reduksi dimensi
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df_sample_no_noise[numeric_cols])
        
        # Hitung Silhouette Score
        silhouette_avg = silhouette_score(X_pca, cluster_labels_no_noise)
        
        end_time = time.time() # Hentikan timer
        computation_time = end_time - start_time

        # Tampilkan Score dan Interpretasi
        st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
        
        # 
        
        if silhouette_avg > 0.5:
            st.success("✅ Clustering berkualitas baik (silhouette score > 0.5)")
        elif silhouette_avg > 0.25:
            st.warning("⚠️ Clustering berkualitas sedang (silhouette score 0.25-0.5)")
        else:
            st.error("❌ Clustering berkualitas rendah (silhouette score < 0.25)")

        # Tampilkan Waktu Komputasi
        st.info(f"⏱️ Waktu komputasi Silhouette Score: {computation_time:.2f} detik")
            
    except ImportError:
          st.error("Modul Scikit-learn tidak ditemukan. Silakan install: pip install scikit-learn")
    except Exception as e:
        st.warning(f"Tidak dapat menghitung silhouette score: {e}")

def create_scatter_plots(df, numeric_cols):
    """
    Membuat scatter plot matrix (pairplot) dengan coloring berdasarkan cluster.
    Data 'df' yang diterima adalah data KABUPATEN UNIK (representatif).
    Menggunakan SEMUA fitur yang dipilih.
    """
    try:
        df_sample = df.copy() 
        
        if len(numeric_cols) < 2:
            st.warning("Scatter matrix memerlukan minimal 2 fitur.")
            return

        # Tentukan palette warna
        colors = {'Cluster 0': 'green', 'Cluster 1': 'orange', 'Cluster 2': 'red'} 
        
        # Urutan untuk hue
        hue_order = ['Cluster 0', 'Cluster 1', 'Cluster 2']
        
        # Buat pairplot
        plot_data = df_sample[df_sample['Cluster'].isin(hue_order)][numeric_cols + ['Cluster']] # Filter out noise
        
        g = sns.pairplot(
            plot_data,
            vars=numeric_cols, # Kolom yang akan diplot
            hue='Cluster',       # Kolom untuk pewarnaan
            palette=colors,
            hue_order=hue_order, 
            diag_kind='kde',     # Gunakan Kernel Density Estimate di diagonal
            plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k', 'linewidth': 0.5},
            diag_kws={'fill': True}
        )
        
        # 
        
        g.fig.suptitle('Scatter Matrix - Hubungan Antar Fitur', y=1.02, fontsize=16)
        st.pyplot(g.fig)
        
    except Exception as e:
        st.warning(f"Tidak dapat membuat scatter matrix: {e}")

def create_boxplots(df, numeric_cols):
    """
    Membuat boxplot untuk setiap fitur berdasarkan cluster (versi Seaborn).
    Data 'df' yang diterima adalah data KABUPATEN UNIK (representatif).
    Menggunakan SEMUA fitur yang dipilih.
    """
    try:
        # Gunakan semua fitur yang dipilih
        selected_features = numeric_cols
        
        if len(selected_features) == 0:
            st.warning("Boxplot memerlukan minimal 1 fitur.")
            return
        
        # Tentukan layout subplot
        n_cols = 2
        n_rows = (len(selected_features) + n_cols - 1) // n_cols
        if n_rows == 0: n_rows = 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]  
        else:
            axes = axes.flatten() 

        colors = {'Cluster 0': 'green', 'Cluster 1': 'orange', 'Cluster 2': 'red'}
        order = ['Cluster 0', 'Cluster 1', 'Cluster 2']
        
        df_plot = df[df['Cluster'].isin(order)]

        for idx, feature in enumerate(selected_features):
            if idx < len(axes):
                ax = axes[idx]
                # Gunakan Seaborn boxplot
                sns.boxplot(
                    data=df_plot, 
                    x='Cluster', 
                    y=feature, 
                    ax=ax, 
                    palette=colors, 
                    order=order
                )
                ax.set_title(f'Distribusi {feature} per Cluster ')
                ax.set_ylabel(feature)
                ax.set_xlabel('Cluster')
                ax.grid(True, alpha=0.3)

        for idx in range(len(selected_features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Boxplot Distribusi Fitur Numerik per Cluster ', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Tidak dapat membuat boxplot: {e}")

def display_kabupaten_list_improved(df_filtered, cluster_type, max_display=10):
    """
    Menampilkan daftar kabupaten dengan format yang lebih rapi dan informatif.
    Data 'df_filtered' yang diterima adalah data KABUPATEN UNIK (representatif).
    """
    
    try:
        cluster_data = df_filtered[df_filtered["Cluster"] == cluster_type]
        
        if len(cluster_data) == 0:
            return f"Tidak ada kabupaten dalam kategori {cluster_type}"
        
        # Urutkan berdasarkan density (nilai tertinggi pertama)
        cluster_data_sorted = cluster_data.sort_values('Density', ascending=False)
        
        kabupaten_list = [kab.title() for kab in cluster_data_sorted["Kabupaten"].unique()]

        display_text = f"**Jumlah Kabupaten Unik:** {len(kabupaten_list)} kabupaten\n\n"

        # Tampilkan kabupaten dengan density tertinggi
        if len(kabupaten_list) > 0:
            display_text += "**Kabupaten Representatif (Density Tertinggi):**\n"
            top_kabupaten = cluster_data_sorted.head(min(3, len(cluster_data_sorted)))
            for idx, row in top_kabupaten.iterrows():
                kabupaten_name = row['Kabupaten'].title()
                density_value = row['Density']
                tahun_val = row.get('Tahun', 'N/A')
                display_text += f"• {kabupaten_name} (Tahun: {int(tahun_val)}) (Density: {density_value:.4f})\n"
        
        # Tampilkan daftar lengkap kabupaten
        if len(kabupaten_list) <= max_display:
            display_text += f"\n**Daftar Lengkap ({len(kabupaten_list)} kabupaten):**\n"
            for kab in kabupaten_list:
                display_text += f"• {kab}\n"
        else:
            display_text += f"\n**{max_display} Kabupaten Pertama:**\n"
            for kab in kabupaten_list[:max_display]:
                display_text += f"• {kab}\n"
            
            # Sembunyikan sisanya dalam expander
            with st.expander(f"📋 Lihat {len(kabupaten_list) - max_display} kabupaten lainnya"):
                for kab in kabupaten_list[max_display:]:
                    st.write(f"• {kab}")
        
        return display_text
    except Exception as e:
        return f"Error dalam menampilkan daftar kabupaten: {str(e)}"

def create_cluster_summary_table(df_filtered):
    """
    Membuat tabel ringkasan cluster yang lebih informatif.
    Data 'df_filtered' yang diterima adalah data KABUPATEN UNIK (representatif).
    """
    
    try:
        summary_data = []
        
        for cluster in ["Cluster 0", "Cluster 1", "Cluster 2"]: 
            cluster_data = df_filtered[df_filtered["Cluster"] == cluster]
            
            if len(cluster_data) > 0:
                avg_density = cluster_data['Density'].mean()
                min_density = cluster_data['Density'].min()
                max_density = cluster_data['Density'].max()
                top_row = cluster_data.nlargest(1, 'Density').iloc[0]
                top_kabupaten = f"{top_row['Kabupaten'].title()} ({int(top_row['Tahun'])})"
            else:
                avg_density = min_density = max_density = 0
                top_kabupaten = "-"
            
            summary_data.append({
                'Kategori': cluster,
                'Jumlah Kabupaten Unik': len(cluster_data),
                'Density Rata-rata': f"{avg_density:.4f}",
                'Density Min': f"{min_density:.4f}" if len(cluster_data) > 0 else "-",
                'Density Max': f"{max_density:.4f}" if len(cluster_data) > 0 else "-",
                'Kabupaten Density Tertinggi': top_kabupaten
            })
        
        return pd.DataFrame(summary_data)
    except Exception as e:
        st.error(f"Error membuat tabel ringkasan: {e}")
        return pd.DataFrame()

# FUNGSI UNTUK MENAMPILKAN PEMILIHAN KOLOM DI HALAMAN UTAMA
def display_column_selection(df):
    """Menampilkan pemilihan kolom numerik di halaman utama"""
    
    try:
        st.subheader("📊 Pilih fitur untuk Clustering")
        
        # Dapatkan semua kolom numerik, kecuali kolom tahun
        all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'Tahun' in all_numeric_cols:
            all_numeric_cols.remove('Tahun')
        # Hapus juga Lat/Lon jika ada
        if 'Latitude' in all_numeric_cols:
            all_numeric_cols.remove('Latitude')
        if 'Longitude' in all_numeric_cols:
            all_numeric_cols.remove('Longitude')
            
        if len(all_numeric_cols) > 0:
            # Pemilihan kolom dengan multiselect
            selected_numeric_cols = st.multiselect(
                "Pilih fitur numerik yang akan digunakan untuk clustering:",
                all_numeric_cols,
                default=all_numeric_cols,
                help="""
                Pilih fitur numerik yang relevan untuk analisis clustering. 
                Kolom 'Tahun' tidak termasuk karena merupakan variabel temporal.
                """
            )
            
            if len(selected_numeric_cols) < 2:
                st.warning("⚠️ Pilih minimal 2 kolom numerik untuk clustering")
                return None
            
            # Tampilkan kolom yang dipilih
            st.success(f"✅ {len(selected_numeric_cols)} kolom terpilih untuk clustering")
            
            return selected_numeric_cols
        
        else:
            st.error("❌ Tidak ada kolom numerik dalam dataset!")
            return None
    except Exception as e:
        st.error(f"❌ Error dalam pemilihan kolom: {str(e)}")
        return None

def display_map_filters(df):
    """Menampilkan filter peta dan tahun di halaman utama"""
    
    try:
        st.subheader("🎯 Filter Data")
        
        filter_container = st.container()
        
        with filter_container:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                all_provinsi = ["Semua"] + sorted(df['Provinsi'].dropna().unique().tolist())
                selected_provinsi = st.selectbox(
                    "Pilih Provinsi:",
                    all_provinsi,
                    help="Filter data berdasarkan provinsi yang dipilih. 'Semua' akan menampilkan data dari seluruh provinsi.",
                    key="provinsi_filter_main"
                )
            
            with col2:
                if selected_provinsi == "Semua":
                    available_kabupaten = ["Semua"] + sorted(df['Kabupaten'].dropna().unique().tolist())
                else:
                    provinsi_kabupaten = df[df['Provinsi'] == selected_provinsi]['Kabupaten'].dropna().unique().tolist()
                    available_kabupaten = ["Semua"] + sorted(provinsi_kabupaten)
                
                selected_kabupaten = st.selectbox(
                    "Pilih Kabupaten:",
                    available_kabupaten,
                    help="Filter data berdasarkan kabupaten yang dipilih. 'Semua' akan menampilkan data dari provinsi yang dipilih.",
                    key="kabupaten_filter_main"
                )
            
            with col3:
                # Filter berdasarkan tahun
                available_years = sorted(df['Tahun'].dropna().unique().tolist())
                if available_years:
                    min_year = int(min(available_years))
                    max_year = int(max(available_years))
                    
                    year_range = st.slider(
                        "Pilih Rentang Tahun:",
                        min_value=min_year,
                        max_value=max_year,
                        value=(min_year, max_year),
                        help="Pilih rentang tahun untuk analisis. Hanya data dalam rentang ini yang akan diproses."
                    )
                    selected_years = list(range(year_range[0], year_range[1] + 1))
                    
                else:
                    st.warning("Tidak ada data tahun tersedia")
                    selected_years = []
            
            with col4:
                st.write("")
                st.write("")
                if st.button("🔄 Reset Filter", key="reset_filter_main", help="Klik untuk mengembalikan semua filter (Provinsi, Kabupaten, Tahun) ke pengaturan awal."):
                    st.session_state.provinsi_filter_main = "Semua"
                    st.session_state.kabupaten_filter_main = "Semua"
                    st.rerun()
        
        return selected_provinsi, selected_kabupaten, selected_years
    except Exception as e:
        st.error(f"❌ Error dalam menampilkan filter: {str(e)}")
        return "Semua", "Semua", []

def filter_data_by_year(df, selected_years):
    """Memfilter data berdasarkan tahun yang dipilih"""
    try:
        if selected_years:
            df_filtered = df[df['Tahun'].isin(selected_years)].copy()
            
            if len(selected_years) == 1:
                year_display = str(selected_years[0])
            elif len(selected_years) > 1:
                year_display = f"{min(selected_years)}-{max(selected_years)}"
            else:
                year_display = "Tidak ada" 

            st.info(f"📅 Memfilter data untuk tahun: {year_display} - {len(df_filtered)} total kejadian ditemukan.")
            return df_filtered
        else:
            return df.copy()
    except Exception as e:
        st.warning(f"Error dalam memfilter data berdasarkan tahun: {e}")
        return df.copy()

def display_filtered_analysis(df_filtered, df_original, numeric_cols, selected_provinsi, selected_kabupaten, selected_years):
    """
    Menampilkan analisis berdasarkan data yang difilter.
    'df_filtered' = data kabupaten unik yang sudah difilter lokasi.
    'df_original' = semua kejadian yang sudah difilter tahun.
    """
    
    try:
        filter_info = []
        if selected_provinsi != "Semua":
            filter_info.append(f"Provinsi: {selected_provinsi}")
        if selected_kabupaten != "Semua":
            filter_info.append(f"Kabupaten: {selected_kabupaten}")
        
        year_display_str = ""
        if selected_years:
            if len(selected_years) == 1:
                year_display = str(selected_years[0])
            else:
                year_display = f"{min(selected_years)}-{max(selected_years)}"
            
            year_display_str = f"Rentang Tahun: **{year_display}**"
            
            all_years_in_original = df_original['Tahun'].dropna().unique()
            if len(all_years_in_original) > 0 and (len(selected_years) < len(all_years_in_original) or selected_years != sorted(all_years_in_original.tolist())):
                filter_info.append(f"Tahun: {year_display}")
        else:
            year_display_str = "Rentang Tahun: **Semua Tahun**"
        
        if filter_info:
            st.info(f"**Filter Aktif:** {', '.join(filter_info)} - Menampilkan {len(df_filtered)} kabupaten unik (dari {len(df_original)} total kejadian di rentang tahun ini)")
        
        if len(df_filtered) > 0:
            cluster_causes = analyze_cluster_causes(df_filtered, numeric_cols)
            
            st.subheader("📋 Detail Kabupaten per Kategori (Berdasarkan Filter)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🟢 **Cluster 0 (Ringan)**") 
                ringan_text = display_kabupaten_list_improved(df_filtered, "Cluster 0") 
                lines = ringan_text.split('\n')
                for line in lines:
                    if line.startswith('**'):
                        st.write(line)
                    elif line.startswith('•'):
                        st.write(line)
                    else:
                        st.text(line)
                
                if "Cluster 0" in cluster_causes: 
                    st.info(f"**Karakteristik:** {cluster_causes['Cluster 0']}")
            
            with col2:
                st.markdown("### 🟠 **Cluster 1 (Sedang)**")
                sedang_text = display_kabupaten_list_improved(df_filtered, "Cluster 1")
                lines = sedang_text.split('\n')
                for line in lines:
                    if line.startswith('**'):
                        st.write(line)
                    elif line.startswith('•'):
                        st.write(line)
                    else:
                        st.text(line)
                
                if "Cluster 1" in cluster_causes:
                    st.info(f"**Karakteristik:** {cluster_causes['Cluster 1']}")
            
            with col3:
                st.markdown("### 🔴 **Cluster 2 (Berat)**") 
                berat_text = display_kabupaten_list_improved(df_filtered, "Cluster 2") 
                lines = berat_text.split('\n')
                for line in lines:
                    if line.startswith('**'):
                        st.write(line)
                    elif line.startswith('•'):
                        st.write(line)
                    else:
                        st.text(line)
                
                if "Cluster 2" in cluster_causes: 
                    st.warning(f"**Karakteristik:** {cluster_causes['Cluster 2']}")
            
            st.subheader("📊 Statistik Clustering")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Total Kabupaten Unik (Sesuai Filter)", len(df_filtered))
            with col_stat2:
                st.metric("Cluster 0", len(df_filtered[df_filtered["Cluster"] == "Cluster 0"])) 
            with col_stat3:
                st.metric("Cluster 1", len(df_filtered[df_filtered["Cluster"] == "Cluster 1"]))
            with col_stat4:
                st.metric("Cluster 2", len(df_filtered[df_filtered["Cluster"] == "Cluster 2"])) 
            
            st.markdown("")
            st.subheader("📈 VISUALISASI ANALISIS CLUSTERING")
            
            plot_tab1, plot_tab2, plot_tab3 = st.tabs([
                "📊 Silhouette Analysis", 
                "🔵 Scatter Matrix", 
                "📦 Boxplot Distribusi"
            ])

            with plot_tab1:
                st.write("### Silhouette Analysis")
                st.info(f"Silhouette analysis mengevaluasi kualitas clustering (berdasarkan fitur yang digunakan untuk clustering). {year_display_str}. Score mendekati 1 menunjukkan clustering yang baik.")
                create_silhouette_plot(df_filtered, numeric_cols) 
            
            plot_cols = numeric_cols.copy()
            
            with plot_tab2:
                st.write("### Scatter Matrix")
                st.info(f"Scatter matrix menunjukkan hubungan antara berbagai fitur (yang digunakan untuk clustering) dan bagaimana clustering memisahkan data. {year_display_str}.")
                create_scatter_plots(df_filtered, plot_cols) 
            
            with plot_tab3:
                st.write("### Boxplot Distribusi per Cluster")
                st.info(f"Boxplot menunjukkan distribusi (rentang kuartil) setiap fitur numerik (yang digunakan untuk clustering) dalam setiap cluster. {year_display_str}.")
                create_boxplots(df_filtered, plot_cols)
            
            return True
        else:
            st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.")
            return False
            
    except Exception as e:
        st.error(f"❌ Error dalam menampilkan analisis terfilter: {str(e)}")
        return False

if uploaded_file:
    if st.session_state.get('current_file_name') != uploaded_file.name:
        st.info(f"Memproses file baru: {uploaded_file.name}")
        try:
                keys_to_clear = [
                    'master_df', 'df_clustered', 'processed_cols', 
                    'current_file_name_clustered', 'current_file_name'
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.session_state['current_file_name'] = uploaded_file.name

                # Validasi ekstensi file
                if not validate_file_extension(uploaded_file.name):
                    st.error("❌ Format file tidak didukung. Silakan upload file CSV.")
                    st.stop()
                
                st.success("✅ File berhasil diunggah! (Format .csv terdeteksi)")
                
                # Load data
                try:
                    df_init = pd.read_csv(uploaded_file)
                
                except Exception as e:
                    st.error(f"❌ Error membaca file CSV: {str(e)}. Pastikan file tidak rusak atau formatnya benar.")
                    st.stop()

                # Validasi struktur
                is_valid_structure, structure_message = validate_csv_structure(df_init)
                if not is_valid_structure:
                    st.error(f"❌ {structure_message}")
                    st.stop()
                
                # Validasi kualitas
                quality_issues = validate_data_quality(df_init)
                if quality_issues and any("Data terlalu sedikit" in issue for issue in quality_issues):
                    st.error("❌ Tidak dapat melanjutkan karena data tidak memenuhi syarat minimum.")
                    st.stop()

                # Normalisasi dan Ekstraksi Tahun
                df_init = normalize_kabupaten_names(df_init)
                df_init = extract_year_from_date(df_init, 'Tanggal Kejadian')

                # Geocoding
                with st.spinner("🚀 Melakukan geocoding (ini mungkin perlu waktu)..."):
                    if ("Latitude" not in df_init.columns or 
                        "Longitude" not in df_init.columns or 
                        df_init.get("Latitude", pd.Series([np.nan])).isna().any()):
                            
                        df_init["Latitude"], df_init["Longitude"] = geocode_kabupaten_fast(df_init)
                        df_init = apply_fallback_coordinates(df_init)
                        st.success("✅ Geocoding selesai.")
                    else:
                        st.info("✅ Data geocoding sudah ada di dalam file.")
                
                st.session_state['master_df'] = df_init
                st.success("✅ File baru berhasil diproses dan disimpan di session.")
                st.rerun() 
            
        except Exception as e:
            st.error(f"❌ Error tidak terduga saat memproses file: {str(e)}")
            if st.button("🔄 Reset Aplikasi"):
                st.session_state.clear()
                st.rerun()
            st.stop()

if 'master_df' in st.session_state:
    try:
        df = st.session_state['master_df']
        
        st.write("### 📋 Data Awal")
        st.dataframe(df.head())
        
        tab1, tab2 = st.tabs(["🗺️ Clustering DENCLUE", "🌊 Trend Banjir"])
        
        with tab1:
            try:
                is_valid_numeric, numeric_cols, numeric_message = validate_numeric_columns(df)
                if not is_valid_numeric:
                    st.error(f"❌ {numeric_message}")
                    st.stop()
                
                selected_numeric_cols = display_column_selection(df)
                
                if selected_numeric_cols is None:
                    st.stop()
                
                if ('df_clustered' not in st.session_state or
                    st.session_state.get('processed_cols') != selected_numeric_cols or
                    st.session_state.get('current_file_name_clustered') != st.session_state.get('current_file_name')):
                    
                    X, final_numeric_cols = advanced_preprocessing(
                        df, 
                        selected_cols=selected_numeric_cols
                    )
                    
                    clusters, density = denclue_clustering_cached(
                        X, SIGMA, THRESHOLD, len(df) 
                    )
                    
                    # Buat dataframe hasil cluster
                    df_clustered = df.copy()
                    df_clustered.loc[:, "Cluster"] = clusters
                    df_clustered.loc[:, "Density"] = density
                    
                    # Simpan di session state
                    st.session_state['df_clustered'] = df_clustered
                    st.session_state['processed_cols'] = final_numeric_cols
                    st.session_state['current_file_name_clustered'] = st.session_state.get('current_file_name')
                    st.success("✅ Clustering pada seluruh dataset selesai!")
                
                df_clustered = st.session_state['df_clustered']
                final_numeric_cols = st.session_state['processed_cols']
                
                selected_provinsi, selected_kabupaten, selected_years = display_map_filters(df_clustered)
                
                df_filtered_by_year_ALL_INCIDENTS = filter_data_by_year(df_clustered, selected_years)
                
                if len(df_filtered_by_year_ALL_INCIDENTS) == 0:
                    st.warning("⚠️ Tidak ada data untuk rentang tahun yang dipilih.")
                    st.stop()

                st.markdown("---")
                st.info(
                    "Analisis berikut adalah **ringkasan (agregat)** dari seluruh rentang tahun yang dipilih. "
                    "Peta dan grafik di bawah ini menunjukkan **status tertinggi** (berdasarkan density) yang pernah dicapai "
                    "setiap kabupaten *kapan saja* dalam rentang tahun tersebut."
                )

                with st.spinner("Membuat data Peta Agregat (1 titik per Kabupaten)..."):
                    df_MAP_DATA_UNIQUE_KAB = df_filtered_by_year_ALL_INCIDENTS.sort_values('Density', ascending=False).drop_duplicates('Kabupaten', keep='first')

                st.subheader("🗺️ Visualisasi Peta Hasil Clustering")
                
                try:
                    folium_map, df_ANALYSIS_DATA = create_filtered_map(
                        df_MAP_DATA_UNIQUE_KAB, 
                        selected_provinsi, 
                        selected_kabupaten
                    )
                    st_folium(folium_map, width=900, height=600, key="agregat_map")
                except Exception as e:
                    st.error(f"❌ Error dalam visualisasi peta: {str(e)}")
                
                # Tampilkan analisis berdasarkan filter (AGREGAT)
                try:
                  
                    display_success = display_filtered_analysis(
                        df_ANALYSIS_DATA,           
                        df_filtered_by_year_ALL_INCIDENTS,  
                        final_numeric_cols,          
                        selected_provinsi,       
                        selected_kabupaten,      
                        selected_years           
                    )
                except Exception as e:
                    st.error(f"❌ Error dalam menampilkan analisis agregat: {str(e)}")
                    display_success = False

                
                st.markdown("---")
                st.header(f"🔍 Analisis Detail Per Tahun (Rentang {min(selected_years)}-{max(selected_years)})")
                st.info(
                    "Berikut adalah rincian untuk **setiap tahun secara terpisah** dalam rentang yang Anda pilih. "
                    "Buka setiap 'expander' di bawah untuk melihat analisis (termasuk peta dan plot) khusus untuk tahun tersebut. "
                    "Filter Provinsi dan Kabupaten di atas juga akan diterapkan di sini."
                )

                if len(selected_years) > 10:
                    st.warning(f"⚠️ Anda memilih {len(selected_years)} tahun. Menampilkan analisis terpisah untuk setiap tahun dapat membuat halaman menjadi lambat. Buka expander satu per satu.")

                active_provinsi = st.session_state.provinsi_filter_main
                active_kabupaten = st.session_state.kabupaten_filter_main
                
                # Loop untuk setiap tahun yang dipilih
                for year in sorted(selected_years):
                    
                    # Buat expander untuk setiap tahun
                    with st.expander(f"## 📅 Analisis Tahun {year}"):
                        try:
                            # 1. Filter data MASTER (df_clustered) untuk TAHUN INI SAJA
                            df_year_all_incidents = df_clustered[df_clustered['Tahun'] == year]

                            if df_year_all_incidents.empty:
                                st.warning(f"Tidak ada data untuk tahun {year}.")
                                continue
                 
                            df_year_unique_kab = df_year_all_incidents.sort_values('Density', ascending=False).drop_duplicates('Kabupaten', keep='first')

                            if df_year_unique_kab.empty:
                                st.warning(f"Tidak ada data kabupaten unik untuk tahun {year}.")
                                continue

                            df_year_filtered_kab = apply_filters(df_year_unique_kab, active_provinsi, active_kabupaten)

                            if df_year_filtered_kab.empty:
                                st.warning(f"Tidak ada data untuk tahun {year} dengan filter lokasi '{active_provinsi}' / '{active_kabupaten}'.")

                            st.write(f"#### 🗺️ Peta Kategori Tahun {year}")

                            folium_map_year, df_analysis_data_year = create_filtered_map(
                                df_year_unique_kab, 
                                active_provinsi,
                                active_kabupaten
                            )
                            st_folium(folium_map_year, width=800, height=450, key=f"map_{year}")

                            st.write(f"#### 📊 Analisis Kategori Tahun {year}")
                            
                            display_filtered_analysis(
                                df_analysis_data_year,      
                                df_year_all_incidents,      
                                final_numeric_cols,        
                                active_provinsi,            
                                active_kabupaten,           
                                [year]                   
                            )

                        except Exception as e:
                            st.error(f"❌ Gagal memuat analisis untuk tahun {year}: {e}")

            except Exception as e:
                st.error(f"❌ Error tidak terduga dalam tab Clustering: {str(e)}")
        
        with tab2:
            try:
                if 'selected_years' in locals() and selected_years:
                    
                    # Buat info rentang tahun
                    if len(selected_years) == 1:
                        year_display = str(selected_years[0])
                    elif len(selected_years) > 1:
                        year_display = f"{min(selected_years)}-{max(selected_years)}"
                    else:
                        year_display = "Tidak ada"
                    
                    st.info(f"📅 Menampilkan analisis trend untuk rentang tahun yang dipilih di Tab 1: **{year_display}**")
                    
                    df_for_trends = df[df['Tahun'].isin(selected_years)].copy()
                    
                    if len(df_for_trends) == 0:
                        st.warning("⚠️ Tidak ada data untuk rentang tahun yang dipilih.")
                    else:
                        banjir_df = analyze_banjir_trends(df_for_trends)
                
                else:
                    st.warning("⚠️ Rentang tahun tidak terdefinisi. Menampilkan analisis untuk semua data.")
                    banjir_df = analyze_banjir_trends(df) 

            except Exception as e:
                st.error(f"❌ Error dalam analisis trend banjir: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error tidak terduga dalam proses utama: {str(e)}")
        st.info("💡 Silakan periksa format file dan pastikan data sesuai dengan struktur yang diharapkan.")
        if st.button("🔄 Reset Aplikasi"):
            st.session_state.clear()
            st.rerun()

# UNTUK HALAMAN KOSONG
else:
    st.warning("📥 Silakan unggah file data (hanya .csv yang akan diterima)")