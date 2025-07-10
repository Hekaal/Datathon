# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import json

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard SmartDemand-ID",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI PEMUATAN ASET (dengan cache) ---

@st.cache_data
def load_raw_data():
    """Memuat data mentah historis dan prediksi."""
    try:
        hist_df = pd.read_csv('master_table_modified_fix.csv')
        forecast_df = pd.read_csv('hasil_prediksi_12_bulan.csv')
        
        # Pra-pemrosesan 'tanggal'
        if 'tanggal' in hist_df.columns:
            hist_df['tanggal'] = pd.to_datetime(hist_df['tanggal'], errors='coerce')
        if 'tanggal' in forecast_df.columns:
            forecast_df['tanggal'] = pd.to_datetime(forecast_df['tanggal'], errors='coerce')

        return hist_df, forecast_df
    except FileNotFoundError as e:
        st.error(f"File data tidak ditemukan: {e}. Pastikan file CSV berada di direktori yang benar.")
        return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return None, None

@st.cache_resource
def load_models():
    """Memuat semua model machine learning."""
    try:
        model = joblib.load('model.joblib')
        kmeans = joblib.load("kmeans_model.joblib")
        isoforest = joblib.load("isolation_forest_model.joblib")
        scaler = joblib.load("unsupervised_scaler.joblib")
        return model, kmeans, isoforest, scaler
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}. Pastikan semua file .joblib ada.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None, None, None

# --- MEMUAT SEMUA ASET ---
df_hist, df_forecast_raw = load_raw_data()
model, kmeans_model, iso_model, scaler = load_models()
geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province.json"

if any(v is None for v in [df_hist, df_forecast_raw, model, kmeans_model, iso_model, scaler]):
    st.stop()

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Navigasi Dashboard")
    list_provinsi = sorted(df_hist['provinsi'].unique())
    selected_province = st.selectbox(
        'Pilih Provinsi:',
        options=list_provinsi
    )

    combined_dates = pd.concat([
        df_hist[df_hist['provinsi'] == selected_province]['tanggal'],
        df_forecast_raw[df_forecast_raw['provinsi'] == selected_province]['tanggal']
    ]).dropna().drop_duplicates().sort_values()

    selected_historical_date = None
    if not combined_dates.empty:
        selected_historical_date = st.selectbox(
            'Pilih Tanggal Acuan:',
            options=combined_dates,
            index=len(combined_dates) - 1,
            format_func=lambda x: x.strftime('%B %Y')
        )
    else:
        st.warning(f"Tidak ada data untuk {selected_province}.")

    st.info("Dashboard ini menganalisis dan memprediksi harga beras untuk mendukung stabilitas pangan nasional.")

# --- JUDUL UTAMA ---
st.title("🍚 Dashboard Prediksi Harga Beras")
st.markdown("Analisis Prediktif dan Deteksi Risiko untuk Mendukung Stabilitas Pangan Nasional")

# --- PERSIAPAN DATA BERDASARKAN INPUT PENGGUNA ---
selected_date_data = None
if selected_historical_date:
    data_df = df_hist[
        (df_hist['provinsi'] == selected_province) & 
        (df_hist['tanggal'] == selected_historical_date)
    ]
    if not data_df.empty:
        selected_date_data = data_df.iloc[0]

target_prediction_date = None
if selected_historical_date:
    target_prediction_date = selected_historical_date + pd.DateOffset(months=1)

# =================================================================================
#                         BAGIAN UTAMA APLIKASI DENGAN TABS
# =================================================================================

tab1, tab2 = st.tabs(["📊 Analisis Prediktif Utama", "🔬 Analisis Unsupervised (Risiko)"])

with tab1:
    st.header(f"Analisis untuk Provinsi: {selected_province}")

    # --- KEY PERFORMANCE INDICATORS (KPIs) ---
    kpi1, kpi2, kpi3 = st.columns(3)
    
    # KPI 1: Harga pada tanggal terpilih
    harga_value = pd.to_numeric(selected_date_data['harga_beras'], errors='coerce') if selected_date_data is not None else None
    kpi1.metric(
        label=f"Harga pada {selected_historical_date.strftime('%B %Y')}" if selected_historical_date else "Harga",
        value=f"Rp {harga_value:,.0f}" if pd.notnull(harga_value) else "N/A"
    )
    
    # KPI 2: Harga prediksi bulan depan
    pred_price_numeric = None
    if target_prediction_date:
        pred_data_df = df_forecast_raw[
            (df_forecast_raw['provinsi'] == selected_province) & 
            (df_forecast_raw['tanggal'] == target_prediction_date)
        ]
        if not pred_data_df.empty:
            pred_price_numeric = pd.to_numeric(pred_data_df.iloc[0]['harga_prediksi'], errors='coerce')

    delta_value = pred_price_numeric - harga_value if pd.notnull(pred_price_numeric) and pd.notnull(harga_value) else None
    kpi2.metric(
        label=f"Prediksi pada {target_prediction_date.strftime('%B %Y')}" if target_prediction_date else "Prediksi Bulan Depan",
        value=f"Rp {pred_price_numeric:,.0f}" if pd.notnull(pred_price_numeric) else "N/A",
        delta=f"Rp {delta_value:,.0f}" if pd.notnull(delta_value) else None
    )

    # KPI 3: Akurasi Model
    kpi3.metric(label="Akurasi Model (MAE)", value="~ Rp 161", help="Rata-rata kesalahan prediksi model pada data historis.")

    # --- GRAFIK TIME SERIES ---
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Grafik Tren Harga Historis & Prediksi")
        hist_province = df_hist[df_hist['provinsi'] == selected_province]
        forecast_province = df_forecast_raw[df_forecast_raw['provinsi'] == selected_province]
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=hist_province['tanggal'], y=hist_province['harga_beras'], mode='lines', name='Harga Historis', line=dict(color='royalblue')))
        fig_ts.add_trace(go.Scatter(x=forecast_province['tanggal'], y=forecast_province['harga_prediksi'], mode='lines+markers', name='Harga Prediksi', line=dict(dash='dash', color='red')))
        fig_ts.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_ts, use_container_width=True)

    with col2:
        st.subheader("Analisis Pendorong Harga (Rule-Based)")
        
        # Momentum Tren Harga
        with st.container(border=True):
            st.markdown("##### Momentum Tren Harga")
            if selected_date_data is not None and 'harga_beras_bulan_lalu' in selected_date_data:
                last_month_price = pd.to_numeric(selected_date_data['harga_beras_bulan_lalu'], errors='coerce')
                if pd.notnull(harga_value) and pd.notnull(last_month_price):
                    delta = harga_value - last_month_price
                    if delta > 100:
                        st.markdown("🔴 **Cenderung Naik**")
                    elif delta < -100:
                        st.markdown("🟢 **Cenderung Turun**")
                    else:
                        st.markdown("🟡 **Stabil**")
                else:
                    st.markdown("⚪ Data tidak lengkap")
            else:
                st.markdown("⚪ Data tidak tersedia")

        # Status Stok Beras
        with st.container(border=True):
            st.markdown("##### Status Stok Beras")
            if selected_date_data is not None and 'stok_beras_ton' in selected_date_data:
                stock_value = pd.to_numeric(selected_date_data['stok_beras_ton'], errors='coerce')
                if pd.notnull(stock_value):
                    if stock_value < 100000:
                        st.markdown("🔴 **Rendah** (Potensi tekanan harga naik)")
                    else:
                        st.markdown("🟢 **Aman** (Stok mencukupi)")
                else:
                    st.markdown("⚪ Data tidak lengkap")
            else:
                st.markdown("⚪ Data tidak tersedia")

with tab2:
    st.header("Analisis Risiko Menggunakan Unsupervised Learning")
    st.markdown("Menganalisis data berdasarkan pola yang ditemukan secara otomatis oleh model K-Means (segmentasi) dan Isolation Forest (anomali).")

    if selected_date_data is not None:
        features_for_unsup = ['harga_beras', 'stok_beras_ton', 'jumlah_bencana', 'jumlah_curah_hujan']
        
        # Validasi ketersediaan kolom
        if all(col in selected_date_data and pd.notnull(selected_date_data[col]) for col in features_for_unsup):
            input_data = pd.DataFrame([selected_date_data[features_for_unsup]])
            input_data_scaled = scaler.transform(input_data)
            
            # Prediksi dari model unsupervised
            cluster_label = kmeans_model.predict(input_data_scaled)[0]
            anomaly_label = iso_model.predict(input_data_scaled)[0]
            
            unsup_col1, unsup_col2 = st.columns(2)
            with unsup_col1:
                st.subheader("Segmentasi Risiko (K-Means)")
                st.metric(label="Data Masuk ke Cluster", value=f"Cluster {cluster_label}")
                if cluster_label == 0:
                    st.success("🟢 **Kondisi Aman**: Harga stabil dan pasokan mencukupi.")
                elif cluster_label == 1:
                    st.warning("🟠 **Waspada**: Terdapat tekanan pada harga atau stok.")
                elif cluster_label == 2:
                    st.error("🔴 **Kritis**: Kombinasi stok rendah, bencana, dan harga tinggi.")

            with unsup_col2:
                st.subheader("Deteksi Anomali (Isolation Forest)")
                if anomaly_label == -1:
                    st.error("🚨 **Anomali Terdeteksi!**")
                    st.markdown("Kondisi data untuk tanggal dan provinsi ini menyimpang secara signifikan dari pola normal yang telah dipelajari.")
                else:
                    st.success("✅ **Normal**")
                    st.markdown("Kondisi data sesuai dengan pola yang umum terjadi.")
        else:
            st.warning("Data tidak lengkap untuk melakukan analisis unsupervised. Pastikan semua fitur (harga, stok, bencana, curah hujan) tersedia.")
    else:
        st.warning("Pilih tanggal historis yang valid untuk melihat analisis unsupervised.")
