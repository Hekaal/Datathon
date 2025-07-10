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

@st.cache_data
def load_raw_data():
    hist_df = None
    forecast_df = None
    try:
        hist_df = pd.read_csv('master_table_modified_fix.csv')
        forecast_df = pd.read_csv('hasil_prediksi_12_bulan.csv')
        return hist_df, forecast_df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None, None

@st.cache_resource
def load_model():
    try:
        return joblib.load('model.joblib')
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

@st.cache_resource
def load_unsupervised_models():
    try:
        kmeans = joblib.load("kmeans_model.joblib")
        isoforest = joblib.load("isolation_forest_model.joblib")
        scaler = joblib.load("unsupervised_scaler.joblib")
        return kmeans, isoforest, scaler
    except Exception as e:
        st.error(f"Gagal memuat model unsupervised: {e}")
        return None, None, None

# --- FUNGSI TAMBAHAN: Terapkan model unsupervised ke hasil prediksi ---
def apply_unsupervised_to_forecast(df_forecast, kmeans_model, iso_model, scaler):
    df = df_forecast.copy()
    fitur_unsup = ['harga_prediksi', 'stok_beras_ton', 'jumlah_bencana', 'jumlah_curah_hujan']
    for col in fitur_unsup:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=fitur_unsup, inplace=True)
    if df.empty:
        return df_forecast
    fitur_scaled = scaler.transform(df[fitur_unsup])
    df['cluster_kmeans'] = kmeans_model.predict(fitur_scaled)
    df['anomaly_isolation_forest'] = iso_model.predict(fitur_scaled)
    return df

# --- MEMUAT SEMUA ASET ---
df_hist_raw, df_forecast_raw = load_raw_data()
model = load_model()
kmeans_model, iso_model, scaler = load_unsupervised_models()
geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province.json"

if None in [df_hist_raw, df_forecast_raw, model, kmeans_model, iso_model, scaler]:
    st.stop()

# Terapkan model unsupervised ke data prediksi
if 'df_forecast' not in st.session_state:
    st.session_state['df_forecast'] = apply_unsupervised_to_forecast(df_forecast_raw, kmeans_model, iso_model, scaler)

df_forecast = st.session_state['df_forecast']

# --- DASHBOARD UTAMA ---
st.title("📊 SmartDemand Dashboard")
st.markdown("Analisis Prediktif Harga dan Risiko Beras Nasional")

# --- TAMPILKAN METRIK ---
st.subheader("🔍 Ringkasan Prediksi dan Risiko")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Prediksi Harga per Provinsi")
    st.dataframe(df_forecast[['provinsi', 'tanggal', 'harga_prediksi']].sort_values(by='tanggal'))

with col2:
    st.markdown("#### Cluster Risiko")
    st.dataframe(df_forecast[['provinsi', 'tanggal', 'cluster_kmeans', 'anomaly_isolation_forest']])

# --- TABS: Peta Visualisasi ---
if not df_forecast.empty and 'provinsi' in df_forecast.columns:
    tab1, tab2 = st.tabs(["🗺️ Peta Risiko (Cluster)", "🚨 Deteksi Anomali"])

    with tab1:
        st.subheader("Segmentasi Risiko (KMeans Clustering)")
        fig_cluster = px.choropleth(
            df_forecast,
            geojson=geojson_url,
            locations='provinsi',
            featureidkey="properties.Propinsi",
            color='cluster_kmeans',
            color_continuous_scale=["#2ca02c", "#ff7f0e", "#d62728"],
            scope="asia",
            labels={'cluster_kmeans': 'Cluster Risiko'}
        )
        fig_cluster.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_cluster, use_container_width=True)

    with tab2:
        st.subheader("Deteksi Anomali (Isolation Forest)")
        df_forecast['anomali_flag'] = df_forecast['anomaly_isolation_forest'].map({-1: 'Anomali', 1: 'Normal'})
        fig_anomali = px.choropleth(
            df_forecast,
            geojson=geojson_url,
            locations='provinsi',
            featureidkey="properties.Propinsi",
            color='anomali_flag',
            color_discrete_map={'Normal': 'green', 'Anomali': 'red'},
            scope="asia",
            labels={'anomali_flag': 'Status'}
        )
        fig_anomali.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_anomali, use_container_width=True)
else:
    st.warning("Data prediksi tidak tersedia atau tidak lengkap.")

# --- FITUR TAMBAHAN: Download Hasil ---
st.subheader("⬇️ Unduh Data")
with st.expander("Klik untuk mengunduh hasil prediksi dan analisis"):
    col_download1, col_download2 = st.columns(2)

    with col_download1:
        csv_prediksi = df_forecast.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Unduh Prediksi & Cluster (CSV)", data=csv_prediksi, file_name="hasil_prediksi_dengan_analisis.csv", mime='text/csv')

    with col_download2:
        json_data = df_forecast.to_json(orient='records')
        st.download_button("🧾 Unduh Prediksi (JSON)", data=json_data, file_name="hasil_prediksi.json", mime='application/json')
