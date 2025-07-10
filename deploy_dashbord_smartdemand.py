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

# --- FUNGSI UNTUK MEMUAT DATA & ASET ---
@st.cache_data
def load_raw_data():
    """Memuat data mentah historis dan prediksi."""
    hist_df = None
    forecast_df = None
    try:
        # Menggunakan file yang sesuai dengan analisis unsupervised
        hist_df = pd.read_csv('master_table_modified_unsupervised.csv')
        print("master_table_modified_unsupervised.csv loaded successfully.")

        forecast_df = pd.read_csv('hasil_prediksi_dengan_unsupervised.csv')
        print("hasil_prediksi_dengan_unsupervised.csv loaded successfully.")
        
        return hist_df, forecast_df
    except FileNotFoundError as e:
        st.error(f"File data tidak ditemukan: {e}. Pastikan 'master_table_modified_unsupervised.csv' dan 'hasil_prediksi_dengan_unsupervised.csv' berada di direktori yang benar.")
        return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data mentah: {e}")
        return None, None

@st.cache_resource
def load_model():
    """Memuat model machine learning yang sudah dilatih."""
    try:
        model = joblib.load('model.joblib')
        print("Model 'model.joblib' loaded successfully.")
        return model
    except FileNotFoundError:
        st.error("File model 'model.joblib' not found. Please run the notebook to create it.")
        return None
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        return None

# --- MEMUAT DATA & MODEL ---
df_hist_raw, df_forecast_raw = load_raw_data()
model = load_model()
geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province.json"

if df_hist_raw is None or df_forecast_raw is None or model is None:
    st.stop()

@st.cache_resource
def load_unsupervised_models():
    """Memuat model-model unsupervised."""
    try:
        kmeans = joblib.load("kmeans_model.joblib")
        isoforest = joblib.load("isolation_forest_model.joblib")
        scaler = joblib.load("unsupervised_scaler.joblib")
        return kmeans, isoforest, scaler
    except Exception as e:
        st.error(f"Gagal memuat model unsupervised: {e}")
        return None, None, None

kmeans_model, iso_model, scaler = load_unsupervised_models()

# --- PRA-PEMROSESAN DATA ---
df_hist = df_hist_raw.copy()
if 'tanggal' in df_hist.columns:
    df_hist['tanggal'] = pd.to_datetime(df_hist['tanggal'], errors='coerce')
else:
    st.error("Kolom 'tanggal' tidak ditemukan di master_table_modified_unsupervised.csv.")
    st.stop()

for col in ['harga_beras', 'harga_beras_bulan_lalu', 'stok_beras_ton']:
    if col in df_hist.columns:
        if df_hist[col].dtype == 'object':
            df_hist[col] = df_hist[col].astype(str).str.replace(',', '', regex=False)
        df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')

df_forecast = df_forecast_raw.copy()
if 'tanggal' in df_forecast.columns:
    df_forecast['tanggal'] = pd.to_datetime(df_forecast['tanggal'], errors='coerce')
else:
    st.error("Kolom 'tanggal' tidak ditemukan di hasil_prediksi_dengan_unsupervised.csv.")
    st.stop()

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("Navigasi Dashboard")
    list_provinsi = sorted(df_hist['provinsi'].unique())
    selected_province = st.selectbox('Pilih Provinsi:', options=list_provinsi)

    hist_dates = df_hist[df_hist['provinsi'] == selected_province]['tanggal'].dropna()
    pred_dates = df_forecast[df_forecast['provinsi'] == selected_province]['tanggal'].dropna()
    combined_dates = pd.concat([hist_dates, pred_dates]).drop_duplicates().sort_values()

    selected_historical_date = None
    if not combined_dates.empty:
        default_index = len(combined_dates) - 1 if len(combined_dates) > 0 else 0
        selected_historical_date = st.selectbox(
            'Pilih Tanggal Analisis:',
            options=combined_dates,
            index=default_index,
            format_func=lambda x: x.strftime('%B %Y')
        )
    else:
        st.warning(f"Tidak ada data untuk {selected_province}.")

    st.info("Dashboard untuk analisis dan prediksi harga beras demi mendukung stabilitas pangan nasional.")

# --- JUDUL UTAMA ---
st.title("🍚 Dashboard SmartDemand-ID")
st.markdown("Analisis Prediktif Harga Beras untuk Mendukung Stabilitas Pangan Nasional")

# --- PERSIAPAN DATA UNTUK KPI & RINGKASAN ---
selected_date_data = None
if selected_historical_date:
    selected_date_data_df = df_hist[(df_hist['provinsi'] == selected_province) & (df_hist['tanggal'] == selected_historical_date)]
    if selected_date_data_df.empty:
        selected_date_data_df = df_forecast[(df_forecast['provinsi'] == selected_province) & (df_forecast['tanggal'] == selected_historical_date)]
    if not selected_date_data_df.empty:
        selected_date_data = selected_date_data_df.iloc[0]

target_prediction_date = None
if selected_historical_date:
    target_prediction_date = (selected_historical_date + pd.DateOffset(months=1)).replace(day=1)

# --- KPI (KEY PERFORMANCE INDICATORS) ---
kpi1, kpi2, kpi3 = st.columns(3)

harga_value = None
harga_value_source = "Historis"
if selected_date_data is not None:
    if 'harga_beras' in selected_date_data and pd.notnull(selected_date_data['harga_beras']):
        harga_value = pd.to_numeric(selected_date_data['harga_beras'], errors='coerce')
    elif 'harga_prediksi' in selected_date_data and pd.notnull(selected_date_data['harga_prediksi']):
        harga_value = pd.to_numeric(selected_date_data['harga_prediksi'], errors='coerce')
        harga_value_source = "Prediksi"
kpi1.metric(
    label=f"Harga ({selected_historical_date.strftime('%B %Y')} - {harga_value_source})" if selected_historical_date else "Harga Terpilih",
    value=f"Rp {harga_value:,.0f}" if pd.notnull(harga_value) else "N/A"
)

pred_price_numeric = None
source_kpi = "N/A"
if target_prediction_date:
    predicted_data_df = df_forecast[(df_forecast['provinsi'] == selected_province) & (df_forecast['tanggal'] == target_prediction_date)]
    if not predicted_data_df.empty:
        pred_price_numeric = pd.to_numeric(predicted_data_df.iloc[0]['harga_prediksi'], errors='coerce')
        source_kpi = "(Prediksi)"
    else:
        hist_next_month = df_hist[(df_hist['provinsi'] == selected_province) & (df_hist['tanggal'] == target_prediction_date)]
        if not hist_next_month.empty:
            pred_price_numeric = pd.to_numeric(hist_next_month.iloc[0]['harga_beras'], errors='coerce')
            source_kpi = "(Historis)"
delta_value = "N/A"
if pd.notnull(harga_value) and pd.notnull(pred_price_numeric):
    delta_value = f"Rp {pred_price_numeric - harga_value:,.0f}"
kpi2.metric(
    label=f"Harga Bulan Depan {source_kpi} ({target_prediction_date.strftime('%B %Y')})" if target_prediction_date else "Harga Bulan Depan",
    value=f"Rp {pred_price_numeric:,.0f}" if pd.notnull(pred_price_numeric) else "N/A",
    delta=delta_value
)

kpi3.metric(
    label="Akurasi Model (MAE)",
    value="~ Rp 110",
    help="Rata-rata, prediksi model pada data historis hanya meleset sebesar Rp 110 dari harga aktual."
)

# --- [PERBAIKAN] RINGKASAN OTOMATIS ---
st.subheader("📝 Ringkasan Otomatis")

# Ambil harga untuk tanggal yang dipilih (variabel `harga_value` dari KPI 1 sudah ada)
harga_sekarang_num = harga_value

# Ambil harga untuk bulan depan (variabel `pred_price_numeric` dari KPI 2 sudah ada)
harga_bulan_depan_num = pred_price_numeric
sumber_bulan_depan = source_kpi.strip("()") if source_kpi != "N/A" else None

# Hasilkan ringkasan jika kedua harga valid
if pd.notnull(harga_sekarang_num) and pd.notnull(harga_bulan_depan_num) and target_prediction_date:
    selisih = harga_bulan_depan_num - harga_sekarang_num
    trend = "kenaikan" if selisih > 0 else "penurunan" if selisih < 0 else "stabil"
    emoji = "🔺" if trend == "kenaikan" else "🔻" if trend == "penurunan" else "⚖️"

    st.markdown(
        f"""
        {emoji} Berdasarkan data tanggal terpilih, pada bulan **{target_prediction_date.strftime('%B %Y')}** diprediksi akan terjadi **{trend}** harga.

        Harga di provinsi **{selected_province}** diperkirakan berubah dari **Rp {harga_sekarang_num:,.0f}** menjadi **Rp {harga_bulan_depan_num:,.0f}**.
        *(Sumber data bulan depan: {sumber_bulan_depan})*
        """
    )
else:
    error_msgs = []
    if pd.isna(harga_sekarang_num) and selected_historical_date:
        error_msgs.append(f"harga untuk {selected_historical_date.strftime('%B %Y')}")
    if pd.isna(harga_bulan_depan_num) and target_prediction_date:
        error_msgs.append(f"harga untuk {target_prediction_date.strftime('%B %Y')}")
    
    st.warning(f"❗ Data tidak lengkap untuk menghasilkan ringkasan otomatis. Data berikut tidak tersedia: {', '.join(error_msgs)}.")

# --- KONTEN UTAMA ---
col1, col2 = st.columns([2, 1.2])

with col1:
    st.subheader(f"Grafik Prediksi Harga di {selected_province}")
    hist_province = df_hist[df_hist['provinsi'] == selected_province]
    forecast_province = df_forecast[df_forecast['provinsi'] == selected_province]
    fig_ts = go.Figure()
    if not hist_province.empty:
        fig_ts.add_trace(go.Scatter(x=hist_province['tanggal'], y=hist_province['harga_beras'], mode='lines', name='Harga Historis', line=dict(color='royalblue')))
    if not forecast_province.empty:
        fig_ts.add_trace(go.Scatter(x=forecast_province['tanggal'], y=forecast_province['harga_prediksi'], mode='lines+markers', name='Harga Prediksi', line=dict(dash='dash', color='red')))
    fig_ts.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader(f"Peta Risiko Kenaikan Harga Nasional ({target_prediction_date.strftime('%B %Y') if target_prediction_date else 'Bulan Depan'})")
    map_data = []
    if selected_historical_date and target_prediction_date:
        for prov in df_hist['provinsi'].unique():
            current_price = None
            next_month_price = None
            
            current_price_df = df_hist[(df_hist['provinsi'] == prov) & (df_hist['tanggal'] == selected_historical_date)]
            if not current_price_df.empty:
                current_price = pd.to_numeric(current_price_df.iloc[0]['harga_beras'], errors='coerce')
            else:
                current_price_df = df_forecast[(df_forecast['provinsi'] == prov) & (df_forecast['tanggal'] == selected_historical_date)]
                if not current_price_df.empty:
                    current_price = pd.to_numeric(current_price_df.iloc[0]['harga_prediksi'], errors='coerce')

            next_month_df = df_forecast[(df_forecast['provinsi'] == prov) & (df_forecast['tanggal'] == target_prediction_date)]
            if not next_month_df.empty:
                next_month_price = pd.to_numeric(next_month_df.iloc[0]['harga_prediksi'], errors='coerce')
            else:
                next_month_df = df_hist[(df_hist['provinsi'] == prov) & (df_hist['tanggal'] == target_prediction_date)]
                if not next_month_df.empty:
                    next_month_price = pd.to_numeric(next_month_df.iloc[0]['harga_beras'], errors='coerce')

            if pd.notnull(current_price) and pd.notnull(next_month_price) and current_price > 0:
                kenaikan_persen = ((next_month_price - current_price) / current_price) * 100
                map_data.append({'provinsi': prov, 'kenaikan_persen': kenaikan_persen})

    if map_data:
        df_map = pd.DataFrame(map_data)
        fig_map = px.choropleth(
            df_map, geojson=geojson_url, locations='provinsi', featureidkey="properties.Propinsi",
            color='kenaikan_persen', color_continuous_scale="Reds", scope="asia",
            labels={'kenaikan_persen': 'Kenaikan Harga (%)'}
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar=dict(title="Kenaikan (%)"))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("❗ Tidak cukup data untuk menampilkan peta risiko.")

with col2:
    st.subheader(f"Analisis Unsupervised: Risiko Harga di {selected_province}")
    
    with st.container(border=True):
        st.markdown("##### Momentum Tren Harga")
        delta_momentum = "⚪ **Data Tidak Tersedia**"
        if selected_date_data is not None and 'harga_beras_bulan_lalu' in selected_date_data:
            current_price_numeric = harga_value
            last_month_price_numeric = pd.to_numeric(selected_date_data['harga_beras_bulan_lalu'], errors='coerce')
            if pd.notnull(current_price_numeric) and pd.notnull(last_month_price_numeric):
                delta_value = current_price_numeric - last_month_price_numeric
                if delta_value > 100: delta_momentum = "🔴 **Cenderung Naik**"
                elif delta_value < -100: delta_momentum = "🟢 **Cenderung Turun**"
                else: delta_momentum = "🟡 **Stabil**"
        st.markdown(delta_momentum)

    if selected_date_data is not None and kmeans_model and iso_model and scaler:
        cluster_features = ['harga_beras', 'stok_beras_ton', 'jumlah_bencana', 'jumlah_curah_hujan']
        data_for_unsupervised = selected_date_data.copy()
        if 'harga_prediksi' in data_for_unsupervised and 'harga_beras' not in data_for_unsupervised:
            data_for_unsupervised['harga_beras'] = data_for_unsupervised['harga_prediksi']
        
        missing_cols = [col for col in cluster_features if col not in data_for_unsupervised or pd.isna(data_for_unsupervised[col])]
        if missing_cols:
            st.warning(f"❗ Kolom tidak tersedia untuk analisis unsupervised: {', '.join(missing_cols)}")
        else:
            try:
                fitur_input = pd.DataFrame([data_for_unsupervised[cluster_features]]).apply(pd.to_numeric, errors='coerce')
                if fitur_input.isnull().any().any():
                    st.warning("❗ Data tidak valid untuk analisis unsupervised.")
                else:
                    fitur_scaled = scaler.transform(fitur_input)
                    
                    cluster_label = kmeans_model.predict(fitur_scaled)[0]
                    with st.container(border=True):
                        st.markdown("##### Segmentasi Risiko (K-Means)")
                        st.markdown(f"📊 Data masuk ke **Cluster {cluster_label}**")
                        if cluster_label == 0: st.markdown("🟢 **Kondisi Aman** — Harga stabil, suplai mencukupi.")
                        elif cluster_label == 1: st.markdown("🟠 **Waspada** — Ada tekanan pada harga atau stok.")
                        elif cluster_label == 2: st.markdown("🔴 **Kritis** — Stok rendah, bencana, dan harga tinggi.")
                    
                    anomaly = iso_model.predict(fitur_scaled)[0]
                    with st.container(border=True):
                        st.markdown("##### Deteksi Anomali (Isolation Forest)")
                        if anomaly == -1: st.markdown("🚨 **Anomali Terdeteksi!** Kondisi menyimpang dari pola normal.")
                        else: st.markdown("✅ **Normal** — Tidak ada anomali signifikan.")
            except Exception as e:
                st.error(f"❗ Gagal melakukan analisis unsupervised: {e}")
    else:
        st.warning("❗ Model unsupervised atau data tidak tersedia.")
