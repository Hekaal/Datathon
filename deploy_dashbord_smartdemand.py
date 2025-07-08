# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import json

# --- KONFIGURASI HALAMAN ---
# Mengatur konfigurasi halaman Streamlit. Ini harus menjadi perintah st pertama.
st.set_page_config(
    page_title="Dashboard SmartDemand-ID",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UNTUK MEMUAT DATA & ASET ---
# Menggunakan cache agar data dan model only loaded once for faster performance.
@st.cache_data
def load_raw_data():
    """Memuat data mentah historis dan prediksi."""
    hist_df = None
    forecast_df = None

    try:
        hist_df = pd.read_csv('master_table_modified_fix.csv')
        print("master_table_fix.csv loaded successfully.")
        print("Columns in master_table_modified_fix.csv:", hist_df.columns.tolist())

        forecast_df = pd.read_csv('hasil_prediksi_12_bulan.csv')
        print("hasil_prediksi_12_bulan.csv loaded successfully.")
        print("Columns in hasil_prediksi_12_bulan.csv:", forecast_df.columns.tolist())

        return hist_df, forecast_df
    except FileNotFoundError as e:
        st.error(f"File data not found: {e}. Make sure 'master_table_fix.csv' and 'hasil_prediksi_12_bulan.csv' are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during raw data loading: {e}")
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
geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province.json" # Default GeoJSON URL

# If raw data or model failed to load, stop application execution.
if df_hist_raw is None or df_forecast_raw is None or model is None or geojson_url is None:
    st.stop()

# Load all unsupervised models
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

kmeans_model, iso_model, scaler = load_unsupervised_models()

# --- DATA PREPROCESSING (outside cache to apply consistently) ---

# Handle 'tanggal' column for historical data
df_hist = df_hist_raw.copy() # Work on a copy to avoid modifying cached raw data

# Now that 'tanggal' is expected to be in the CSV, convert it directly
if 'tanggal' in df_hist.columns:
    df_hist['tanggal'] = pd.to_datetime(df_hist['tanggal'], errors='coerce')
    print("Date column 'tanggal' found and converted in df_hist.")
elif 'Tanggal' in df_hist.columns: # Keep this as a fallback if 'tanggal' wasn't the exact name
    df_hist['tanggal'] = pd.to_datetime(df_hist['Tanggal'], errors='coerce')
    print("Date column 'Tanggal' found and converted in df_hist.")
# Removed the old logic for creating 'tanggal' from 'tahun' and 'bulan'
else:
    st.error("Could not find a 'tanggal' or 'Tanggal' column in master_table_fix.csv. Please ensure the date column exists and is named 'tanggal' or 'Tanggal'. Stopping application.")
    st.stop()


# Convert price and stock columns to numeric in historical data
for col in ['harga_beras', 'harga_beras_bulan_lalu', 'stok_beras_ton']:
    if col in df_hist.columns:
        if df_hist[col].dtype == 'object':
            df_hist[col] = df_hist[col].astype(str).str.replace(',', '', regex=False)
            df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')
        elif not pd.api.types.is_numeric_dtype(df_hist[col]):
             st.warning(f"Column '{col}' in df_hist is not object or numeric type: {df_hist[col].dtype}. Attempting numeric conversion anyway.")
             df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')

    if df_hist[col].isnull().any():
        st.warning(f"NaN values introduced in '{col}' column of df_hist after conversion. Check original data for non-numeric entries.")


# Create 'tanggal' column for forecast data (assuming it has a date column)
df_forecast = df_forecast_raw.copy() # Work on a copy
date_col_forecast = None
if 'Tanggal' in df_forecast.columns:
    date_col_forecast = 'Tanggal'
elif 'tanggal' in df_forecast.columns:
    date_col_forecast = 'tanggal'

if date_col_forecast:
    df_forecast['tanggal'] = pd.to_datetime(df_forecast[date_col_forecast], errors='coerce')
    print(f"Date column '{date_col_forecast}' found and converted in df_forecast.")
else:
    st.error("Could not find a date column named 'Tanggal' or 'tanggal' in hasil_prediksi_12_bulan.csv. Please check the column names in your CSV file. Stopping application.")
    st.stop()


# Filter out rows with NaT (Not a Time) in the date column if necessary, or handle NaNs later.
# For now, let's keep them and handle potential errors in plotting.
# df_hist.dropna(subset=['tanggal'], inplace=True)
# df_forecast.dropna(subset=['tanggal'], inplace=True)


# --- UI SIDEBAR ---
with st.sidebar:
    st.header("Navigasi Dashboard")
    # Membuat daftar provinsi yang unik dan terurut untuk dropdown
    list_provinsi = sorted(df_hist['provinsi'].unique())
    selected_province = st.selectbox(
        'Pilih Provinsi untuk Analisis Detail:',
        options=list_provinsi
    )

    # Gabungkan tanggal dari data historis dan prediksi
    hist_dates = df_hist[df_hist['provinsi'] == selected_province]['tanggal'].dropna()
    pred_dates = df_forecast[df_forecast['provinsi'] == selected_province]['tanggal'].dropna()

    combined_dates = pd.concat([hist_dates, pred_dates]).drop_duplicates().sort_values()

    selected_historical_date = None
    if not combined_dates.empty:
        selected_historical_date = st.selectbox(
            'Pilih Tanggal Historis:',
            options=combined_dates,
            index=len(combined_dates) - 1,
            format_func=lambda x: x.strftime('%B %Y')
        )
    else:
        st.warning(f"Tidak ada data yang tersedia untuk {selected_province}.")

    st.info("Dashboard ini displays rice price predictions and their drivers to support national food stability.")

# --- JUDUL UTAMA ---
st.title("🍚 Dashboard SmartDemand-ID")
st.markdown("Analisis Prediktif Harga Beras untuk Mendukung Stabilitas Pangan Nasional")

# --- KEY PERFORMANCE INDICATORS (KPIs) ---
# Get data for the selected historical date
selected_date_data = None
if selected_historical_date is not None:
    selected_date_data_df = df_hist[(df_hist['provinsi'] == selected_province) & (df_hist['tanggal'] == selected_historical_date)]

    if not selected_date_data_df.empty:
        selected_date_data = selected_date_data_df.iloc[0]
    else:
        # Gunakan data prediksi jika tidak ada data historis
        selected_date_data_df = df_forecast[(df_forecast['provinsi'] == selected_province) & (df_forecast['tanggal'] == selected_historical_date)]
        if not selected_date_data_df.empty:
            selected_date_data = selected_date_data_df.iloc[0]


# Determine the latest historical date for the selected province
latest_hist_date_for_province = None
hist_province_latest_df = df_hist[df_hist['provinsi'] == selected_province].sort_values('tanggal')
if not hist_province_latest_df.empty:
    # Ensure 'tanggal' is not NaT before getting the latest date
    valid_hist_dates = hist_province_latest_df['tanggal'].dropna()
    if not valid_hist_dates.empty:
        latest_hist_date_for_province = valid_hist_dates.iloc[-1]

# Determine the target prediction date (month after selected historical date)
target_prediction_date = None
predicted_data_for_target_date = None

if selected_historical_date is not None and pd.notna(selected_historical_date):
    try:
        # Calculate the month immediately following the selected historical date
        target_prediction_date = selected_historical_date + pd.DateOffset(months=1)
        # Ensure it's the first day of the month for matching
        target_prediction_date = target_prediction_date.replace(day=1)

        # Retrieve the prediction data for the target prediction date
        forecast_for_province = df_forecast[df_forecast['provinsi'] == selected_province]
        if not forecast_for_province.empty and target_prediction_date is not None and pd.notna(target_prediction_date):
            # Find the prediction entry matching the target date
            predicted_data_df = forecast_for_province[forecast_for_province['tanggal'] == target_prediction_date]
            if not predicted_data_df.empty:
                predicted_data_for_target_date = predicted_data_df.iloc[0]

    except Exception as e:
        print(f"Could not calculate target prediction date or retrieve data: {e}")
        target_prediction_date = None
        predicted_data_for_target_date = None

def get_prediction_for_next_month(df_forecast, selected_province, selected_date):
    """
    Mengambil data prediksi harga beras untuk provinsi dan bulan setelah tanggal yang dipilih.
    Pencocokan dilakukan berdasarkan bulan dan tahun (mengabaikan hari).
    """
    if pd.isna(selected_date):
        return None, None

    target_month = selected_date + pd.DateOffset(months=1)
    target_month = target_month.replace(day=1)  # Samakan ke awal bulan

    forecast_for_province = df_forecast[df_forecast['provinsi'] == selected_province]

    if forecast_for_province.empty:
        return None, target_month

    # Cocokkan berdasarkan bulan dan tahun
    predicted_data_df = forecast_for_province[
        (forecast_for_province['tanggal'].dt.month == target_month.month) &
        (forecast_for_province['tanggal'].dt.year == target_month.year)
    ]

    if predicted_data_df.empty:
        return None, target_month

    return predicted_data_df.iloc[0], target_month


kpi1, kpi2, kpi3 = st.columns(3)

# --- KPI 1: Harga Historis (selected historical date) ---
harga_value = None
if selected_date_data is not None:
    if 'harga_beras' in selected_date_data and pd.notnull(selected_date_data['harga_beras']):
        harga_value = pd.to_numeric(selected_date_data['harga_beras'], errors='coerce')
    elif 'harga_prediksi' in selected_date_data and pd.notnull(selected_date_data['harga_prediksi']):
        harga_value = pd.to_numeric(selected_date_data['harga_prediksi'], errors='coerce')

if harga_value is not None:
    kpi1.metric(
        label=f"Harga ({selected_historical_date.strftime('%B %Y')})",
        value=f"Rp {harga_value:,.0f}"
    )
else:
    kpi1.metric(label="Harga", value="N/A")


# --- KPI 2: Harga Prediksi Bulan Depan (Fallback ke data historis jika prediksi tidak tersedia) ---
if target_prediction_date:
    # Coba ambil dari forecast dulu
    forecast_for_province = df_forecast[df_forecast['provinsi'] == selected_province]
    predicted_data_df = forecast_for_province[
        (forecast_for_province['tanggal'].dt.month == target_prediction_date.month) &
        (forecast_for_province['tanggal'].dt.year == target_prediction_date.year)
    ]

    if not predicted_data_df.empty:
        predicted_data_for_target_date = predicted_data_df.iloc[0]
        pred_price_numeric = pd.to_numeric(predicted_data_for_target_date['harga_prediksi'], errors='coerce')
        source = " (Prediksi)"
    else:
        # Fallback: Coba cari harga aktual dari data historis
        hist_next_month = df_hist[
            (df_hist['provinsi'] == selected_province) &
            (df_hist['tanggal'].dt.month == target_prediction_date.month) &
            (df_hist['tanggal'].dt.year == target_prediction_date.year)
        ]

        if not hist_next_month.empty:
            predicted_data_for_target_date = hist_next_month.iloc[0]
            pred_price_numeric = pd.to_numeric(predicted_data_for_target_date['harga_beras'], errors='coerce')
            source = " (Historis)"
        else:
            predicted_data_for_target_date = None

    if predicted_data_for_target_date is not None:
        selected_price_numeric = None
        delta_value = "N/A"

        # Tentukan harga saat ini yang valid
        if harga_value is not None and pd.notnull(harga_value) and pd.notnull(pred_price_numeric):
            selected_price_numeric = harga_value

        # Hitung delta jika kedua harga valid
        if selected_price_numeric is not None and pd.notnull(selected_price_numeric) and pd.notnull(pred_price_numeric):
            delta_value = f"Rp {pred_price_numeric - selected_price_numeric:,.0f}"

        delta_value = "N/A"
        if pd.notnull(selected_price_numeric) and pd.notnull(pred_price_numeric):
            delta_value = f"Rp {pred_price_numeric - selected_price_numeric:,.0f}"


        kpi2_label_text = f"Harga Bulan Depan{source}"
        if pd.notna(selected_historical_date) and pd.notna(latest_hist_date_for_province):
            if selected_historical_date.strftime('%Y-%m-%d') == latest_hist_date_for_province.strftime('%Y-%m-%d'):
                kpi2_label_text = f"Harga Prediksi Bulan Depan{source}"

        kpi2.metric(
            label=f"{kpi2_label_text} ({target_prediction_date.strftime('%B %Y')})",
            value=f"Rp {pred_price_numeric:,.0f}",
            delta=delta_value
        )
    else:
        kpi2.metric(label=f"Harga Bulan Depan ({target_prediction_date.strftime('%B %Y')})", value="N/A", delta="N/A")
else:
    kpi2.metric(label="Harga Bulan Depan", value="N/A", delta="N/A")


# --- KPI 3: Akurasi Model ---
kpi3.metric(
    label="Akurasi Model (MAE)",
    value="~ Rp 161",
    help="On average, model predictions on historical data were only off by Rp 161 from the actual price."
)

st.subheader("📝 Ringkasan Otomatis")

# Cek data historis dan prediksi untuk log (debugging)

# Ambil harga historis saat ini
harga_hist = None
if selected_date_data is not None:
    if 'harga_beras' in selected_date_data:
        harga_hist = pd.to_numeric(selected_date_data['harga_beras'], errors='coerce')
    elif 'harga_prediksi' in selected_date_data:
        harga_hist = pd.to_numeric(selected_date_data['harga_prediksi'], errors='coerce')

# Ambil harga prediksi bulan depan (cek dari data prediksi terlebih dahulu)
harga_pred = None
source = None
if predicted_data_for_target_date is not None:
    if 'harga_prediksi' in predicted_data_for_target_date:
        harga_pred = pd.to_numeric(predicted_data_for_target_date['harga_prediksi'], errors='coerce')
        source = 'prediksi'
    elif 'harga_beras' in predicted_data_for_target_date:
        harga_pred = pd.to_numeric(predicted_data_for_target_date['harga_beras'], errors='coerce')
        source = 'historis'

# Validasi dan tampilkan ringkasan
if pd.notnull(harga_hist) and pd.notnull(harga_pred):
    selisih = harga_pred - harga_hist
    if selisih > 0:
        trend = "kenaikan"
        emoji = "🔺"
    elif selisih < 0:
        trend = "penurunan"
        emoji = "🔻"
    else:
        trend = "stabil"
        emoji = "⚖️"

    st.markdown(
        f"{emoji} Prediksi bulan depan menunjukkan **{trend}** dari harga saat ini "
        f"**Rp {harga_hist:,.0f}** menjadi **Rp {harga_pred:,.0f}** "
        f"pada bulan **{target_prediction_date.strftime('%B %Y')}** "
        f"di provinsi **{selected_province}** (sumber: {source})."
    )
else:
    error_msgs = []
    if harga_hist is None or pd.isna(harga_hist):
        error_msgs.append("harga historis tidak tersedia")
    if harga_pred is None or pd.isna(harga_pred):
        error_msgs.append("harga prediksi bulan depan tidak tersedia")

    st.warning(f"❗ Data harga tidak lengkap untuk menghasilkan ringkasan otomatis: {', '.join(error_msgs)}.")

# --- MAIN DASHBOARD (2 Column Layout) ---
col1, col2 = st.columns([2, 1.2]) # Left column wider

with col1:
    # --- TIME SERIES PREDICTION CHART ---
    st.subheader(f"Grafik Prediksi Harga di {selected_province}")

    hist_province = df_hist[df_hist['provinsi'] == selected_province]
    forecast_province = df_forecast[df_forecast['provinsi'] == selected_province]

    fig_ts = go.Figure()
    # Add historical data line
    if not hist_province.empty:
        fig_ts.add_trace(go.Scatter(x=hist_province['tanggal'], y=hist_province['harga_beras'], mode='lines', name='Harga Historis', line=dict(color='royalblue')))
    # Add prediction data line
    if not forecast_province.empty:
         fig_ts.add_trace(go.Scatter(x=forecast_province['tanggal'], y=forecast_province['harga_prediksi'], mode='lines+markers', name='Harga Prediksi', line=dict(dash='dash', color='red')))

    if hist_province.empty and forecast_province.empty:
        st.warning(f"No data available for {selected_province} to display time series.")

    fig_ts.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_ts, use_container_width=True)

# --- RISK MAP ---
# Hitung bulan depan untuk judul
if selected_historical_date is not None and pd.notna(selected_historical_date):
    bulan_depan = (selected_historical_date + pd.DateOffset(months=1)).strftime('%B %Y')
    st.subheader(f"Peta Risiko Kenaikan Harga Nasional (Bulan {bulan_depan})")
else:
    st.subheader("Peta Risiko Kenaikan Harga Nasional (Bulan Depan)")

map_data = []

if df_hist is not None and df_forecast is not None and selected_historical_date is not None:
    for prov in df_hist['provinsi'].unique():
        # Ambil data historis dan prediksi untuk provinsi ini
        prov_hist_data = df_hist[df_hist['provinsi'] == prov].sort_values('tanggal')
        prov_forecast_data = df_forecast[df_forecast['provinsi'] == prov].sort_values('tanggal')

        # Ambil harga historis pada tanggal yang dipilih pengguna
        harga_hist = None
        hist_match = prov_hist_data[prov_hist_data['tanggal'] == selected_historical_date]
        if not hist_match.empty:
            harga_hist = pd.to_numeric(hist_match.iloc[0]['harga_beras'], errors='coerce')

        # Hitung target bulan depan
        target_date = selected_historical_date + pd.DateOffset(months=1)
        target_date = target_date.replace(day=1)

        # Coba ambil harga dari data historis bulan depan dulu
        harga_bulan_depan = None
        hist_next = prov_hist_data[
            (prov_hist_data['tanggal'].dt.month == target_date.month) &
            (prov_hist_data['tanggal'].dt.year == target_date.year)
        ]
        if not hist_next.empty:
            harga_bulan_depan = pd.to_numeric(hist_next.iloc[0]['harga_beras'], errors='coerce')
        else:
            # Kalau tidak ada di historis, ambil dari prediksi
            pred_next = prov_forecast_data[
                (prov_forecast_data['tanggal'].dt.month == target_date.month) &
                (prov_forecast_data['tanggal'].dt.year == target_date.year)
            ]
            if not pred_next.empty:
                harga_bulan_depan = pd.to_numeric(pred_next.iloc[0]['harga_prediksi'], errors='coerce')

        # Hitung persen kenaikan jika data lengkap
        if pd.notnull(harga_hist) and pd.notnull(harga_bulan_depan) and harga_hist != 0:
            persen_kenaikan = ((harga_bulan_depan - harga_hist) / harga_hist) * 100
            map_data.append({'provinsi': prov, 'kenaikan_persen': persen_kenaikan})
        elif pd.notnull(harga_hist) and pd.notnull(harga_bulan_depan) and harga_hist == 0:
            # Hindari pembagian dengan nol
            map_data.append({'provinsi': prov, 'kenaikan_persen': 0})

# Buat dataframe untuk peta
df_map = pd.DataFrame(map_data)

# Tampilkan choropleth map jika datanya ada
if not df_map.empty:
    fig_map = px.choropleth(
        df_map,
        geojson=geojson_url,
        locations='provinsi',
        featureidkey="properties.Propinsi",
        color='kenaikan_persen',
        color_continuous_scale="Reds",
        scope="asia",
        labels={'kenaikan_persen': 'Kenaikan Harga (%)'}
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Kenaikan (%)")
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("❗ Tidak ada data yang cukup untuk menampilkan peta risiko.")

with col2:
    st.subheader(f"Analisis Unsupervised: Risiko Harga di {selected_province}")

    # Use a container with a border for a neater look
    with st.container(border=True):
        st.markdown("##### Momentum Tren Harga")
        # Ensure prices are numeric before calculation
        delta_momentum = "Data Tidak Tersedia"
        # Use selected_date_data for momentum calculation if available
        data_for_momentum = selected_date_data

        if data_for_momentum is not None and 'harga_beras' in data_for_momentum and 'harga_beras_bulan_lalu' in data_for_momentum:
             current_price_numeric = pd.to_numeric(data_for_momentum['harga_beras'], errors='coerce')
             last_month_price_numeric = pd.to_numeric(data_for_momentum['harga_beras_bulan_lalu'], errors='coerce')

             if pd.notnull(current_price_numeric) and pd.notnull(last_month_price_numeric):
                  delta_value = current_price_numeric - last_month_price_numeric
                  if delta_value > 100: # Assume increase more than Rp 50 is an upward trend
                      delta_momentum = "🔴 **Cenderung Naik** (Harga bulan lalu lebih tinggi dari sebelumnya)"
                  elif delta_value < -100:
                      delta_momentum = "🟢 **Cenderung Turun**"
                  else:
                      delta_momentum = "🟡 **Stabil**"
        else:
             delta_momentum = "⚪ **Data Tidak Tersedia** (Data historis bulan lalu tidak lengkap untuk tanggal/provinsi terpilih)"

        st.markdown(delta_momentum)
    
    if selected_date_data is not None and kmeans_model and iso_model and scaler:
        # Fitur yang digunakan saat pelatihan model unsupervised
        cluster_features = ['harga_beras', 'stok_beras_ton', 'jumlah_bencana', 'jumlah_curah_hujan']

        # Validasi ketersediaan semua fitur
        missing_cols = [col for col in cluster_features if col not in selected_date_data or pd.isna(selected_date_data[col])]
        if missing_cols:
            st.warning(f"❗ Kolom berikut tidak tersedia atau memiliki nilai kosong untuk analisis unsupervised: {', '.join(missing_cols)}")
        else:
            try:
                fitur_input = pd.DataFrame([selected_date_data[cluster_features]])
                fitur_input = fitur_input.apply(pd.to_numeric, errors='coerce')

                # Cek nilai null setelah konversi
                if fitur_input.isnull().any().any():
                    st.warning("❗ Data tidak lengkap atau tidak valid untuk analisis unsupervised.")
                else:
                    # Scaling
                    fitur_scaled = scaler.transform(fitur_input)

                    # --- KMEANS CLUSTERING ---
                    cluster_label = kmeans_model.predict(fitur_scaled)[0]

                    with st.container(border=True):
                        st.markdown("##### Segmentasi Risiko (K-Means)")
                        st.markdown(f"📊 Data masuk ke **Cluster {cluster_label}**")

                        if cluster_label == 0:
                            st.markdown("🟢 **Kondisi Aman** — Harga stabil, suplai mencukupi.")
                        elif cluster_label == 1:
                            st.markdown("🟠 **Waspada** — Ada tekanan pada harga atau stok.")
                        elif cluster_label == 2:
                            st.markdown("🔴 **Kritis** — Kombinasi stok rendah, bencana, dan harga tinggi.")
                        else:
                            st.markdown("⚪ **Kategori tidak diketahui**")

                    # --- ISOLATION FOREST ANOMALY DETECTION ---
                    anomaly = iso_model.predict(fitur_scaled)[0]  # -1 = anomaly, 1 = normal

                    with st.container(border=True):
                        st.markdown("##### Deteksi Anomali (Isolation Forest)")
                        if anomaly == -1:
                            st.markdown("🚨 **Anomali Terdeteksi!** Kondisi saat ini menyimpang dari pola normal.")
                        else:
                            st.markdown("✅ **Normal** — Tidak ada indikasi keanehan signifikan.")
            except Exception as e:
                st.error(f"❗ Gagal melakukan analisis unsupervised: {e}")
    else:
        st.warning("❗ Model atau data tidak tersedia.")
