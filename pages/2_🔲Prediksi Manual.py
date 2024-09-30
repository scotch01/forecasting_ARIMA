import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
# from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

# Fungsi untuk memuat dataset dari file Excel
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])  # Sesuaikan dengan nama kolom tanggal
    df.set_index('Tanggal', inplace=True)
    return df    

# # Fungsi untuk mengidentifikasi outlier berdasarkan Z-Score
# def detect_outliers(df, column, z_threshold=3):
#     z_scores = np.abs(stats.zscore(df[column]))
#     return df[z_scores > z_threshold]

# # Fungsi untuk menangani outliers dengan Capping (Winsorizing) menggunakan Z-Score
# def winsorize_series(series, z_threshold=3):
#     z_scores = np.abs(stats.zscore(series))
#     capped_series = np.where(z_scores > z_threshold, np.sign(series) * z_threshold * np.std(series) + np.mean(series), series)
#     return pd.Series(capped_series, index=series.index)

# Fungsi untuk melakukan uji stasioneritas menggunakan ADF test
def check_stationarity(series, signif=0.05):
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    is_stationary = p_value < signif
    return is_stationary, p_value, result

# Fungsi untuk melakukan differencing pada data
def make_stationary(series):
    return series.diff().dropna()

# Fungsi untuk melakukan differencing pada data secara iteratif hingga stasioner
def difference_until_stationary(series, signif=0.05, max_diff=5):
    """
    Fungsi untuk melakukan differencing pada data secara iteratif hingga data menjadi stasioner.

    Parameters:
    series (pd.Series): Data yang akan dilakukan differencing.
    signif (float): Nilai signifikansi untuk uji ADF. Default adalah 0.05.
    max_diff (int): Jumlah maksimal differencing yang diperbolehkan. Default adalah 5.

    Returns:
    pd.Series: Data yang telah dilakukan differencing hingga stasioner.
    int: Jumlah differencing yang telah dilakukan.
    """
    n_diff = 0
    while not check_stationarity(series, signif)[0] and n_diff < max_diff:
        series = series.diff().dropna()  # Lakukan differencing
        n_diff += 1
    return series, n_diff

# Description
st.header("Manual News Popularity Forecasting using ARIMA")
st.write("This app allows manual parameter tuning for ARIMA models.")

# File uploader untuk dataset
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Memuat dataset
    df = load_data(uploaded_file)
    
    # 1. Menampilkan head dari dataset
    st.subheader('Read Data')
    st.write("Data Head Overview")
    st.write(df.head())
    
    # 2. Menampilkan cek deskriptif dataset
    st.write('Statistik Deskriptif Dataset')
    st.write(df.describe())
    
    # 3. Menampilkan grafik visualisasi
    fig_overall = go.Figure()
    fig_overall.add_trace(go.Scatter(x=df.index, y=df['Views'], mode='lines', name='Views'))
    fig_overall.update_layout(title='Visualisasi Data', xaxis_title='Date', yaxis_title='Views')
    st.plotly_chart(fig_overall)

    # # 4. Identifikasi outlier
    # st.subheader("Identifikasi Outlier")
    # z_threshold = st.slider("Pilih Z-Score threshold untuk mendeteksi outlier:", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    # outliers = detect_outliers(df, 'Views', z_threshold=z_threshold)
    
    # st.write(f"Jumlah data outlier terdeteksi: {len(outliers)}")
    # st.write("Data yang merupakan outlier:")
    # st.write(outliers)
    
    # # 5. Penanganan Outlier dengan Winsorizing (Capping)
    # st.subheader("Penanganan Outlier dengan Winsorizing (Capping)")
    # df['Views'] = winsorize_series(df['Views'], z_threshold=z_threshold)
    
    # # Tampilkan hasil data setelah penanganan outlier
    # st.write("Data setelah penanganan outlier dengan capping (Winsorizing):")
    # st.write(df.describe())

    # 6. Memisahkan data menjadi data train dan data test
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # 7. Preprocessing Data, Check Stationarity
    st.subheader('Preprocessing Data')
    st.write("Stationarity Check on Data Train")
    is_stationary, p_value, adf_result = check_stationarity(train['Views'])
    st.write(f"p-value: {p_value}")
    st.write(f"Data Train Stasioner: {is_stationary}")

    # 8. Lakukan differencing jika data train tidak stasioner
    if not is_stationary:
        st.write("Data Train tidak stasioner. Melakukan differencing...")
        train_diff, n_diff = difference_until_stationary(train['Views'])
        st.write(f"Jumlah differencing yang dilakukan: {n_diff}")
    else:
        st.write("Data Train sudah stasioner. Tidak perlu differencing.")

    # 9. Menampilkan visualisasi data train dan data test
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(x=train.index, y=train['Views'], mode='lines', name='Data Train'))
    fig_split.add_trace(go.Scatter(x=test.index, y=test['Views'], mode='lines', name='Data Test'))
    fig_split.update_layout(title='Visualisasi Data Train dan Test', xaxis_title='Date', yaxis_title='Views')
    st.plotly_chart(fig_split)
    st.write(f"Total baris data train: {len(train)}")
    st.write(f"Total baris data test: {len(test)}")
          
    # 10. Membangun model ARIMA dengan parameter manual
    st.subheader("ARIMA Model - Input Parameter Manual")
    st.write("Silakan pilih parameter model ARIMA (p, d, q)")

    # 11. Input parameter p, d, q dengan slider
    p = st.slider("Masukkan nilai p (Order AR)", min_value=0, max_value=10, value=0)
    d = st.slider("Masukkan nilai d (Order Difference)", min_value=0, max_value=5, value=0)
    q = st.slider("Masukkan nilai q (Order MA)", min_value=0, max_value=10, value=0)

    # 12. Membangun model ARIMA dengan parameter yang dipilih
    model = ARIMA(train['Views'], order=(p, d, q))
    model_fit = model.fit()

    # 13. Melakukan prediksi pada data test
    try:
        forecast = model_fit.forecast(steps=len(test))
        test['Forecast'] = forecast
    except Exception as e:
        st.write("Error saat melakukan prediksi:", e)

    # 14. Menghitung metrik evaluasi MAPE dan RMSE
    st.subheader("Evaluation Metrics")
    try:
        # Menghitung MAE
        mae = mean_absolute_error(test['Views'], test['Forecast'])
        st.write(f"MAE: {mae}")

        # Menghitung MAPE
        mape = mean_absolute_percentage_error(test['Views'], test['Forecast'])
        st.write(f"MAPE: {mape * 100:.2f}%")

        # Menghitung RMSE
        rmse = np.sqrt(mean_squared_error(test['Views'], test['Forecast']))
        st.write(f"RMSE: {rmse}")

    except Exception as e:
        st.write("Error saat menghitung metrik evaluasi:", e)

    # 15. Menampilkan hasil peramalan dalam tabel
    st.subheader('Forecast Results')
    st.write("Hasil peramalan data test dalam bentuk tabel:")
    test['Perbedaan Nilai'] = test['Views'] - test['Forecast']
    st.write(test[['Views', 'Forecast', 'Perbedaan Nilai']])

    # 16. Menampilkan rata - rata forecast
    average_value = test['Views'].mean()
    average_value_difference = test['Perbedaan Nilai'].mean()
    average_forecast = (test['Forecast'].mean())
    st.write("Average Actual Views:", average_value)
    st.write("Average Forecast Views:", average_forecast)
    st.write("Average Views Difference:", average_value_difference)

    # 17. Visualisasi hasil peramalan
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=train.index, y=train['Views'], mode='lines', name='Data Train'))
    fig_forecast.add_trace(go.Scatter(x=test.index, y=test['Views'], mode='lines', name='Data Test'))
    fig_forecast.add_trace(go.Scatter(x=test.index, y=test['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))
    fig_forecast.update_layout(title='Visualisasi Forecast vs Actual', xaxis_title='Date', yaxis_title='Views')
    st.plotly_chart(fig_forecast)

    # 18. Forecasting
    steps = st.number_input("Enter the number of steps for forecasting", min_value=1, max_value=30, value=6, key='steps')
    future_forecast = model_fit.forecast(len(test) + steps)[-steps:]
    st.subheader(f"Forecast for next {steps} Months")

    # 19. Membuat dataframe untuk hasil peramalan masa depan
    df_future = pd.DataFrame({
        'Tanggal': pd.date_range(df.index[-1], periods=steps + 1, freq='MS')[1:],
        'Future Forecast': future_forecast
    })
    st.write("Tabel hasil peramalan masa depan:")
    st.write(df_future.reset_index(drop=True))

    # 20. Visualisasi hasil peramalan masa depan
    fig = go.Figure()
    # Menambahkan data actual
    fig.add_trace(go.Scatter(x=df.index, y=df['Views'], mode='lines', name='Actual'))
    # Menambahkan data forecast ARIMA pada data test
    fig.add_trace(go.Scatter(x=test.index, y=test['Forecast'], mode='lines', name='ARIMA Forecast', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df_future.index, y=df_future['Future Forecast'], mode='lines', name='Future Manual Forecast', line=dict(color='green')))
    # Mengatur layout
    fig.update_layout(
    title='Plot Forecast Masa Depan',
    xaxis_title='Date',
    yaxis_title='Views',
    width=800,  # Lebar grafik
    height=600, # Tinggi grafik
    template='plotly_dark'  # Tema plot
    )
    # Menampilkan plot dan tabel hasil peramalan
    st.plotly_chart(fig)


# else:
#     st.warning("Please upload an Excel file to proceed.")
