import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import itertools
import warnings

warnings.filterwarnings("ignore")

# Fungsi untuk memuat dataset dari file Excel
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])  # Sesuaikan dengan nama kolom tanggal
    df.set_index('Tanggal', inplace=True)
    return df    

# Fungsi untuk melakukan uji stasioneritas menggunakan ADF test
def check_stationarity(series, signif=0.05):
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    is_stationary = p_value < signif
    return is_stationary, p_value, result

# Fungsi untuk melakukan differencing pada data secara iteratif hingga stasioner
def difference_until_stationary(series, signif=0.05, max_diff=5):
    n_diff = 0
    while not check_stationarity(series, signif)[0] and n_diff < max_diff:
        series = series.diff().dropna()  # Lakukan differencing
        n_diff += 1
    return series, n_diff

# Fungsi untuk mengevaluasi model ARIMA dengan MAE, MAPE, dan RMSE
def evaluate_arima_model(train, test, order):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    
    # Menghitung Metrik Evaluasi
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast) * 100
    return mae, mape, rmse

# Fungsi untuk melakukan grid search ARIMA dengan evaluasi MAE, MAPE, dan RMSE
def grid_search_arima(train, test, p_values, d_values, q_values):
    best_mae, best_mape, best_rmse, best_cfg = float("inf"), float("inf"), float("inf"), None
    pdq = list(itertools.product(p_values, d_values, q_values))

    for param in pdq:
        try:
            mae, mape, rmse = evaluate_arima_model(train, test, param)
            if mae < best_mae:
                best_mae, best_mape, best_rmse, best_cfg = mae, mape, rmse, param
            st.write(f'ARIMA{param} - MAE: {mae:.2f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}')
        except:
            continue

    st.write(f"Best Parameters -> p: {best_cfg[0]}, d: {best_cfg[1]}, q: {best_cfg[2]}")
    
    return best_cfg

# Description
st.header("News Popularity Forecasting using ARIMA with Hyperparameter Tuning")
st.write("This app allows automatic hyperparameter tuning using grid search for ARIMA Models.")

# File uploader untuk dataset
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Inisialisasi best_cfg agar tidak menimbulkan error jika grid search belum dijalankan
best_cfg = None

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

    # 4. Memisahkan data menjadi data train dan data test
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # 5. Preprocessing Data, Check Stationarity
    st.subheader('Preprocessing Data')
    st.write("Stationarity Check on Data Train")
    is_stationary, p_value, adf_result = check_stationarity(train['Views'])
    st.write(f"p-value: {p_value}")
    st.write(f"Data Train Stasioner: {is_stationary}")

    # 6. Lakukan differencing jika data train tidak stasioner
    if not is_stationary:
        st.write("Data Train tidak stasioner. Melakukan differencing...")
        train_diff, n_diff = difference_until_stationary(train['Views'])
        st.write(f"Jumlah differencing yang dilakukan: {n_diff}")
    else:
        st.write("Data Train sudah stasioner. Tidak perlu differencing.")

    # 7. Menampilkan visualisasi data train dan data test
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(x=train.index, y=train['Views'], mode='lines', name='Data Train'))
    fig_split.add_trace(go.Scatter(x=test.index, y=test['Views'], mode='lines', name='Data Test'))
    fig_split.update_layout(title='Visualisasi Data Train dan Test', xaxis_title='Date', yaxis_title='Views')
    st.plotly_chart(fig_split)
    st.write(f"Total baris data train: {len(train)}")
    st.write(f"Total baris data test: {len(test)}")
    
    # 8. Grid Search ARIMA Hyperparameter Tuning
    st.subheader("Hyperparameter Tuning with Grid Search")
    st.write("Select the range of p, d, and q values for Grid Search.")
    
    p_values = st.multiselect("Pilih nilai p", options=list(range(0, 6)))
    d_values = st.multiselect("Pilih nilai d", options=list(range(0, 4)))
    q_values = st.multiselect("Pilih nilai q", options=list(range(0, 6)))

    # 9. Validasi input untuk memastikan bahwa p, d, dan q tidak kosong
    if st.button("Cari Parameter Terbaik"):
        if not p_values or not d_values or not q_values:
            st.error("Nilai p, d, dan q tidak boleh kosong! Harap pilih setidaknya satu nilai untuk setiap parameter.")
        else:
            st.subheader("Proses Pencarian Parameter Terbaik")
            with st.spinner('Melakukan Grid Search...'):
                best_cfg = grid_search_arima(train['Views'], test['Views'], p_values, d_values, q_values)

    # 10. Membangun model ARIMA dengan parameter terbaik
    # st.subheader("ARIMA Model - Best Parameters")
    # Pastikan grid search telah dijalankan dengan mengecek apakah best_cfg telah terdefinisi
    if best_cfg:
        model = ARIMA(train['Views'], order=best_cfg)
        model_fit = model.fit()

        # 11. Melakukan prediksi pada data test
        try:
            forecast = model_fit.forecast(steps=len(test))
            test['Forecast'] = forecast
        except Exception as e:
            st.write("Error saat melakukan prediksi:", e)


        # 12. Menampilkan hasil peramalan dalam tabel
        st.subheader('Forecast Results')
        st.write("Hasil peramalan data test dalam bentuk tabel:")
        test['Perbedaan Nilai'] = test['Views'] - test['Forecast']
        st.write(test[['Views', 'Forecast', 'Perbedaan Nilai']])

        # 13. Menampilkan rata - rata forecast
        average_value = test['Views'].mean()
        average_value_difference = test['Perbedaan Nilai'].mean()
        average_forecast = (test['Forecast'].mean())
        st.write("Average Actual Views:", average_value)
        st.write("Average Forecast Views:", average_forecast)
        st.write("Average Views Difference:", average_value_difference)

        # 14. Visualisasi hasil peramalan
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=train.index, y=train['Views'], mode='lines', name='Data Train'))
        fig_forecast.add_trace(go.Scatter(x=test.index, y=test['Views'], mode='lines', name='Data Test'))
        fig_forecast.add_trace(go.Scatter(x=test.index, y=test['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))
        fig_forecast.update_layout(title='Visualisasi Forecast vs Actual', xaxis_title='Date', yaxis_title='Views')
        st.plotly_chart(fig_forecast)

        # 15. Menghitung metrik evaluasi MAE, MAPE, dan RMSE
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

        # # 16. Forecasting
        # steps = st.number_input("Enter the number of steps for forecasting", min_value=1, max_value=30, value=6, key='steps')
        # future_forecast = model_fit.forecast(len(test) + steps)[-steps:]
        # st.subheader(f"Forecast for next {steps} Months")

        # # 17. Membuat dataframe untuk hasil peramalan masa depan
        # df_future = pd.DataFrame({
        #     'Tanggal': pd.date_range(df.index[-1], periods=steps + 1, freq='MS')[1:],
        #     'Future Forecast': future_forecast
        # })
        # st.write(df_future.reset_index(drop=True))

        # # 18. Visualisasi hasil peramalan masa depan
        # fig = go.Figure()
        # # Menambahkan data actual
        # fig.add_trace(go.Scatter(x=df.index, y=df['Views'], mode='lines', name='Actual'))
        # # Menambahkan data forecast ARIMA pada data test
        # fig.add_trace(go.Scatter(x=test.index, y=test['Forecast'], mode='lines', name='ARIMA Forecast', line=dict(color='red')))
        # fig.add_trace(go.Scatter(x=df_future.index, y=df_future['Future Forecast'], mode='lines', name='Future Manual Forecast', line=dict(color='green')))
        # # Mengatur layout
        # fig.update_layout(
        # title='Plot Forecast Masa Depan',
        # xaxis_title='Date',
        # yaxis_title='Views',
        # width=800,  # Lebar grafik
        # height=600, # Tinggi grafik
        # template='plotly_dark'  # Tema plot
        # )
        # # Menampilkan plot
        # st.plotly_chart(fig)
