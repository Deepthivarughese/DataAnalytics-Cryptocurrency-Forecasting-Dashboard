# Install Dependencies:
# pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels prophet tensorflow streamlit pmdarima

import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from prophet import Prophet

# Streamlit Page Setup
st.set_page_config(page_title="Crypto Forecaster", layout="wide")
st.title("📈 Cryptocurrency Forecasting Dashboard")
st.write("Upload your dataset and generate predictions using LSTM, ARIMA, SARIMA, and Prophet.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your crypto CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\ufeff", "")

    # Sidebar Select Inputs
    symbols = df["symbol"].unique()
    selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", symbols)

    time_step = st.sidebar.slider("LSTM Time Steps", 30, 200, 60)
    epochs = st.sidebar.slider("LSTM Epochs", 5, 100, 20)
    forecast_days = st.sidebar.slider("Forecast Future Days", 5, 60, 30)

    # Filter selected symbol
    df = df[df["symbol"] == selected_symbol].copy()
    df.dropna(inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df.set_index("Date", inplace=True)
    df = df[~df.index.duplicated()]

    # Scale data
    data = df[["Close"]]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Train-Test split
    train_size = int(len(scaled) * 0.8)
    train = scaled[:train_size]
    test = scaled[train_size:]

    # Create dataset function
    def create_dataset(dataset, steps):
        X, y = [], []
        for i in range(len(dataset) - steps):
            X.append(dataset[i:i + steps, 0])
            y.append(dataset[i + steps, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train, time_step)
    X_test, y_test = create_dataset(test, time_step)

    X_train = X_train.reshape(X_train.shape[0], time_step, 1)
    X_test = X_test.reshape(X_test.shape[0], time_step, 1)
    # LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    st.write("### 🔄 Training LSTM Model...")
    with st.spinner("Training..."):
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    # Predictions
    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    actual = scaler.inverse_transform(scaled)
    # Index alignment
    train_idx = df.index[time_step:len(train_pred) + time_step]
    test_idx = df.index[len(train_pred) + (time_step * 2):len(df)]

    # Plot LSTM
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, actual, label="Actual")
    ax.plot(train_idx, train_pred, label="LSTM Train")
    ax.plot(test_idx, test_pred, label="LSTM Test")
    ax.legend()
    st.pyplot(fig)

    # LSTM Future Forecast
    future_input = scaled[-time_step:].reshape(1, time_step, 1)
    future_values = []

    for _ in range(forecast_days):
        pred = model.predict(future_input)
        future_values.append(pred[0][0])
        future_input = np.append(future_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    lstm_forecast = scaler.inverse_transform(np.array(future_values).reshape(-1, 1))

    st.write("### 📅 LSTM Forecast")
    st.dataframe(pd.DataFrame({"Day": range(1, forecast_days + 1), "Price": lstm_forecast.flatten()}))

    # Plot forecast
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(lstm_forecast)
    ax2.set_title("LSTM Future Forecast")
    st.pyplot(fig2)


    # ARIMA & SARIMA
    st.write("## 🔮 ARIMA & SARIMA Models")

    arima_model = auto_arima(df["Close"], seasonal=False)
    sarima_model = auto_arima(df["Close"], seasonal=True, m=7)

    arima_forecast = arima_model.predict(forecast_days)
    sarima_forecast = sarima_model.predict(forecast_days)

    st.write("### ARIMA Forecast")
    st.dataframe(pd.DataFrame({"Day": range(1, forecast_days + 1), "Price": arima_forecast}))

    st.write("### SARIMA Forecast")
    st.dataframe(pd.DataFrame({"Day": range(1, forecast_days + 1), "Price": sarima_forecast}))

    # Prophet
    st.write("## ⏳ Prophet Model")

    prophet_df = df.reset_index()[["Date", "Close"]]
    prophet_df.columns = ["ds", "y"]

    prophet_model = Prophet()
    prophet_model.fit(prophet_df)

    future_df = prophet_model.make_future_dataframe(periods=forecast_days)
    forecast = prophet_model.predict(future_df)

    st.dataframe(forecast[["ds", "yhat"]].tail(forecast_days))

    fig3 = prophet_model.plot(forecast)
    
    st.pyplot (fig3)