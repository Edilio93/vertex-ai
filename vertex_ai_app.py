
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("🤖 Vertex AI - Predicción de Mercado")
st.subheader("¿Subirá o bajará el precio mañana?")

# Selección de activo
option = st.selectbox("Selecciona el activo", ["Tesla (TSLA)", "Bitcoin (BTC-USD)"])
ticker = "TSLA" if option == "Tesla (TSLA)" else "BTC-USD"

# Descargar datos
data = yf.download(ticker, period="90d", interval="1d")
df = data[["Close"]].dropna()

# Escalar datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Preparar datos para LSTM
def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Crear y entrenar modelo
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(time_step, 1)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X, y, epochs=10, batch_size=1, verbose=0)

# Predicción del siguiente día
last_sequence = scaled_data[-time_step:]
input_seq = last_sequence.reshape(1, time_step, 1)
predicted_scaled = model.predict(input_seq)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

# Precio actual
current_price = df["Close"].iloc[-1]

# Mostrar resultado
st.metric(label="Precio actual", value=f"${current_price:.2f}")
st.metric(label="Predicción para mañana", value=f"${predicted_price:.2f}")

if predicted_price > current_price:
    st.success("🔼 Vertex predice que el precio SUBIRÁ")
else:
    st.error("🔽 Vertex predice que el precio BAJARÁ")
