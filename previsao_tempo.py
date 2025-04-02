import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("DailyDelhiClimateTrain.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df[['meantemp']])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30  
X, y = create_sequences(df_scaled, SEQ_LENGTH)
X = np.reshape(X, (X.shape[0], X.shape[1], 1)) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, label="Temperatura Real", color='blue')
plt.plot(y_pred, label="Temperatura Prevista", color='red', linestyle='dashed')
plt.legend()
plt.xlabel("Dias")
plt.ylabel("Temperatura Normalizada")
plt.title("Previsão de Temperatura usando LSTM")
plt.show()
