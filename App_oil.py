#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model
model_gru = load_model('trained_model.h5')

# Load the scaler using pickle
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title('Crude Oil Price Forecasting')

# Input for the last 10 days' prices
st.write("87.40,87.93,87.44,89.31,88.00,18.63,18.60,18.55,18.45,18.63")
input_prices = []

for i in range(10):
    day_price = st.number_input(f'Day {i+1} Price', value=18.55)
    input_prices.append(day_price)

# Scale the input using the loaded scaler
scaled_input = scaler.transform(np.array(input_prices).reshape(-1, 1))

# Reshape the input for the model
input_data = scaled_input.reshape(1, 10, 1)  # Adjust the input shape according to your model

# Make predictions
predicted_scaled_price = model_gru.predict(input_data)
predicted_price = scaler.inverse_transform(predicted_scaled_price)

st.write("Predicted Crude Oil Price for the Next Time Step:")
st.write(predicted_price[0][0])


# In[ ]:




