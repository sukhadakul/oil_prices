#!/usr/bin/env python
# coding: utf-8

# ### Importing the library's

# In[55]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import lag_plot
import statsmodels.api as sm
from scipy import stats
import requests
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense  


# ### Getting the data from the website named :alphavantage

# In[56]:


url = 'https://www.alphavantage.co/query?function=BRENT&interval=daily&apikey=G47FZC6JH4IKPIFO'
r = requests.get(url)
data = r.json() 


# In[57]:


df = pd.DataFrame(data)
df


# ### Extraxting only the data column from hole dataset

# In[62]:


df2 = df[['data']] 


# In[63]:


df2.info()


# ### Modifiying the data into suitable form

# In[64]:


# Function to extract the values from the 'data' column
def extract_values(row):
    value = row['value'].replace(',', '.')
    try:
        return row['date'], float(value)
    except ValueError:
        return row['date'], None  # Return None for invalid values

# Applying the function to the 'data' column with apply and zip
df2['date'], df2['value'] = zip(*df2['data'].apply(extract_values))

# Dropping the original 'data' column
df2.drop(columns=['data'], inplace=True)

# Printing the resulting DataFrame
print(df2)


# In[65]:


df2.sample(10)


# ### Converting the date column into date-time format and using it as index.

# In[66]:


df2["date"] = pd.to_datetime(df2.date)


# In[67]:


df2.head()


# In[68]:


df2.info()


# In[69]:


df2.shape


# ### Checking the Null values. There are 258 NA values.Which is 2.73% of data

# In[70]:


df2.isna().sum()/9439*100


# ### Filling the NA values

# In[71]:


df2['value']=df2['value'].fillna(method ='bfill')


# In[72]:


print(df2.isna().sum())
print(df2.shape)


# ### Checking for duplicated values.

# In[73]:


df2.duplicated().sum()


# In[74]:


df2.describe()


# In[75]:


df2 = df2.set_index('date')


# - #### The average price of the crude oil in it's entire history is 49 dollar
# - #### The range is 9 dollar to 143.95 dollar with the standard deviation of 32.84 dollar.

# In[76]:


plt.figure(figsize=(12, 6))
plt.plot(df2.index, df2['value'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid(True)


# ### There are some sharp drops - 
# 1. In 2008 Because of recession 
# 2. In 2016 because of some investors funding issue
# 3. In 2020 bescause of Covid-19.

# ### Creating and plotting the 200 day's Moving Average to know the trend of data

# In[77]:


ma200 = df2.value.rolling(200).mean()
ma200


# In[78]:


plt.figure(figsize=(12,6))
plt.plot(df2.value)
plt.plot(ma200,'r')
plt.grid(True)


# ### In its earlier stage it follows the up-trend but after 2008 it becomes volatile.

# In[79]:


plt.figure(figsize=(10, 6))
sns.histplot(df2['value'], bins=30, kde=True)
plt.xlabel('Crude Oil Price')
plt.ylabel('Frequency')
plt.title('Distribution of Crude Oil Price')
plt.show()


# In[80]:


sns.boxplot(df2['value'])


# In[81]:


df2['Year'] = pd.DatetimeIndex(df2.index).year
df2['Month'] = pd.DatetimeIndex(df2.index).month


# In[82]:


df2.head()


# In[83]:


plt.figure(figsize=(14, 8))
plt.subplot(211)
sns.boxplot(x=df2.index.year, y=df2['value'], data=df2)
plt.xticks(rotation=90)
plt.show()


# In[84]:


Q1 = df2['value'].quantile(0.20)
Q3 = df2['value'].quantile(0.80)
IQR = Q3 - Q1
whisker_width = 1.5
lower_whisker = Q1 -(whisker_width*IQR)
upper_whisker = Q3 + (whisker_width*IQR)
df2['value']=np.where(df2['value']>upper_whisker,upper_whisker,np.where(df2['value']<lower_whisker,lower_whisker,df2['value']))


# ### ACF/Autocorrelation plot

# In[85]:


# Autocorrelation Funcation(ACF)
plot_acf(df2.value,lags = 30)
plt.show()


# ### Lag Plot

# In[86]:


lag_plot(df2)


# In[87]:


df2['log_Price']=np.log(df2['value'])


# In[88]:


df2


# In[89]:


t = np.arange(1,9455)
df2['t']=t 


# In[90]:


t_squared=np.array(t*t)
df2['t_squared']=t_squared


# In[91]:


df2.head()


# In[92]:


df2.sort_index(ascending=True, inplace=True)


# In[93]:


#Train Test Split in the ratio of 80:20
train=df2.head(7560)
test=df2.tail(1889)


# In[94]:


train


# In[95]:


# Linear Model
Linear_model=smf.ols('value~t', data=train).fit( )    #y~x
pred_linear=pd.Series(Linear_model.predict(test['t']))   #predicting for test
rmse_linear = np.sqrt(np.mean((np.array(test['value'])-np.array(pred_linear))**2))    #sqrt of mean(y act - y pred)^2
rmse_linear

results = pd.DataFrame({'Method':['Linear method'], 'RMSE': [rmse_linear]})
results = results[['Method', 'RMSE']]
results


# In[96]:


#Exponential Model similiar to Linear but insted of Yt we use log(Yt)
#SAME LIKE LINEAR BUT IN PLACE OF Yt WE USE LOG(Yt)
Exp=smf.ols('log_Price~t', data=train).fit( )                                   #log(y)~x
pred_Exp=pd.Series(Exp.predict(test['t']))   #predicting for test
rmse_Exp = np.sqrt(np.mean((np.array(test['value'])-np.array(np.exp(pred_Exp)))**2))    #sqrt of mean(y act - y pred)^2
rmse_Exp
tempResults = pd.DataFrame({'Method':['Exponential Model'], 'RMSE': [rmse_Exp] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE']]
results


# In[97]:


# Quadratic here we are creating two independent variables

Quad=smf.ols('value~t+t_squared', data=train).fit( )

pred_Quad=pd.Series(Quad.predict(test[['t', 't_squared']]))    #predicting for test
rmse_Quad = np.sqrt(np.mean((np.array(test['value'])-np.array(pred_Quad))**2))    #sqrt of mean(y act - y pred)^2
rmse_Quad

tempResults = pd.DataFrame({'Method':['Quadratic Model'], 'RMSE': [rmse_Exp] })
results = pd.concat([results,tempResults])
results = results[['Method', 'RMSE']]
results


# In[98]:


train_len = 7560
y_hat_naive = test.copy()
y_hat_naive['naive_forecast'] = train['value'][train_len-1]


# In[99]:


#Naive Model
rmse = np.sqrt(mean_squared_error(test['value'], y_hat_naive['naive_forecast'])).round(2)
tempResults = pd.DataFrame({'Method':['Naive Model'], 'RMSE': [rmse] })
results = pd.concat([results,tempResults])
results = results[['Method', 'RMSE']]
results


# In[100]:


y_hat_sma = df2.copy()
ma_window = 365
y_hat_sma['sma_forecast'] = df2['value'].rolling(ma_window).mean()
y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len-1]


# In[109]:


# Double Exponential Smooting
from statsmodels.tsa.holtwinters import ExponentialSmoothing

mod_add12 = ExponentialSmoothing(df2['value'], trend='add')
fitted_model= mod_add12.fit()
df2['DES'] = fitted_model.fittedvalues.shift(-1)
df2.head()


# In[110]:


#Triple Exponential Smooting by adding
df2['TESA'] = ExponentialSmoothing(df2['value'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
df2.head()


# In[111]:


#Triple Exponential Smooting by multiplying
df2['TESM'] = ExponentialSmoothing(df2['value'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
df2.head()


# In[112]:


df2[['value','TESA','TESM']].plot(figsize=(12,6))


# In[113]:


from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df2.value,period=12)
decompose_ts_add.plot()
plt.show()


# **ARIMA and SARIMA Model**

# In[114]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import randint

# Define parameter ranges for random search
param_dist = {
    'p': randint(0, 3),
    'd': randint(0, 2),
    'q': randint(0, 3)
}

# Number of random search iterations
n_iter = 10

# Split the data into train and test sets
train_size = int(len(df2) * 0.8)
train_data = df2[:train_size]
test_data = df2[train_size:]

best_mae = float('inf')
best_params = None

# Perform random search
for _ in range(n_iter):
    try:
        random_params = {param: dist.rvs() for param, dist in param_dist.items()}
        order = (random_params['p'], random_params['d'], random_params['q'])

        arima_model = ARIMA(train_data['value'], order=order)
        arima_results = arima_model.fit()

        arima_forecast = arima_results.forecast(steps=len(test_data))
        arima_mae = mean_absolute_error(test_data['value'], arima_forecast)

        if arima_mae < best_mae:
            best_mae = arima_mae
            best_params = order
    except:
        continue

print("Best ARIMA MAE:", best_mae)
print("Best ARIMA Parameters:", best_params)


# In[115]:


# You need to specify appropriate values for p, d, and q
order = best_params


# In[116]:


# Initialize and fit the ARIMA model
arima_model = ARIMA(train['value'], order=order)
arima_results = arima_model.fit()


# In[117]:


arima_forecast = arima_results.forecast(steps=len(test))


# In[118]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

arima_mae = mean_absolute_error(test['value'], arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test['value'], arima_forecast))


# In[119]:


tempResults = pd.DataFrame({'Method':['ARIMA'], 'RMSE': [arima_rmse] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE']]
results


# ## LSTM Model

# In[120]:


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df2[['value']])


# In[121]:


sequence_length = 10


# In[122]:


sequences = []
target = []
for i in range(sequence_length, len(scaled_data)):
    sequences.append(scaled_data[i - sequence_length:i, 0])
    target.append(scaled_data[i, 0])

X = np.array(sequences)
y = np.array(target)

# Split the data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[123]:


model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)


# In[124]:


# Predictions
lstm_predictions = model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)
# Evaluate the LSTM model
lstm_mae = mean_absolute_error(test['value'][-len(lstm_predictions):], lstm_predictions)
lstm_rmse = np.sqrt(mean_squared_error(test['value'][-len(lstm_predictions):], lstm_predictions))
print("LSTM MAE:", lstm_mae)
print("LSTM RMSE:", lstm_rmse)


# In[125]:


tempResults = pd.DataFrame({'Method':['LSTM'], 'RMSE': [lstm_rmse] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE']]
results


# ## GRU

# In[126]:


from tensorflow.keras.layers import GRU


# In[127]:


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df2[['value']])

sequence_length = 10
sequences = []
target = []
for i in range(sequence_length, len(scaled_data)):
    sequences.append(scaled_data[i - sequence_length:i, 0])
    target.append(scaled_data[i, 0])

X = np.array(sequences)
y = np.array(target)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[128]:


model_gru = Sequential()
model_gru.add(GRU(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
model_gru.add(Dense(units=1))
model_gru.compile(optimizer='adam', loss='mean_squared_error')
model_gru.fit(X_train, y_train, epochs=50, batch_size=32)


# In[129]:


#predictions
gru_predictions = model_gru.predict(X_test)
gru_predictions = scaler.inverse_transform(gru_predictions)

gru_mae = mean_absolute_error(test['value'][-len(gru_predictions):], gru_predictions)
gru_rmse = np.sqrt(mean_squared_error(test['value'][-len(gru_predictions):], gru_predictions))

print("GRU MAE:", gru_mae)
print("GRU RMSE:", gru_rmse)


# In[130]:


tempResults = pd.DataFrame({'Method':['GRU'], 'RMSE': [gru_rmse] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE']]
results


# In[131]:


# Calculate Naive Forecast
data['Naive_Forecast'] = df2['value'].shift(1)
# Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error
naive_mae = mean_absolute_error(df2['value'].iloc[1:], data['Naive_Forecast'].iloc[1:])
naive_rmse = np.sqrt(mean_squared_error(df2['value'].iloc[1:], data['Naive_Forecast'].iloc[1:]))

print("Naive Forecast MAE:", naive_mae)
print("Naive Forecast RMSE:", naive_rmse)



# In[132]:


tempResults = pd.DataFrame({'Method':['Naive'], 'RMSE': [naive_rmse] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE']]
results


# ### Saving the `GRU` Model

# In[133]:


import pickle


# In[134]:


# Save the entire model to an .h5 file
model_gru.save('trained_model.h5')


# In[135]:


# Save the scaler using pickle
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)


# In[136]:


# Save only the model weights
model_gru.save_weights('gru_model_weights.h5')


# In[137]:


df2. to_csv('dataset.csv', index=True)


# In[ ]:




