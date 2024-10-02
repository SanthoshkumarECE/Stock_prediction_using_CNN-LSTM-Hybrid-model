
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Conv1D, MaxPooling1D
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
print('shape is = {}'.format(dataset_train.shape))
print(dataset_train.head())

training_set = dataset_train.iloc[:, 1:2].values
print('shape is ={}'.format(training_set.shape))
print(training_set[0:5])
plt.plot(training_set, color='red', label='Google Stock Price in Test set')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled[0:5])
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print('X_train shape = {}'.format(X_train.shape))
print('y_train shape = {}'.format(y_train.shape))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape
regressor = Sequential()
# Convolutional Layer
regressor.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
# Max Pooling Layer (Optional)
regressor.add(MaxPooling1D(pool_size=2))
# LSTMs with Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# Output Layer
regressor.add(Dense(units=1))

#Compiling the model
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fitting the model
regressor.fit(X_train, y_train, epochs=150, batch_size=32)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat(
    (dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60: i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue',
         label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
#finding rmse value for accuracy since the prediction is continuous
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print("root mean squared error : {}.".format(rmse))

