import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import MeanAbsolutePercentageError

# Function to scale inputs and outputs
def scale_data(x_train, x_test, y_train, y_test, input_scaler=None, output_scaler=None):
    if input_scaler:
        x_train = input_scaler.fit_transform(x_train)
        x_test = input_scaler.transform(x_test)
    if output_scaler:        
         # reshape 1d arrays to 2d arrays
        y_train = y_train.values.reshape(len(y_train), 1)
        y_test = y_test.values.reshape(len(y_test), 1)
        # fit scaler on training dataset
        # transform training dataset
        y_train = output_scaler.fit_transform(y_train)
        # transform test dataset
        y_test = output_scaler.transform(y_test)

    return x_train, x_test, y_train, y_test

# Function to evaluate the neural network
def evaluate_nn (x_train, x_test, y_train, y_test):
    nn_model = Sequential()
    nn_model.add(Dense(2, input_dim=3, activation='sigmoid'))
    nn_model.add(Dense(1, activation='linear'))
    nn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', MeanAbsolutePercentageError()])
    nn_model.fit(x_train, y_train, epochs=150, batch_size=16, validation_data=(x_test, y_test))   
    train_mse, train_mae, train_mape = nn_model.evaluate(x_train, y_train)
    test_mse, test_mae, test_mape  = nn_model.evaluate(x_test, y_test)
    return train_mse, train_mae, train_mape, test_mse, test_mae, test_mape

# Train test split for sequential data
def train_test_split(data, test_size):
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# setting the seed for reproducibility of weights 
np.random.seed(19)
tf.random.set_seed(19)

# Import the input data
data = pd.read_csv('Input_c.csv')

# Extract features and target variable
x = data[['Monte Carlo Price', 'Trinomial Tree Price', 'Finite Difference Price']]
y = data['Market Price']

# split the data into training and testing
X_train, X_test = train_test_split(x, 0.2)
Y_train, Y_test = train_test_split(y, 0.2)

# Initialize MinMaxScaler objects for input and output
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

# Scale data
x_train, x_test, y_train, y_test = scale_data(X_train, X_test, Y_train, Y_test, input_scaler, output_scaler)

results = []

train1, train2, train3 , test1, test2, test3 = evaluate_nn(x_train, x_test, y_train, y_test)
# print("MSE, MAE, MAPE on training data:", train1, train2, train3)
# print("MSE, MAE, MAPE on test data:", test1, test2, test3)

# repeat 30 times of evaluation and create a csv file with results
for i in range(30):
    train1, train2, train3 , test1, test2, test3 = evaluate_nn(x_train, x_test, y_train, y_test)
    results.append([train1, test1, train2, test2, train3, test3])

results = pd.DataFrame(results, columns=['Train MSE', 'Test MSE', 'Train MAE', 'Test MAE', 'Train MAPE', 'Test MAPE'])
results.to_csv('evaluate_c.csv', index=False)
