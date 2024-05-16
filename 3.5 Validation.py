# Test Model on Validation data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt


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

def train_test_split(data, test_size):
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

np.random.seed(19)
tf.random.set_seed(19)

# Import the input data
data = pd.read_csv('Out_of_sample_c.csv')

# Extract features and target variable
x = data[['Monte Carlo Price', 'Trinomial Tree Price', 'Finite Difference Price']]
y = data['Market Price']

# split the data into training and testing
X_train, X_val = train_test_split(x, 0.2)
Y_train, Y_val = train_test_split(y, 0.2)

# Initialize MinMaxScaler objects for input and output
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

x_train, x_val, y_train, y_val = scale_data(X_train, X_val, Y_train, Y_val, input_scaler, output_scaler)

# Load the model
json_file = open('Models/nnarchitecture_c.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights("Models/nnweights_c.h5")
print("Model loaded")

loaded_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])

#Evaluate the model on the training data
train_metrics = loaded_model.evaluate(x_train, y_train, verbose=0)
# Evaluate the model on the validation data
val_metrics = loaded_model.evaluate(x_val, y_val, verbose=0)

print("Training Data - MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}".format(train_metrics[1], train_metrics[2], train_metrics[3]))
print("Validation Data - MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}".format(val_metrics[1], val_metrics[2], val_metrics[3]))

# Plotting MSE values as a bar chart
plt.bar(['Train MSE', 'Validation MSE'], [train_metrics[1], val_metrics[1]])
plt.xlabel('Dataset')
plt.ylabel('MSE')
plt.title('Train vs Validation MSE')
plt.show()

# Scatter plot of MSE values
labels = ['Train', 'Validation']
values = [train_metrics[1], val_metrics[1]]
plt.figure(figsize=(6, 4))
plt.scatter(range(len(labels)), values, color='blue')
plt.xticks(range(len(labels)), labels)
plt.title('Train vs Validation MSE')
plt.ylabel('MSE')
plt.show()
