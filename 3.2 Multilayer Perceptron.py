import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import MeanAbsolutePercentageError
import matplotlib.pyplot as plt

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

def invert_scaling(data, scaler):
    if scaler:
        inverted_data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    return inverted_data

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
data = pd.read_csv('Input_p.csv')

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
x_train, x_test,y_train, y_test = scale_data(X_train, X_test, Y_train, Y_test, input_scaler, output_scaler)

#Multilayer perceptron model for regression
nn_model = Sequential() # initialise model
nn_model.add(Dense(2, input_dim=3, activation='sigmoid')) # Add 1 hidden layer, 2 neurons, sigmoid activation              
nn_model.add(Dense(1, activation='linear')) # Output layer with linear activation for regression
nn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', MeanAbsolutePercentageError()])               
# Train the model
history = nn_model.fit(x_train, y_train, epochs=150, batch_size=16, validation_data=(x_test, y_test))

# Evaluate the model
train_mse, train_mae, train_mape = nn_model.evaluate(x_train, y_train)
test_mse, test_mae, test_mape = nn_model.evaluate(x_test, y_test)

#print rounded values
train_mse = round(train_mse, 4)
train_mae = round(train_mae, 4)
train_mape = round(train_mape, 4)
test_mse = round(test_mse, 4)
test_mae = round(test_mae, 4)
test_mape = round(test_mape, 4)

print("MSE, MAE, MAPE on training data:", train_mse, train_mae, train_mape)
print("MSE, MAE, MAPE on test data:", test_mse, test_mae, test_mape)

#plot the training and test loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Train vs Test MSE of NN')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# predict the NN option price
Y_pred = nn_model.predict(x_test)

# Invert scaling for predicted output
Y_pred = invert_scaling(Y_pred, output_scaler)
# print(Y_pred)

# Convert y_test.index to a pandas Index object
index = pd.Index(Y_test.index)

# Create a pandas Series with Y_pred and the converted index
# Add the Series to the DataFrame
C_nn = pd.Series(Y_pred, index=index)
data['NN Price'] = C_nn

# Remove empty rows
data = data.dropna()

# Save the dataset to a csv file
data.to_csv('NN_p.csv', index=False)


# # Save the model with json
# model_json = nn_model.to_json()
# with open("Models/nnarchitecture_c.json", "w") as json_file:
#     json_file.write(model_json)

# # Save the weights
# nn_model.save_weights("Models/nnweights_c.h5")
# print("Model saved ")


# from keras.models import model_from_json
# # Load the model
# json_file = open('Models/nn_architecture_c.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # Load weights into new model
# loaded_model.load_weights("Models/nn_weights_c.h5")
# print("Model loaded")
