from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from keras.metrics import MeanAbsolutePercentageError
import pandas as pd

# Import the input data
data = pd.read_csv('Input_c.csv')

# Extract features and target variable
x = data[['Monte Carlo Price', 'Trinomial Tree Price', 'Finite Difference Price']]
y = data['Market Price']

# split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


## HPO using grid search
# Load the model
json_file = open('Models/nn_architecture_c.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights("Models/nn_weights_c.h5")
print("Model loaded")

## HPO using grid search
# hyperparameters are the epochs and batch size

# Function to create model, --> reinitiate weights (required for KerasRegressor)
def create_nn():
    # create model
    nn = Sequential()
    nn.add(Dense(2, input_dim=3, activation='sigmoid'))
    nn.add(Dense(1, activation='linear'))
    # Compile model
    nn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', MeanAbsolutePercentageError()])
    return nn

# create KerasRegressor object to be used in grid search
kr = KerasRegressor(build_fn=create_nn, verbose=0)

# define hyperparameters
param_grid = {'batch_size': [16, 32, 64],
              'epochs': [50, 100, 150]}

# perform grid search
grid = GridSearchCV(estimator=kr, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
    # print("%f (%f) with: %r" % (mean, stdev, param))
    # batch size: 16, epochs: 150

# Save the best model
model_json = grid_result.best_estimator_.model.to_json()
with open("Models/nn_architecture_best.json", "w") as json_file:
    json_file.write(model_json)
# Save the weights
grid_result.best_estimator_.model.save_weights("Models/nn_weights_best.h5")
print("Best model saved ")

