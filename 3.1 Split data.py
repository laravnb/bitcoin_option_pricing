# Split data into 6 weeks for training/ testing and 2 weeks for validation
import pandas as pd


data = pd.read_csv("Input_ML_c.csv")

data['Date'] = pd.to_datetime(data['Date'])
specific_date = pd.to_datetime('2022-06-17')  # Change this date to your specific date

# Split the dataset into two based on the specific date
input = data[data['Date'] <= specific_date]
out_of_sample = data[data['Date'] > specific_date]

input.to_csv('Input_c.csv', index=False)
out_of_sample.to_csv('Out_of_sample_p.csv', index=False)
