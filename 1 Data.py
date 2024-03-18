import pandas as pd
import numpy as np
import glob
from csv import writer, reader
from datetime import datetime

# download bitcoin options data from deribit
# Specify directory path containing the CSV files, * for variations
files_path = 'path/to/*-BTC-*MAY22-export.csv'
files = glob.glob(files_path) # glob to get list of matching files
combined_df = pd.DataFrame()

# Loop through each input file
for file in files:
    df1 = pd.read_csv(file)
    # Extract the date from the file name
    date_str = file.split('-')[2]
    date_from_file = datetime.strptime(date_str, '%d%b%y').strftime('%Y-%m-%d')
    # Create a new column with the date
    df1['Date'] = date_from_file

    # download bitcoin historical volatility from deribit or calculate it with their code
    # Read the historical value from another document based on the date
    vol_df = pd.read_csv('historical_volatility.csv')
    merged_df = pd.merge(df1, vol_df, on='Date', how='left')
    combined_df = combined_df.append(merged_df, ignore_index=True)

# clearing all unnecessary columns (and removing NaN values)
df2 = combined_df.drop(['Last', 'Size', 'IV (Bid)', 'Mark', 'IV (Ask)','Close'], axis = 1)
df2 = df2.drop(['Size.1', 'Open', 'Δ|Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Volume', ], axis = 1)
# dropt all rows if there is a '-' value
df2 = df2.replace('-', np.nan)
df2 = df2.dropna()

# Read the combined data back in to add another column
K = []
exercise_prices = []
expirations = []
option_types = []

for i, row in df2.iterrows():
    # extract the exercise price from the first collumn ,first collumn, integer after the second '-'
    K = int(row['Instrument'].split('-')[2])  
    exercise_prices.append(K)

    option_type = row['Instrument'].split('-')[3]
    option_types.append(option_type)

    # convert the 'Expiration' collumn to 3 decimal places
    # and then convert it to years --> necessary for option pricing methods
    expiration = round(row['Expiration']/ 365, 3)
    expirations.append(expiration)

df2['Exercise Price'], df2['Option Type'], df2['Expiration'] = exercise_prices, option_types, expirations
 
#rearrainging the columns such that the date is first
df2 = df2[['Date', 'Instrument', 'Option Type', 'Bitcoin Price', 'Expiration', 'Interest Rate', 'Exercise Price', 'Volatility', 'Ask', 'Bid']]
# df2 = df2.sort_values(by='Date', ascending=True)

#Turn the collumns into a series list
df2['Bitcoin Price'] = pd.to_numeric(df2['Bitcoin Price'].str.replace(',', ''))
df2['Expiration'] = pd.to_numeric(df2['Expiration']) 
df2['Interest Rate'] = pd.to_numeric(df2['Interest Rate'].str.replace(',', '.'))
df2['Exercise Price'] = pd.to_numeric(df2['Exercise Price'])
df2['Volatility'] = pd.to_numeric(df2['Volatility']) 
df2['Ask'] = pd.to_numeric(df2['Ask']) 
df2['Bid'] = pd.to_numeric(df2['Bid']) 

# dataset restrictions
# Only keep in the money options and drop the other rows
df2 = df2[(df2['Bitcoin Price'] > df2['Exercise Price']) & (df2['Option Type'] == 'C') | (df2['Bitcoin Price'] < df2['Exercise Price']) & (df2['Option Type'] == 'P')]
df2['Spread_Percentage'] = ((df2['Ask'] - df2['Bid']) / df2['Ask']) * 100
# only keep options with a spread lower than 50%
df2 = df2[df2['Spread_Percentage'] < 50]
df2 = df2.drop(['Spread_Percentage'], axis=1)
# only keep options with expiration days between 5 and 20
df2 = df2[(df2['Expiration'] >= 5/365) & (df2['Expiration'] <= 20/365)]
df = df2.drop([ 'Bid', 'Ask'], axis=1)

# Save the dataset for further use
df.to_csv('final_data.csv', index=False)
