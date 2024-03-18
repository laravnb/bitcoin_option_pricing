import numpy as np
import pandas as pd

# Define volatility function
def calculate_historical_vols(df, sessions_in_year):
    # calculate first log returns using the open
    log_returns = []
    log_returns.append(np.log(df.loc[0, 'Close'] / df.loc[0, 'Open']))
    # calculate all but first log returns using close to close
    for index in range(len(df) - 1):
        log_returns.append(np.log(df.loc[index + 1, 'Close'] / df.loc[index, 'Close']))
    df = df.assign(log_returns=log_returns)

    # calculate the 15-day standard deviation and vol
    if len(df) > 14:
        sd_15_day = [np.nan] * 14
        Volatility = [np.nan] * 14
        for index in range(len(df) - 14):
            sd = np.std(df.loc[index:index + 6, 'log_returns'], ddof=1)
            sd_15_day.append(sd)
            Volatility.append(sd * np.sqrt(sessions_in_year))
        df = df.assign(sd_15_day=sd_15_day)
        df = df.assign(Volatility=Volatility)
    return df

# import data as df from the file 'BTC-USD.csv'
df = pd.read_csv('Data/BTC-USD.csv')

# Call the function to calculate historical vols
df = calculate_historical_vols(df, sessions_in_year=365)
# Round the values to 4 decimal places
df = df.round(3)
df = df[df['Date'] >= '2022-05-06']
# Drop all other collumns
df = df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'log_returns', 'sd_15_day'], axis=1)
df.to_csv('historical_volatility.csv', index=False)  # Save to a new file
