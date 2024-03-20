import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read the input parameters from the file 'final_data.csv'
data = pd.read_csv('final_data.csv')

#Extract the input parameters for option pricing
S = data['Bitcoin Price'].values    
K = data['Exercise Price'].values
r = data['Interest Rate'].values
T = data['Expiration'].values
sigma = data['Volatility'].values
option_types = data['Option Type'].values

### Monte Carlo simulations using Brownian motion
N = 1000  #simulations
n = 365  # Number of steps or intervals in the simulation 
mc_prices = []

# Call options function
def c_mc_options_pricing (S, K, r, T, sigma, N, n):
    dt = T/n # Time step
    # Simulating N paths of the underlying asset
    S = S * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(N, n)), axis=1))
    # Calculating the payoff for each path
    # Differentiate between call and put options
    payoff = np.maximum(S[:, -1] - K, 0)
    # Calculating the price of the option and discounting it back to present value
    C_mc_c = np.mean(payoff) * np.exp(-r * T)
    return C_mc_c
# Put options function
def p_mc_options_pricing (S, K, r, T, sigma, N, n):
    dt = T/n # Time step
    # Simulating N paths of the underlying asset
    S = S * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(N, n)), axis=1))
    # Calculating the payoff for each path
    # Differentiate between call and put options
    payoff = np.maximum(K - S[:, -1], 0)
    # Calculating the price of the option and discounting it back to present value
    C_mc_p = np.mean(payoff) * np.exp(-r * T)
    return C_mc_p

# Calculate the Monte Carlo price for each option in the dataset
for i, row in data.iterrows():
    option_type = option_types[i]
    if option_type == 'C':
        mc_prices.append(c_mc_options_pricing(S[i], K[i], r[i], T[i], sigma[i], N, n))
    elif option_type == 'P':
        mc_prices.append(p_mc_options_pricing(S[i], K[i], r[i], T[i], sigma[i], N, n))    

# Add the Monte Carlo price to the dataset (and save it to a file)
data['Monte Carlo Price'] = mc_prices
data.to_csv('C_mc.csv', index=False)


### Calculate the Trinomial Tree price for each option in the dataset
periods = 100   # number of time steps
tt_prices = []

def tt_option_pricing (S,T,r,sigma,K, periods, option_type):
    dt = T/periods
    C_tt = np.zeros((2*periods+1))
    cp = 0      # variable to differentiate between call or put option

    # option type
    if option_type == 'C': cp = 1
    else: cp = -1

    # jump sizes
    u = np.exp(sigma*np.sqrt(2*dt))
    d = 1/u
    m = 1

    # Transition probabilities
    pu = ((np.exp(r*dt/2)-np.exp(-sigma*np.sqrt(dt/2)))**2)/(np.exp(sigma*np.sqrt(dt/2))-np.exp(-sigma*np.sqrt(dt/2)))**2
    pd = ((np.exp(sigma*np.sqrt(dt/2))-np.exp(r*dt/2))**2)/(np.exp(sigma*np.sqrt(dt/2))-np.exp(-sigma*np.sqrt(dt/2)))**2
    pm = 1-pu-pd

    St = np.zeros((2*periods+1, periods+1))
    C_tt = np.zeros((2*periods+1, periods+1))

    # Calculate Option prices at each node at maturity
    for j in range(0, 2*periods+1):
        St[j, periods] = S*np.power(u, periods-j)
        C_tt[j, periods] = max(0, cp*(St[j, periods]-K))

    #Calculate Option prices at each node backwards
    disc = np.exp(-r * dt)

    for i in range(periods-1, -1, -1):
        for j in range(0, 2*i+1):
            St[j, i] = St[j, i+1]*u
            C_tt[j, i] = disc *(pu*C_tt[j, i+1]+pm*C_tt[j+1, i+1]+pd*C_tt[j+2, i+1])

    return C_tt[0, 0]

for i, row in data.iterrows():
    S_max = K[i]*2 # Maximum stock price in grid
    tt_prices.append(tt_option_pricing(S[i], T[i], r[i], sigma[i], K[i], periods, option_types[i]))

# Add the Trinomial Tree price to the dataset and save it to a file
data['Trinomial Tree Price'] = tt_prices
data.to_csv('C_tt.csv', index=False)


### Calculate the option price using the finite difference method to solve the Black-Scholes PDE
def fd_option_pricing(option_type, S0, K, r, T, sigma, Smax):
    # grid size is 3T
    N = int(3 * T * 365)        # Number of time steps in grid
    M = int(3 * T * Smax)       # Number of asset price steps in grid
    dt = T / N
    ds = Smax / M
    S = np.arange(M + 1) * ds
    t = np.arange(N + 1) * dt

    alpha = 0.5 * sigma**2 * dt / ds**2
    beta = 0.5 * r * dt / ds
    gamma = r * dt
    
    # Initialize terminal condition for put and call options
    C = np.zeros((M + 1, N + 1))
    if option_type == 'C':
        C[:, -1] = np.maximum(S - K, 0)
    elif option_type == 'P':
        C[:, -1] = np.maximum(K - S, 0)

    # Apply explicit finite difference method
    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            C[i, j] = alpha * C[i + 1, j + 1] + (1 - 2 * alpha - beta + gamma) * C[i, j + 1] + alpha * C[i - 1, j + 1] + beta * C[i + 1, j + 1]

        # Apply boundary conditions
        if option_type == 'C':
            C[0, j] = 0
            C[-1, j] = Smax - K * np.exp(-r * (N - j) * dt)

        elif option_type == 'P':
            C[0, j] = K * np.exp(-r * (N - j) * dt)
            C[-1, j] = 0
    
    # Interpolate to get option price at S0
    C_fd = np.interp(S0, S, C[:, 0])
    return C_fd

# Calculate the finite difference price for each option in the dataset
fd_prices = []

for i, row in data.iterrows():
    fd_prices.append(fd_option_pricing(option_types[i], S[i], K[i], r[i], T[i], sigma[i], S_max))

# Add the finite difference price to the dataset and save it to a file
data['Finite Difference Price'] = fd_prices

#drop all columns except the different option prices
data = data.drop(columns=['Date', 'Instrument', 'Option Type','Bitcoin Price', 'Exercise Price', 'Interest Rate', 'Expiration', 'Volatility'])
data.to_csv('C_fd.csv', index=False)

