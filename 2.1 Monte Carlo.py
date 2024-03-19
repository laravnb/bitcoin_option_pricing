import numpy as np

# Using Monte Carlo Simulations for Option Pricing using Brownian motion
N = 10000  #simulations
n = 365  # Number of steps or intervals in the simulation 
mc_prices = []

# Call options
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
# Put options
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

# Example
S, T, r, sigma, n, N, K = 35468, 0.041, 0.0014, 0.604, 365, 10000, 25000
print(c_mc_options_pricing(S, K, r, T, sigma, N, n))
