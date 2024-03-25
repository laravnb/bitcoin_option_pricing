import numpy as np

# Calculate option prices using the explicit finite difference method
def fd_option_pricing(S0, K, r, T, sigma, Smax, M, N, option_type):
    # Calculate time and asset step sizes
    dt = T / N
    ds = Smax / M
    
    # Generate arrays for asset and time indices
    S = np.arange(M)
    t = np.arange(N)
    
    # Initialize the grid and boundary conditions
    grid = np.zeros(shape=(M+1, N+1))
    boundary_conds = np.linspace(0, Smax, M+1)

    # Set boundary conditions depending on option type
    if option_type == 'P':
        grid[:, -1] = np.maximum(0, K - boundary_conds)
        grid[0, :-1] = (K - Smax) * np.exp(-r * dt * (N - t))
    else:
        grid[:, -1] = np.maximum(0, boundary_conds - K)
        grid[-1, :-1] = (Smax - K) * np.exp(-r * dt * (N - t))

    # Calculate coefficients for the finite difference method
    a = 0.5 * dt * ((sigma ** 2) * (S ** 2) - r * S)
    b = 1 - dt * ((sigma ** 2) * (S ** 2) + r)
    c = 0.5 * dt * ((sigma ** 2) * (S ** 2) + r * S)
    
    # Iterate through the grid points to compute option prices
    for j in reversed(t):
        for i in range(M)[2:]:
            grid[i, j] = a[i] * grid[i-1, j+1] + b[i] * grid[i, j+1] + c[i] * grid[i+1, j+1]
    
    # Interpolate to find option price at S0
    C_fd = np.interp(S0, boundary_conds, grid[:, 0])
    return C_fd

# Example
T, S0, K, sigma, r, S_max, N, M  = 0.049, 30095, 25000, 0.649, 0.0014, K*2, 500, 50

option_price = fd_option_pricing('C',S0, K, r, T, sigma, S_max, N, M)
print(option_price)
