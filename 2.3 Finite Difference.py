import numpy as np

def fd_option_pricing(option_type, S0, K, r, T, sigma, Smax, M, N):
    T = T * 365
    dt = T / N
    ds = Smax / M
    print('dt:', dt, 'ds:', ds)
    S = np.arange(M + 1) * ds
    t = np.arange(N + 1) * dt

    alpha = 0.5 * sigma**2 * dt / ds**2
    beta = 0.5 * r * dt / ds
    gamma = r * dt
    
    # Initialize terminal condition
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
    return round(C_fd, 3)

# Example usage
T = 0.049
S0 = 30095
K = 25000
sigma = 0.649
r = 0.0014 
S_max = K*2
N = 200
M = 200

option_price = fd_option_pricing('C',S0, K, r, T, sigma, S_max, M, N)
print(option_price)

# Plot the option price
# plt.plot(S0, option_price[:, 0])
# plt.xlabel('Stock Price')
# plt.ylabel('Option Price')
# plt.title('Option Price at t=0')
# plt.show()

# # Plotting the results
# dt = T / N
# time_steps = np.arange(N + 1) * dt
# S, T = np.meshgrid(S, time_steps)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(S, T, option_price_c.T, cmap='viridis')
# ax.set_xlabel('Asset Price (S)')
# ax.set_ylabel('Time to Expiration (T)')
# ax.set_zlabel('Option Value')
# ax.set_title('European Call Option Pricing using Explicit FDM')
# plt.show()