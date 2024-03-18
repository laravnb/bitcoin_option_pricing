import numpy as np

# Trinomial Tree Method for Option pricing
def tt_option_pricing (S,T,r,sigma,K, periods, option_type):
    dt = T/periods
    C_tt = np.zeros((2*periods+1))
    cp = 0      # variable to differentiate between call or put option

    # option type
    if option_type == 'C': cp = 1
    else: cp = -1

    # jump sizes in tree
    u = np.exp(sigma*np.sqrt(2*dt))
    d = 1/u
    m = 1

    # Transition probabilities
    pu = ((np.exp(r*dt/2)-np.exp(-sigma*np.sqrt(dt/2)))**2)/(np.exp(sigma*np.sqrt(dt/2))-np.exp(-sigma*np.sqrt(dt/2)))**2
    pd = ((np.exp(sigma*np.sqrt(dt/2))-np.exp(r*dt/2))**2)/(np.exp(sigma*np.sqrt(dt/2))-np.exp(-sigma*np.sqrt(dt/2)))**2
    pm = 1-pu-pd

    # Initialize the stock and option price arrays
    St = np.zeros((2*periods+1, periods+1))
    C_tt = np.zeros((2*periods+1, periods+1))

    # Calculate the stock price at each node at maturity
    # Calculate Option prices at each node at maturity
    # Interating through the last collumn of nodes in the tree
    # okay, u and d factors related --> implicitly uses d as well since u = 1/d
    for j in range(0, 2*periods+1):
        St[j, periods] = S*np.power(u, periods-j)
        C_tt[j, periods] = max(0, cp*(St[j, periods]-K))

    #Calculate Option prices at each node backwards
    disc = np.exp(-r * dt)

    for i in range(periods-1, -1, -1):
        for j in range(0, 2*i+1):
            St[j, i] = St[j, i+1]*u
            # dicounting the option price at each node back to present value
            # Calculate the option price using the transition probabilities
            C_tt[j, i] = disc *(pu*C_tt[j, i+1]+pm*C_tt[j+1, i+1]+pd*C_tt[j+2, i+1])

    C_tt = round(C_tt, 3)
        
    return C_tt[0, 0]

# Example
S, K, r, T, sigma, periods, option_type = 30095, 25000, 0.0014, 0.049, 0.649, 1000, 'P'
print (tt_option_pricing(S,T,r,sigma,K, periods, option_type))

    
























