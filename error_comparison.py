import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to calculate the L1 norm error
def calculate_errors(i, W, num_samples):
    A = A_list[i]
    A = A.T.flatten()
    D = D_list[i]
    ground_truth_frac = ground_truth_fracs[i]
    A_jacob = A.reshape((10,50))

    D_jacob = D.T
    E_jacob = E.T
    Sigma_jacob = Sigma.T

    teddy_errors = []
    jacob_errors = []

    for _ in tqdm(range(num_samples), desc=f"i = {i}, W = {W}"):
        x1 = grad_descent_known(C1, C2=0, C3=0, C4=0, C5=0, P=0, L=0, W=W, x1_size=x1_size, A=A, D=D, E=E, Sigma=Sigma, known=known)
        jacob_results = pytorch_cov_prediction(A_jacob, E_jacob, D_jacob, Sigma_jacob, W, 1, 1, num_epochs=num_epochs)
        
        x1 = x1.squeeze()
        teddy_alpha = x1 / sum(x1)
        jacob_alpha = jacob_results[0]

        ted_error = L1_norm(teddy_alpha, ground_truth_frac)
        jacob_error = L1_norm(jacob_alpha, ground_truth_frac)
        
        teddy_errors.append(ted_error)
        jacob_errors.append(jacob_error)
    
    return np.mean(teddy_errors), np.std(teddy_errors), np.mean(jacob_errors), np.std(jacob_errors)

# Plotting
for i in i_values:
    teddy_errors_means = []
    teddy_errors_stds = []
    jacob_errors_means = []
    jacob_errors_stds = []
    
    for W in W_values:
        mean_teddy, std_teddy, mean_jacob, std_jacob = calculate_errors(i, W, num_samples)
        teddy_errors_means.append(mean_teddy)
        teddy_errors_stds.append(std_teddy)
        jacob_errors_means.append(mean_jacob)
        jacob_errors_stds.append(std_jacob)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(W_values, teddy_errors_means, yerr=teddy_errors_stds, label='Teddy Errors', fmt='o-', capsize=5)
    plt.errorbar(W_values, jacob_errors_means, yerr=jacob_errors_stds, label='Jacob Errors', fmt='s-', capsize=5)
    plt.xlabel('W')
    plt.ylabel('L1 Norm of Error')
    plt.title(f'L1 Norm of Error vs W for i = {i}')
    plt.legend()
    plt.grid(True)
    plt.show()
