import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

# Function to calculate the L1 norm error
def plot_errors(i_values, W_values, num_samples, C1, x1_size, A_list, D_list, E, Sigma, ground_truth_fracs, known, num_epochs, A_jacob, E_jacob, D_jacob, Sigma_jacob):
    # Plotting
    for i in i_values:
        teddy_errors_means = []
        teddy_errors_stds = []
        jacob_errors_means = []
        jacob_errors_stds = []
        
        for W in W_values:
            mean_teddy, std_teddy, mean_jacob, std_jacob = calculate_errors(i, W, num_samples,  C1, x1_size, A_list, D_list, E, Sigma, ground_truth_fracs, known, num_epochs)
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

