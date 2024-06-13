# -*- coding: utf-8 -*-
"""optimization

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Quol8c0P6DHaIR8Gso6ff_mZAkV-2xdI
"""

# Cell for Optimization and Loss Computation

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils import *
from cvxpy_prediction_algorithm_codebase import *
from metrics_and_plots_codebase import *

def f(x1, A, E_combined, E_transpose, x2, C1, C2, C3, C4, C5, P, L, W, sigma, x3, D, known):
    """
    Defines the loss function for the optimization process.

    Args:
    x1, x2, x3 (torch.Tensor): Parameters to be optimized.
    A, E_combined, E_transpose, sigma (torch.Tensor): Matrices involved in the loss computation.
    C1, C2, C3, C4, C5, P, L, W (float): Constants defining the specifics of the loss function.
    D (torch.Tensor): Vector D in the optimization context.

    Returns:
    torch.Tensor: Computed loss value.
    """
    x1_positive = compute_positive(x1)
    x2_positive = compute_positive(x2)
    x3_positive = compute_positive(x3)
    x1_diag = torch.diag(x1_positive)
    E_x_expanded_full, sigma_x1, E_x = matrix_operations(A, E_combined, x1_positive, sigma, known)
    if not known:
      det_approx = compute_determinant_approx(x3_positive)
    else:
      det_approx = x3_positive


    # Regularization terms (adjust or clarify for your loss function)
    log_sum_1 = C1 * torch.log(x1_positive).sum()
    log_sum_2 = C2 * torch.log(torch.norm(x2_positive - C3) ** 2).sum()
    log_sum_3 = C4 * torch.log(det_approx).sum()
    log_sum_4 = C5 * torch.log(torch.norm(x3)**2 -P).sum()

    # Loss computation
    E_result = torch.matmul(E_combined, x1_diag) @ E_combined.t()
    E_result = E_result.flatten()
    combined_result = torch.norm((D - sigma_x1 - E_result) + (1 / sum(x1)) * ((E_x.unsqueeze(0).t() * (E_x.unsqueeze(0))).flatten()))**2
    norm_squared_sum = torch.norm(A - E_x_expanded_full, dim=1).pow(2).sum()

    return (((1 - W) * combined_result) + log_sum_3 + log_sum_4) + ((W * norm_squared_sum) + log_sum_1 + log_sum_2)

def grad_descent_known(C1, C2, C3, C4, C5, P, L, W, x1, x2, x3, A, D, E, Sigma, known):
  x1 = x1.detach().clone().requires_grad_(True)

  E_combined = torch.tensor(E, dtype=torch.float32)
  E_transpose = E_combined.t()

  A = torch.tensor(A, dtype=torch.float32)
  Sigma = torch.tensor(Sigma, dtype=torch.float32)
  D = torch.tensor(D, dtype=torch.float32)
  optimizer = optim.Adam([x1], lr=0.001)

  losses = []
  parameter_changes = []
  previous_parameters = x1.detach().flatten()

  # Optimization loop
  for i in range(10000):
      optimizer.zero_grad()
      loss = f(x1, A, E_combined, E_transpose, x2, C1, C2, C3, C4, C5, P, L, W, Sigma, x3, D, known)
      C1 = 1 / ((i + 1) ** 2)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        x1.clamp_(min=0)
      # Store loss
      losses.append(loss.item())

      # Calculate and store parameter changes for x1 only
      current_parameters = x1.detach().flatten()
      param_change = torch.log(torch.norm(current_parameters - previous_parameters))
      parameter_changes.append(param_change.item())
      previous_parameters = current_parameters

  #     # Plot loss over iterations
  # plt.figure(figsize=(12, 5))
  # plt.subplot(1, 2, 1)
  # plt.plot(losses, label='Loss')
  # plt.title('Loss Function over Iterations')
  # plt.xlabel('Iteration')
  # plt.ylabel('Loss')
  # plt.legend()

  # # Plot parameter changes
  # plt.subplot(1, 2, 2)
  # plt.plot(parameter_changes, label='Log of Parameter Changes')
  # plt.title('Log of Norm of Parameter Changes')
  # plt.xlabel('Iteration')
  # plt.ylabel('Log(Norm of Changes)')
  # plt.legend()
  # plt.show()
  
  # print("x1 optimized: ")
  # print(x1)
  
  # print("x2 optimized: ")
  # print(x2) 
  
  # print("x3 optimized: ")
  # print(x3)

  return x1
  
  pass

def grad_descent(C1, C2, C3, C4, C5, P, L, W, known_column_Esig, unknown_column_Esig, x1_size, A_column, A_row, D_size, E_size, MS_size, known):
  """
  Performs the gradient descent optimization over specified parameters and hyperparameters.

  Args:
  C1, C2, C3, C4, C5, P, L, W (float): Constants defining the specifics of the loss function and its regularization.
  known_column_Esig, unknown_column_Esig (int): Specifies the sizes of known and learnable parts of matrices E and Sigma.
  x1_size, A_column, A_row, D_size, E_size, MS_size (int): Dimensions of vectors and matrices involved in the computation.
  """
    # Define the vector as a learnable parameter
  x1 = torch.randn(x1_size, requires_grad=True)

  # Matrix A
  A = torch.randint(low=0, high=2, size=(A_row, A_column)).float()

  # Matrix E known columns
  E_known = torch.randint(low=0, high=2, size=(E_size, known_column_Esig)).float()

  # Unknown Values of Matrix E
  x2 = torch.randn(100, unknown_column_Esig, requires_grad=True)

  # Matrix E
  E_combined = torch.cat((E_known, x2), dim=1)
  E_transpose = E_combined.t()

  # Vector D size
  D = torch.randn(D_size)

  # Matrix sigma known columns
  sigma_known = torch.randn(MS_size, known_column_Esig)

  # Matrix sigma unknown columns
  x3 = torch.randn(MS_size, unknown_column_Esig, requires_grad=True)

  # Matrix Sigma
  sigma = torch.cat((sigma_known, x3), dim=1)



  # Set up the optimizer
  optimizer = optim.Adam([x1, x2, x3], lr=0.01)

  losses = []
  parameter_changes = []
  previous_parameters = torch.cat((x1.detach().flatten(), x2.detach().flatten(), x3.detach().flatten()))

  # Optimization loop
  for i in range(1000):
      optimizer.zero_grad()
      loss = f(x1, A, E_combined, E_transpose, x2, C1, C2, C3, C4, C5, P, L, W, sigma, x3, D, known)
      loss.backward()
      optimizer.step()

      # Store loss
      losses.append(loss.item())

      # Calculate and store parameter changes
      current_parameters = torch.cat((x1.detach().flatten(), x2.detach().flatten(), x3.detach().flatten()))
      param_change = torch.log(torch.norm(current_parameters - previous_parameters))
      parameter_changes.append(param_change.item())
      previous_parameters = current_parameters

      if i % 10 == 0:
          print(f"Step {i}, Current loss: {loss.item()}")

  # Plot loss over iterations
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(losses, label='Loss')
  plt.title('Loss Function over Iterations')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.legend()

  # Plot parameter changes
  plt.subplot(1, 2, 2)
  plt.plot(parameter_changes, label='Log of Parameter Changes')
  plt.title('Log of Norm of Parameter Changes')
  plt.xlabel('Iteration')
  plt.ylabel('Log(Norm of Changes)')
  plt.legend()
  plt.show()
  
  print("x1 optimized: ")
  print(x1)
  
  print("x2 optimized: ")
  print(x2) 
  
  print("x3 optimized: ")
  print(x3)
  
  pass



def calculate_errors(i, W, num_samples,  C1, x1, x2, x3, A_list, D_list, E, Sigma, ground_truth_fracs, known, num_epochs):
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
        x1 = grad_descent_known(C1, C2=0, C3=0, C4=0, C5=0, P=0, L=0, W=W, x1 = x1, x2 = x2, x3 = x3, A=A, D=D, E=E, Sigma=Sigma, known=known)
        jacob_results = pytorch_cov_prediction(A_jacob, E_jacob, D_jacob, Sigma_jacob, W, 1, 1, num_epochs=num_epochs)
        
        x1 = x1.squeeze()
        teddy_alpha = x1 / sum(x1)
        jacob_alpha = jacob_results[0]

        ted_error = L1_norm(teddy_alpha, ground_truth_frac)
        jacob_error = L1_norm(jacob_alpha, ground_truth_frac)
        
        teddy_errors.append(ted_error)
        jacob_errors.append(jacob_error)
    
    return torch.mean(teddy_errors), torch.std(teddy_errors), torch.mean(jacob_errors), torch.std(jacob_errors)


      