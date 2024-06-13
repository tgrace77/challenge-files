#metrics_and_plots_codebase
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal, wishart, sem, entropy
from scipy.linalg import qr, sqrtm
import cvxpy as cp
from statistics import stdev
from collections import defaultdict
import math


def avg(array):
    return sum(array)/len(array)


def L1_norm(p,q):
    return sum([abs(p.flatten()[i]-q.flatten()[i]) for i in range(len(p))])

def L2_norm(p,q):
    return math.sqrt(sum([(p.flatten()[i]-q.flatten()[i])**2 for i in range(len(p))]))
               
def max_cluster(p,q):
    max_index = list(p.flatten()).index(max(p.flatten()))
    predicted_fraction = q.flatten()[max_index]
    return sorted(list(q), reverse=True).index(predicted_fraction)
   
               
def KL_div(p,q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def max_frac_diff(p,q):
    return max(p)/max(q)

def min_frac_diff(p,q):
    return min(p)/min(q) if min(q) else -1

def sorted_L1(p,q):
    p = sorted(p)
    q = sorted(q)
    return L1_norm(p,q)

def process_results(results):
    result_alphas = {key:[result[0] for result in results[key]] for key in results.keys()}
    result_Ns = {key:[result[1] for result in results[key]] for key in results.keys()}
    result_mins = {key:[result[2] for result in results[key]] for key in results.keys()}
    return result_alphas, result_Ns, result_mins

def process_results_list(results_list):
    if type(results_list) == dict:
        results_list = list(results_list.values())
    result_alphas_list = []
    result_Ns_list = []
    result_mins_list = []
    for result in results_list:
        try:
            alphas, Ns, mins = process_results(result)
            result_alphas_list.append(alphas)
            result_Ns_list.append(Ns)
            result_mins_list.append(mins)
        except:
            pass
    return result_alphas_list, result_Ns_list, result_mins_list



def make_errors_from_alphas(alphas, ground_truths, error_fn):
    full_errors = {key:[error_fn(alphas[key][i],ground_truths[i]) for i in range(len(alphas[key])) if not(alphas[key][i] is None)] for key in alphas.keys()}
    avg_errors_and_stdevs = {key:[avg([x for x in full_errors[key] if not(math.isnan(x))]), stdev([x for x in full_errors[key] if not(math.isnan(x))])] for key in alphas.keys()}
    return full_errors, avg_errors_and_stdevs

def make_errors_from_alphas_list(alphas_list, ground_truth_fracs, error_fn):
    full_errors_list = []
    avg_errors_and_stdevs_list = []
    for alphas in alphas_list:
        full_errors, avg_errors_and_stdevs = make_errors_from_alphas(alphas, ground_truth_fracs, L1_norm)
        full_errors_list.append(full_errors)
        avg_errors_and_stdevs_list.append(avg_errors_and_stdevs)
    return full_errors_list, avg_errors_and_stdevs_list

def make_weights_plot(avg_errors_and_stdevs, single_run = True, title = "", ylabel = ""):
    if single_run:
        X = sorted(avg_errors_and_stdevs.keys())
        Y = [avg_errors_and_stdevs[key][0] for key in X]
        Yerr = [avg_errors_and_stdevs[key][1] for key in X]
        plt.title(title)
        plt.xlabel('weight')
        plt.ylabel(ylabel)
        plt.errorbar(X,Y,Yerr)
        plt.show()
        return
    else:
        X = sorted(avg_errors_and_stdevs[0].keys())
        Y = [avg([avg_errors_and_stdevs[i][key][0] for i in range(len(avg_errors_and_stdevs))]) for key in X]
        print(X)
        print(Y)
        try:
            Yerr = [stdev([avg_errors_and_stdevs[i][key][0] for i in range(len(avg_errors_and_stdevs))]) for key in X]
        except:
            Yerr = None
        plt.title(title)
        plt.xlabel('weight')
        plt.ylabel(ylabel)
        try:
            plt.errorbar(X,Y,Yerr)
        except:
            plt.scatter(X,Y)
        plt.show()
        return
    
def multi_weights_plot(avg_errors_and_stdevs_list, legend, title = "", ylabel = ""):
    X = []
    Y = []
    Yerr = []
    for errors in avg_errors_and_stdevs_list:
        X.append(sorted(errors[0].keys()))
        Y.append([avg([errors[i][key][0] for i in range(len(errors))]) for key in sorted(errors[0].keys())])
        try:
            Yerr.append([stdev([errors[i][key][0] for i in range(len(errors))]) for key in sorted(errors[0].keys())])
        except:
            Yerr.append([0 for key in sorted(errors[0].keys())])
        plt.title(title)
        plt.xlabel('weight')
        plt.ylabel(ylabel)
    for i in range(len(avg_errors_and_stdevs_list)):
        plt.errorbar(X[i],Y[i],Yerr[i],label = legend[i])
    plt.legend()
    plt.show()
    return

def plot_cdf_from_dict(data_dict):
    # Combine all non-NaN values into a single list
    all_values = [val for sublist in data_dict.values() for val in sublist if not np.isnan(val)]

    # Compute the CDF
    sorted_data = np.sort(all_values)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))

    # Plotting the CDF
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_data, yvals, label="CDF")
    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Density Function (CDF)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_cdf_from_dict_list(data_list):
    all_values = []

    for data_dict in data_list:
        all_values.extend([val for sublist in data_dict.values() for val in sublist if not np.isnan(val)])

    sorted_data = np.sort(all_values)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))

    # Plotting the CDF
    plt.figure(figsize=(10, 8))
    plt.plot(sorted_data, yvals, label="Combined CDF")
    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Density Function (CDF)')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_cdf_from_dict_lists(data_list, labels):
    plt.figure(figsize=(10, 8))

    for data, label in zip(data_list, labels):
        all_values = []
        for data_dict in data:
            all_values.extend([val for sublist in data_dict.values() for val in sublist if not np.isnan(val)])

        sorted_data = np.sort(all_values)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))

        # Plotting the CDF for each data list
        plt.plot(sorted_data, yvals, label=label)

    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Density Function (CDF)')
    plt.grid(True)
    plt.legend()
    plt.show()
