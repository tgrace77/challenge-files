#toy_model_codebase
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal, wishart, sem
from scipy.linalg import qr, sqrtm
import cvxpy as cp
from statistics import stdev
from collections import defaultdict
import math
#import gurobipy
#import mosek
from cvxpy_prediction_algorithm_codebase import *




def principal_direction(cov):
    """returns the principal direction of a covariance matrix"""
    evals, evecs = np.linalg.eig(cov)
    i = list(evals).index(max(evals))
    return evecs[:,i]

def avg(array):
    return sum(array)/len(array)

def random_frac(num_clusters):
    a = [0] + sorted(np.random.rand(num_clusters-1)) + [1]
    return np.diff(a)
def generate_expression_data(mean, cov, num_genes, num_cells):
    return mean @ np.ones((num_genes, num_cells)) + sqrtm(cov) @ np.random.randn(num_genes, num_cells)

def generate_sc_reference_data(num_genes, num_clusters, num_cells_in_ref, num_freedom_degree, same_means = False, nonneg = False):
    
    #generate fractions of reference data
    fraction_of_each_type_in_reference = random_frac(num_clusters)
    
    
    #guassian probablity model the expression of each type
    ref_S = np.eye(num_genes)
    cov_of_each_cluster = [wishart.rvs(df = num_freedom_degree, scale = np.array(ref_S))/num_freedom_degree for _ in range(num_clusters)]
    cov_of_each_cluster = [cov/np.linalg.norm(cov) for cov in cov_of_each_cluster]
    cluster_cov_principal_directions = np.array([principal_direction(cov) for cov in cov_of_each_cluster])
    reference_principal_directions_vol = np.linalg.det(cluster_cov_principal_directions@cluster_cov_principal_directions.T)
    vectorized_cov_vol = np.linalg.det(np.stack([cov.flatten() for cov in cov_of_each_cluster])@np.stack([cov.flatten() for cov in cov_of_each_cluster]).T)
    flat_cov_of_each_cluster = np.stack([cov.flatten() for cov in cov_of_each_cluster])
    

    if same_means:
        mean_of_each_cluster = np.hstack(num_clusters*[np.random.randn(num_genes,1)])
    else:
        mean_of_each_cluster = np.random.randn(num_genes,num_clusters)
    
    #normalize means
    norm_of_each_mean = np.linalg.norm(mean_of_each_cluster, axis = 0)
    mean_of_each_cluster = mean_of_each_cluster/norm_of_each_mean
    
    #build single cell reference data
    num_cells_in_cluster_in_ref = [max(round(num_cells_in_ref*fraction),round(.01*num_cells_in_ref)) for fraction in fraction_of_each_type_in_reference]
    fraction_of_each_type_in_reference = np.array([cluster/sum(num_cells_in_cluster_in_ref) for cluster in num_cells_in_cluster_in_ref])

    
    #build sc reference data from gaussain
    expression_in_each_cluster_in_ref = [
    generate_expression_data(mean_of_each_cluster[:,i], cov_of_each_cluster[i], num_genes, num_cells_in_cluster_in_ref[i])
        for i in range(num_clusters)]
    if nonneg:
        #make data nonnegative by shifting each cluster so that the minimum entry is 0
        for i in range(num_clusters):
            expression_in_each_cluster_in_ref[i] -= np.min(expression_in_each_cluster_in_ref[i])
        
        #recalculate means and covariances after shift
        mean_of_each_cluster = np.vstack([avg(profile.T) for profile in expression_in_each_cluster_in_ref]).T
        cov_of_each_cluster = [np.cov(profile) for profile in expression_in_each_cluster_in_ref]
        flat_cov_of_each_cluster = np.stack([cov.flatten() for cov in cov_of_each_cluster])
    

            
    return fraction_of_each_type_in_reference,\
           cov_of_each_cluster,\
           cluster_cov_principal_directions,\
           reference_principal_directions_vol,\
           vectorized_cov_vol,\
           flat_cov_of_each_cluster,\
           mean_of_each_cluster,\
           expression_in_each_cluster_in_ref


def generate_bulk_sample(num_genes, num_clusters, num_cells_in_bulk, num_reps_in_each_bulk_sample, mean_of_each_cluster, cov_of_each_cluster, fraction_of_each_type_in_bulk = None, nonneg = False):
    if fraction_of_each_type_in_bulk is None:
        fraction_of_each_type_in_bulk = random_frac(num_clusters)
    
    #build bulk data
    bulk_sample = np.zeros((num_genes, num_reps_in_each_bulk_sample))
    for j in range(num_reps_in_each_bulk_sample):
        for i in range(num_clusters):
            num_cells_in_cluster_in_bulk = round(num_cells_in_bulk*fraction_of_each_type_in_bulk.flatten()[i])
            expression_of_each_cluster_in_bulk =\
                generate_expression_data(mean_of_each_cluster[:,i], cov_of_each_cluster[i], num_genes, num_cells_in_cluster_in_bulk)
            if nonneg:
                expression_of_each_cluster_in_bulk = expression_of_each_cluster_in_bulk
                
            bulk_sample[:,j] = bulk_sample[:,j] + np.sum(expression_of_each_cluster_in_bulk,1)
    bulk_cov = np.cov(bulk_sample.T, rowvar = False)
    return fraction_of_each_type_in_bulk, bulk_sample, bulk_cov




def generate_sc_ref_and_bulks(num_genes, num_clusters, num_cells_in_ref, num_cells_in_bulk, num_freedom_degree,\
                  num_reps_in_each_bulk_sample, num_tests_to_get_distribution_of_error,\
                  ground_truth_fracs=None, ground_truth_opposed_fracs = None, same_means = False, nonneg = False):
    
    #build one sc reference and num_tests_to_get_distribution_of_error bulk samples
    
    fraction_of_each_type_in_reference,\
    cov_of_each_cluster,\
    cluster_cov_principal_directions,\
    reference_principal_directions_vol,\
    vectorized_cov_vol,\
    flat_cov_of_each_cluster,\
    mean_of_each_cluster,\
    expression_in_each_cluster_in_ref = generate_sc_reference_data(num_genes,\
                                                                   num_clusters,\
                                                                   num_cells_in_ref,\
                                                                   num_freedom_degree,\
                                                                   same_means = same_means,\
                                                                   nonneg = nonneg)
    
    if not(ground_truth_fracs):
        ground_truth_fracs = []
    bulk_sample_expressions = []
    bulk_sample_covs = []
    for test_id in range(num_tests_to_get_distribution_of_error):
        if ground_truth_fracs == None:
            fraction_of_each_type, bulk_sample, bulk_cov = generate_bulk_sample(num_genes, num_clusters, num_cells_in_bulk, num_reps_in_each_bulk_sample, mean_of_each_cluster, cov_of_each_cluster, fraction_of_each_type_in_bulk = None, nonneg = nonneg)
        else: 
            fraction_of_each_type, bulk_sample, bulk_cov = generate_bulk_sample(num_genes, num_clusters, num_cells_in_bulk, num_reps_in_each_bulk_sample, mean_of_each_cluster, cov_of_each_cluster, fraction_of_each_type_in_bulk = ground_truth_fracs[test_id], nonneg = nonneg)
        ground_truth_fracs.append(fraction_of_each_type)
        bulk_sample_expressions.append(bulk_sample)
        bulk_sample_covs.append(bulk_cov)
        
        
    return fraction_of_each_type_in_reference,\
    cov_of_each_cluster,\
    cluster_cov_principal_directions,\
    reference_principal_directions_vol,\
    vectorized_cov_vol,\
    flat_cov_of_each_cluster,\
    mean_of_each_cluster,\
    expression_in_each_cluster_in_ref,\
    ground_truth_fracs[:num_tests_to_get_distribution_of_error],\
    bulk_sample_expressions,\
    bulk_sample_covs





def run_experiment(num_clusters, mean_of_each_cluster, bulk_sample_expressions, flat_cov_of_each_cluster, bulk_sample_covs, num_tests_to_get_distribution_of_error, ground_truth_fracs=None, ground_truth_opposed_fracs = None, same_means = False, lambda_resolution = 64, lambda_step = 64, correct_cov = True, alternate_N_reps = 1, wrong_cov = True, random_pred = True, ignore = False, solver = cp.SCS):
                   
                   
                  
    
    
    list_lambda = [x/lambda_resolution for x in range(lambda_step+1)]
       
    correct_cov_results = defaultdict(list) if correct_cov else None
    wrong_cov_results = defaultdict(list) if wrong_cov else None
    KL_results = defaultdict(list) if KL_objective else None
    random_preds = defaultdict(list) if random_pred else None
    
    
    
    for weight in list_lambda:
        
        if correct_cov:
            if ground_truth_opposed_fracs == None:
                correct_cov_results[weight] = [bento_correct_cov(mean_of_each_cluster.T, bulk_sample_expressions[i].T,\
                               flat_cov_of_each_cluster, bulk_sample_covs[i], weight,\
                               ignore = ignore, alternate_N_reps = alternate_N_reps, solver = solver)\
                               for i in range(num_tests_to_get_distribution_of_error)]
            else:
                correct_cov_results[weight] = [bento_correct_cov(mean_of_each_cluster.T, bulk_sample_expressions[i].T,\
                               flat_cov_of_each_cluster, bulk_sample_covs[i], weight,\
                               ignore = ignore, alternate_N_reps = alternate_N_reps,\
                               opp_frac = ground_truth_opposed_fracs[i], solver = solver)\
                               for i in range(num_tests_to_get_distribution_of_error)]
        
                   
        if wrong_cov:
            if ground_truth_opposed_fracs == None:
                wrong_cov_results[weight] = [bento_wrong_cov(mean_of_each_cluster.T, bulk_sample_expressions[i].T,\
                               flat_cov_of_each_cluster, bulk_sample_covs[i], weight,\
                               solver = solver)\
                               for i in range(num_tests_to_get_distribution_of_error)]
            else:
                wrong_cov_results[weight] = [bento_wrong_cov(mean_of_each_cluster.T, bulk_sample_expressions[i].T,\
                               flat_cov_of_each_cluster, bulk_sample_covs[i], weight,\
                               opp_frac = ground_truth_opposed_fracs[i], solver = solver)\
                               for i in range(num_tests_to_get_distribution_of_error)]
                
        if KL_objective:
            KL_results[weight] = [KL_objective(mean_of_each_cluster.T, bulk_sample_expressions[i].T, flat_cov_of_each_cluster,\
                                  bulk_sample_covs[i], weight, solver = solver)\
                                  for i in range(num_tests_to_get_distribution_of_error)]
            
                
        if random_pred:
            random_preds[weight] = [random_frac(num_clusters) for i in range(num_tests_to_get_distribution_of_error)]
            
    
    return correct_cov_results, wrong_cov_results, KL_results, random_preds


def run_experiment_n_times(n, num_genes, num_clusters, num_cells_in_ref, num_cells_in_bulk, num_freedom_degree, num_reps_in_each_bulk_sample, num_tests_to_get_distribution_of_error, ground_truth_fracs=None, ground_truth_opposed_fracs = None, same_means = False, nonneg = False, lambda_resolution = 25, lambda_step = 25, correct_cov = False, alternate_N_reps = 1, wrong_cov = True, KL_objective = False, random_pred = False, ignore = False, solver = cp.SCS):
    input_data_list = []
    correct_cov_results_list = []
    wrong_cov_results_list = []
    KL_results_list = []
    random_preds_list = []
    for i in range(n):
        random.seed(i)
        np.random.seed(i)
        fraction_of_each_type_in_reference,\
        cov_of_each_cluster,\
        cluster_cov_principal_directions,\
        reference_principal_directions_vol,\
        vectorized_cov_vol,\
        flat_cov_of_each_cluster,\
        mean_of_each_cluster,\
        expression_in_each_cluster_in_ref,\
        ground_truth_fracs[:num_tests_to_get_distribution_of_error],\
        bulk_sample_expressions,\
        bulk_sample_covs = generate_sc_ref_and_bulks(num_genes, num_clusters, num_cells_in_ref, num_cells_in_bulk,\
                                                     num_freedom_degree, num_reps_in_each_bulk_sample,\
                                                     num_tests_to_get_distribution_of_error,\
                                                     ground_truth_fracs=ground_truth_fracs,\
                                                     ground_truth_opposed_fracs = ground_truth_opposed_fracs,\
                                                     same_means = same_means, nonneg = nonneg)
        
        input_data_list.append((fraction_of_each_type_in_reference,\
        cov_of_each_cluster,\
        cluster_cov_principal_directions,\
        reference_principal_directions_vol,\
        vectorized_cov_vol,\
        flat_cov_of_each_cluster,\
        mean_of_each_cluster,\
        expression_in_each_cluster_in_ref,\
        ground_truth_fracs[:num_tests_to_get_distribution_of_error],\
        bulk_sample_expressions,\
        bulk_sample_covs))
        
       
        
        correct_cov_results, wrong_cov_results, KL_results, random_preds = run_experiment(num_clusters, mean_of_each_cluster,\
                                                                              bulk_sample_expressions,\
                                                                              flat_cov_of_each_cluster, bulk_sample_covs,\
                                                                              num_tests_to_get_distribution_of_error,\
                                                                              ground_truth_fracs, ground_truth_opposed_fracs,\
                                                                              same_means, lambda_resolution, lambda_step,\
                                                                              correct_cov, alternate_N_reps, wrong_cov,\
                                                                              random_pred, ignore, solver)
        correct_cov_results_list.append(correct_cov_results)
        wrong_cov_results_list.append(wrong_cov_results)
        KL_results_list.append(KL_results)
        random_preds_list.append(random_preds)
    return input_data_list, correct_cov_results_list, wrong_cov_results_list, KL_results_list, random_preds_list
        
    
