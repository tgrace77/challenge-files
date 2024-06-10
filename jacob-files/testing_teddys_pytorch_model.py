# -*- coding: utf-8 -*-
"""testing_teddys_pytorch_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZfpVYuse1JXSAEyuBU1vMfzoUexFs1a1
"""

import numpy as np
from toy_model_codebase import *
import torch

#You can play around with any of these parameters, except for
#num_clusters, num_freedom_degree, and num_tests_to_get_distributions_of_error
def generate_sample_data():
    num_genes = 50
    num_clusters = 3
    num_cells_in_ref = 400
    num_cells_in_bulk = 200
    num_freedom_degree = 2*num_genes
    num_reps_in_each_bulk_sample = 10
    num_tests_to_get_distribution_of_error = 19



    #don't play with these
    ground_truth_fracs = [[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],\
                          [2/3,1/3,0],[2/3,0,1/3],[1/3,2/3,0],[1/3,0,2/3],[0,2/3,1/3],[0,1/3,2/3],\
                          [.4, .4,.2],[.4,.2,.4],[.2,.4,.4], [.5,.25,.25],[.25,.5,.25],[.25,.25,.5],\
                          [1/3,1/3,1/3]]


    ground_truth_opposed_fracs = [[0,0,1],[0,0,1],[1,0,0],[0,0,1],[0,1,0],[1,0,0],\
                                 [0,0,1],[0,1,0],[0,0,1],[0,1,0],[1,0,0],[1,0,0],\
                                 [0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0],\
                                 [1,0,0]]

    ground_truth_fracs = [np.array([frac]) for frac in ground_truth_fracs]
    ground_truth_opposed_fracs = [np.array([frac]) for frac in ground_truth_opposed_fracs]

    #this will generate synthetic data for you to test on

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
        bulk_sample_covs = generate_sc_ref_and_bulks(num_genes, num_clusters, num_cells_in_ref, num_cells_in_bulk, num_freedom_degree,\
                      num_reps_in_each_bulk_sample, num_tests_to_get_distribution_of_error,\
                      ground_truth_fracs= ground_truth_fracs, ground_truth_opposed_fracs = None, same_means = False, nonneg = False)

    #the ith entries of A_list and D_list are the relevant matrices for each of the num_tests_to_get_distribution_of_error
    #data set generated in the cell above

    A_list = bulk_sample_expressions
    D_list = [cov.flatten().T for cov in bulk_sample_covs]

    #E and Sigma are the same for each of the synthetic datasets generated above
    E = mean_of_each_cluster
    Sigma = flat_cov_of_each_cluster.T


    #for now, just test on the first bulk sample generated above, but start changing this around when your algorithm
    #is working on this dataset
    A = A_list[0]
    A = A.T.flatten().T
    D = D_list[0]

    #set w = .5 for now, but w can be any value from 0 to 1 (inclusive) play around with this once your model is working

    w = .5

    print("A.shape = ", A.shape)
    print("D.shape = ", D.shape)
    print("E.shape = ", E.shape)
    print("Sigma.shape = ", Sigma.shape)

    #run your algorithm on A, D, E, Sigma, and w from the previous cell
    #make the same plots as in your coding challenge

