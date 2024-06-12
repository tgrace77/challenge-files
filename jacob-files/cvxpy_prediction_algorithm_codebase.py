#cvxpy_prediction_algorithm_codebase
import numpy as np
import cvxpy as cp
import math
import torch
from scipy.linalg import sqrtm
#import gurobipy
#import mosek

def avg(array):
    return sum(array)/len(array)

def first_moment_objective(cluster_avg_expressions, bulk_gene_expressions, unnormalized_ratios):
    return np.sum(np.square((bulk_gene_expressions - unnormalized_ratios@cluster_avg_expressions)))/np.sum(np.square(bulk_gene_expressions))

def bento_cvxpy_first_moment_only_with_normalization_and_preconditioning(cluster_avg_expressions, bulk_gene_expressions, solver = cp.OSQP, verbose = False, normalization_term = 'l2_norm_of_bulk', preconditioning = 'none'):
    if normalization_term == 'l2_norm_of_bulk':
        C = math.sqrt(np.sum(np.square(bulk_gene_expressions)))
    elif normalization_term == 'N_uniform_estimate':
        N = (avg(bulk_gene_expressions)@avg(cluster_avg_expressions).T)/(avg(cluster_avg_expressions)@avg(cluster_avg_expressions).T)
        x_uniform = np.array([len(cluster_avg_expressions)*[N/len(cluster_avg_expressions)]])
        C = np.sum(np.square(np.ones((len(bulk_gene_expressions),1))@x_uniform@cluster_avg_expressions-np.array(bulk_gene_expressions)))
    elif normalization_term == 'original_normalization_term':
        C = np.sum(np.square(bulk_gene_expressions))
        
    if preconditioning == 'diagonal':
        P = np.diag(np.diagonal(sqrtm(cluster_avg_expressions@cluster_avg_expressions.T)))
        cluster_avg_expressions = P@cluster_avg_expressions
        x = cp.Variable((1,len(cluster_avg_expressions)), nonneg = True)
        constraint = [x >= 0]
    elif preconditioning == 'none':
        P = np.eye(len(cluster_avg_expressions))
        x = cp.Variable((1,len(cluster_avg_expressions)), nonneg = True)
        constraint = [x >= 0]
    elif preconditioning == 'exact':
        P = sqrtm(cluster_avg_expressions@cluster_avg_expressions.T)
        cluster_avg_expressions = P@cluster_avg_expressions
        x = cp.Variable((1,len(cluster_avg_expressions)))
        constraint = [x@P >= 0]
         
    objective = cp.Minimize(\
cp.sum_squares(np.ones((len(bulk_gene_expressions),1))@x@cluster_avg_expressions-np.array(bulk_gene_expressions))/C)
    
    prob = cp.Problem(objective, constraint)
        
    result = prob.solve(solver = solver, verbose = verbose)
    if (x.value is None):
        return None, None, None
    else:
        if preconditioning == 'none':
            X = (x.value[0]).flatten()
        else:
            X = (x.value[0]@P).flatten()
        N = sum(X)
        alpha = X/N
        return alpha, N, result

class AlphaCovModel(torch.nn.Module):
    def __init__(self, n_clusters, use_projected_gradient=True, vectorize = True):
        super(AlphaCovModel, self).__init__()
        self.use_projected_gradient = use_projected_gradient
        self.vectorize = vectorize
        self.alpha = torch.nn.Parameter(torch.rand(1, n_clusters) * 0.01) # Initialize with small random values
        self.alpha.clamp(min = 0)
      
    def forward(self, X, C, Y, Sigma, W, non_convex_part_weight, diag_part_weight, epoch):
        C_replicated = C.repeat(1, X.shape[0]).type(torch.FloatTensor)
        C = C.type(torch.FloatTensor)
        first_moment_term = torch.sum((X.flatten() - self.alpha @ C_replicated) ** 2)/(X.flatten()@X.flatten())
        
        diag_alpha = torch.diag(self.alpha.flatten())
                          # this is the ordinary empirical covariance matrix
        #print('non-convex part ', (torch.transpose(self.alpha@C,0,1) @ self.alpha@C))
        #print('convex diag part ', torch.transpose(C,0,1)@diag_alpha@C)
        second_moment_term = torch.sum((Y.flatten()\
                          #this next part is the non-convex part
                  + non_convex_part_weight/torch.sum(self.alpha) * (torch.transpose(self.alpha@C,0,1) @ self.alpha@C).flatten()\
                          #this is the ordinary linear approximation of Sigma
                          - self.alpha @ Sigma.type(torch.FloatTensor)\
                          #this is the convex extra "diag" part
                          - diag_part_weight*(torch.transpose(C,0,1)@diag_alpha@C).flatten())**2)/(Y.flatten()@Y.flatten())
        
        loss = (1-W)*first_moment_term + W*second_moment_term
        
        if not self.use_projected_gradient:
            loss += -1 / (epoch + 1) ** 2 * torch.sum(torch.log(self.alpha + 1e-6))
            
        return loss
    
    def project(self):
        self.alpha.data[self.alpha.data < 0] = 0  # Project onto non-negative values
        
def pytorch_cov_prediction(X, C, Y, Sigma, W, non_convex_part_weight, diag_part_weight, use_projected_gradient=True, vectorize = True, num_epochs=1000):
    X = torch.tensor(X, dtype=float)
    C = torch.tensor(C, dtype=float)
    Y = torch.tensor(Y, dtype=float)
    Sigma = torch.tensor(Sigma, dtype = float)
    
    # Create the model 
    model = AlphaCovModel(len(C), use_projected_gradient=use_projected_gradient, vectorize=vectorize)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # Adjust learning rate
    
    loss_history = []
    point_diff_history = []
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        loss = model(X, C, Y, Sigma, W, non_convex_part_weight, diag_part_weight, epoch)
        learned_loss = loss.detach().clone()

    
        old_parameters = model.alpha.detach().clone()

    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Project onto non-negative values

        model.project()
            
            
        loss_history.append(learned_loss)
        point_diff_history.append(torch.norm(old_parameters - model.alpha.detach().clone()))
    
    alpha = model.alpha.detach().clone().flatten()
    
    return alpha / sum(alpha), loss_history, point_diff_history


class AlphaPlainSolver(torch.nn.Module):
    def __init__(self, n_clusters, use_projected_gradient=True, vectorize = True):
        super(AlphaPlainSolver, self).__init__()
        self.use_projected_gradient = use_projected_gradient
        self.vectorize = vectorize
        self.alpha = torch.nn.Parameter(torch.rand(1, n_clusters) * 0.01) # Initialize with small random values
        self.alpha.clamp(min = 0)

    def forward(self, X, C, epoch):
        
        if not self.vectorize:
            loss = torch.sum(((X-self.alpha @ C.type(torch.FloatTensor)).type(torch.FloatTensor)/(X.flatten()@ X.flatten()))**2)
            
        else:
            # Replicate C n_samples times
            C_replicated = C.repeat(1, X.shape[0]).type(torch.FloatTensor)

            # Compute the objective function
            loss = torch.sum(((X.flatten() - self.alpha @ C_replicated)) ** 2)/(X.flatten()@X.flatten())
            
        if not self.use_projected_gradient:
            loss += -1/(epoch+1)**2 * torch.sum(torch.log(self.alpha + 1e-6))
        
        return loss

    def project(self):
        self.alpha.data[self.alpha.data < 0] = 0   # Project onto non-negative values


def pytorch_plain(X, C, use_projected_gradient=True, vectorize = True, num_epochs=1000, lr = .001):
    X = torch.tensor(X, dtype=float)
    C = torch.tensor(C, dtype=float)
    
    # Create the model 
    model = AlphaPlainSolver(len(C), use_projected_gradient=use_projected_gradient, vectorize = vectorize)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adjust learning rate
    
    loss_history = []
    point_diff_history = []
    alpha_history = []
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        loss = model(X, C, epoch)
        learned_loss = loss.detach().clone()

    
        old_parameters = model.alpha.detach().clone()
        alpha_history.append(old_parameters)

    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Project onto non-negative values
        #if use_projected_gradient:
        model.project()
            
            
        C_replicated = C.repeat(1, X.shape[0]).type(torch.float32)
        loss_history.append(learned_loss)
        point_diff_history.append(torch.norm( old_parameters - model.alpha.detach().clone()   ))
    
    alpha_history.append(model.alpha.detach().clone().flatten())
    
    return alpha_history, loss_history, point_diff_history


def bento_correct_cov(cluster_avg_expressions, bulk_gene_expressions, cluster_covs, bulk_cov, weight, ignore=False, empirical = False, alternate_N_reps = 1, opp_frac = None, solver = cp.SCS):
    
    """
    input:
    
    cluster_avg_expressions (n_clusters x n_genes matrix): average cell-level gene expression of each cluster type
    
    bulk_gene_expressions (n_samples x n_genes matrix): bulk expression data of all available samples
    
    cluster_covs (n_clusters x (n_genes)^2 matrix): cluster-level covariance matrices flattened and stacked into a single matrix
    
    bulk_cov (1 x (n_genes)^2 matrix): flattened covariance matrix of the bulk sample data
    
    weight (float between 0 and 1, inclusive): parameter allocating importance to mean vs. covariance part of the objective function
    ignore  (boolean): flag to determine whether or not the code ignores the term we expect to be small
    
    opp_frac (1 x n_clusters matrix): in the toy model, if we know the true cluster type fraction we sometimes push alpha in the direction of the wrong cluster type fraction to disrupt cvxpy's tendency to predict the uniform fraction
    
    solver: flag to specify which cvxpy solver to use
    
    
    output:
    
    alpha (1 x n_clusters matrix): predicted cell type fractions of the bulk sample
    
    N (float): predicted total number of cells in the bulk sample
    
    result (float): minimization of objective function
    """
    
    
    C = np.sum(np.square(bulk_gene_expressions))     #C, D, and E are used to normalize the objective function to
    D = np.sum(np.square(bulk_cov))                  #give reasonable numbers
    ignore_coeff = 0
    
    
    #we are not sure what to do with the term 1/N*E(bulk_gene_expression)^T@E(bulk_gene_expression) 
    #in the bulk covariance equation, so the ignore flag allows us to do one of two things:
         #1. if ignore == True, we simply do not include this term in the objective function
         #2. if ignore == False, we estimate N by solving bento_plain on the problem data and include the 
         #term in this objective function
        
    if not ignore:
        alpha_plain, N_plain, results_plain = bento_plain(cluster_avg_expressions, bulk_gene_expressions, solver = solver)
        ignore_coeff = 1 / N_plain
  
    expected_bulk = np.array([np.mean(bulk_gene_expressions, axis=0)])
    
    
    results_list = []
    for rep in range(alternate_N_reps):
        
        E = np.sum(np.square(ignore_coeff * expected_bulk.T @ expected_bulk))
        
        x = cp.Variable((1, len(cluster_avg_expressions)), nonneg=True)
    
        objective = (1 - weight) * expectation_objective(cluster_avg_expressions, bulk_gene_expressions, x)/C \
+ weight * covariance_objective(cluster_avg_expressions, bulk_gene_expressions, cluster_covs, bulk_cov, x, ignore_coeff, empirical)/(D+E)
    
        if not(opp_frac is None):
            objective += .00001*cp.sum_squares(x - opp_frac)
    
        #TEMPORARY_CONSTRAINT = [cp.sum(x) == 200]
        prob = cp.Problem(cp.Minimize(objective))
        result = prob.solve(solver = solver)
        if x.value is None:
            results_list.append((None,None,None))
            return results_list
        else:
            N = sum(x.value[0])
            alpha = x.value[0] / N
            ignore_coeff = 1/N
            results_list.append((alpha, N, result))
    return results_list[0] if alternate_N_reps == 1 else results_list
        

def bento_wrong_cov(cluster_avg_expressions, bulk_gene_expressions, cluster_covs, bulk_covs, weight, opp_frac = None, solver = cp.SCS):
    
    """
    This function takes in a cluster average expression profile and
    and a collection of sample bulk gene expression profiles (10
    samples per population) together with covariance data and returns a cell type fraction
    prediction. (This is "Bento's algorithm with covariance".)
    """
    
    C = math.sqrt(np.sum(np.square(bulk_gene_expressions)))
    D = math.sqrt(np.sum(np.square(bulk_covs)))
    x = cp.Variable((1,len(cluster_avg_expressions)), nonneg = True)
    objective = (1-weight)*\
cp.sum_squares(((np.ones((len(bulk_gene_expressions),1))@x@cluster_avg_expressions-np.array(bulk_gene_expressions)))/C)\
            +weight*cp.sum_squares((sum(x@cluster_covs)-bulk_covs.flatten())/D)
    if not(opp_frac is None):
        objective += .00001*cp.sum_squares(x-opp_frac)
    
    prob = cp.Problem(cp.Minimize(objective))
    result = prob.solve(solver = solver)
    if x.value is None:
        return None, None, None
    else:
        N = sum(x.value[0])
        alpha = x.value[0] / N
        return alpha, N, result


def expectation_objective(cluster_avg_expressions, bulk_gene_expressions, x):
    objective = cp.sum_squares(np.ones((len(bulk_gene_expressions), 1)) @ x @ cluster_avg_expressions - np.array(bulk_gene_expressions))
    return objective

def covariance_objective(cluster_avg_expressions, bulk_gene_expressions, cluster_covs, bulk_cov, x, ignore_coeff, empirical):
    
    if empirical:
        expected_bulk = np.array([np.mean(bulk_gene_expressions, axis=0)])
    else:
        expected_bulk = x@cluster_avg_expressions
    mu = [np.array([cluster_avg_expressions[i]]) for i in range(len(cluster_avg_expressions))]
    
    objective = cp.sum_squares((x @ cluster_covs).flatten() \
             + (x @ np.stack([(mu[i].T @ mu[i]).flatten() for i in range(len(mu))])).flatten() \
             - bulk_cov.flatten() \
             - (1*ignore_coeff +0/200) * (expected_bulk.T @ expected_bulk).flatten())
    return objective


"""if you have n_samples bulk samples with n_genes for one problem, 
bulk_gene_expressions should be a n_samples x n_genes matrix"""

def bento_plain(cluster_avg_expressions, bulk_gene_expressions, preconditioning, solver = cp.SCS):
    C = np.sum(np.square(bulk_gene_expressions)) #constant for normalization in the cvxpy objective function
    
    
    #cvxpy routine start
    x = cp.Variable((1,len(cluster_avg_expressions)), nonneg = True) #xi = number of cells in cluster i
    
    P, cluster_avg_expressions, x, constraint = get_precondition(preconditioning, cluster_avg_expressions)

    objective = cp.Minimize(\
cp.sum_squares(np.ones((len(bulk_gene_expressions),1))@x@cluster_avg_expressions-np.array(bulk_gene_expressions))/C)
    
    prob = cp.Problem(objective, constraint)
        
    result = prob.solve(solver = solver)
    #cvxpy routine end

    if (x.value is None):
        return None, None, None
    else:
        if preconditioning == 'none':
            X = (x.value[0]).flatten()
        else:
            X = (x.value[0]@P).flatten()
        N = sum(X)
        alpha = X/N
        return alpha, N, result

def get_precondition(preconditioning, cluster_avg_expressions):
    if preconditioning == 'diagonal':
        P = np.diag(np.diagonal(sqrtm(cluster_avg_expressions@cluster_avg_expressions.T)))
        cluster_avg_expressions = P@cluster_avg_expressions
        x = cp.Variable((1,len(cluster_avg_expressions)), nonneg = True)
        constraint = [x >= 0]
    elif preconditioning == 'none':
        P = np.eye(len(cluster_avg_expressions))
        x = cp.Variable((1,len(cluster_avg_expressions)), nonneg = True)
        constraint = [x >= 0]
    elif preconditioning == 'exact':
        P = sqrtm(cluster_avg_expressions@cluster_avg_expressions.T)
        cluster_avg_expressions = P@cluster_avg_expressions
        x = cp.Variable((1,len(cluster_avg_expressions)))
        constraint = [x@P >= 0]
    return P, cluster_avg_expressions, x, constraint

def KL_objective(cluster_avg_expressions, bulk_gene_expressions, flat_cluster_covs, bulk_cov, weight, solver = cp.SCS):
    
    Sigma_2_inv = np.linalg.inv(bulk_cov + .001*np.eye(len(bulk_gene_expressions[0])))
    #print((Sigma_2_inv==Sigma_2_inv.T).all())
    #Sigma_2_inv = (Sigma_2_inv + Sigma_2_inv.T)/2
    x = cp.Variable((1,len(cluster_avg_expressions)), nonneg = True)
    predicted_expression = cp.Variable((len(bulk_gene_expressions),len(bulk_gene_expressions[0])))
    predicted_covariance = cp.Variable((len(bulk_gene_expressions[0]), len(bulk_gene_expressions[0])))
    error_in_expressions = cp.mean(predicted_expression - bulk_gene_expressions, axis = 0)
    try:
        objective = (1-weight)*(0*len(bulk_gene_expressions)+ 1)*cp.quad_form(error_in_expressions, Sigma_2_inv)\
                    + weight*.5*(cp.trace(Sigma_2_inv @ predicted_covariance) - cp.log_det(predicted_covariance))
    
        constraints = [predicted_expression == np.ones((len(bulk_gene_expressions),1))@x@cluster_avg_expressions,\
                      predicted_covariance == cp.reshape(x@flat_cluster_covs,(len(bulk_gene_expressions[0]),\
                                                                              len(bulk_gene_expressions[0])))]
    
    
        prob = cp.Problem(cp.Minimize(objective), constraints)
    
        
        result = prob.solve(solver = solver)
        N = sum(x.value[0])
        alpha = x.value[0]/N
    
        return alpha, N, result
    except:
        return None, None, None
    
    
'''Notes 3/13

1. check convergences
2. check convergence to the same thing for different initializations
3. when learning missing cluster profiles, renormailize the alphas
4. add a penalty: min_{alpha >=0, v_missing}|(observed - alpha*v|^2 + lambda*sum alpha_i + gamma*sum |v_i|^2 (lambda and gamma very small)
5. alternatively, minimize O - av subject to sum alpha_i <= lambda, sum_{i  = unknown} ||v_i||^2 <= N_unknown*|v_i|^2_i = known
6. understand the relationship between v_missing and v_missing ground truth

0. change set up of toy model to make all "gene expression counts" non-negative
''' 


"""For Katie, 4/12: This function should take a prediction algorithm as an argument and use it in estimating alpha and the missing cluster profiles. Through each iteration of the alternating algorithm, keep track of the current alpha estimate and current missing cluster profiles estimate. 

In order to check convergence, make a plot of log(|(alpha, missing_profiles)_{i+1} -(alpha, missing_profiles)_{i}|) versus i, and check if this plot has a very negative tangent line for sufficiently large i.
"""

def missing_prototype(num_missing_cluster_refs, cluster_avg_expressions, bulk_gene_expressions, num_iters=5, seed = 1, alpha_weight = .0001, R_weight = .0001, check_convergence = False, nonneg = False):
    """
    num_missing_cluster_refs: int
    the number of clusters not represented in the single cell reference data
    
    cluster_avg_expressions: m X n array
    m = number of clusters observed in single cell reference data
    n = number of genes observed in single cell reference data
    
    bulk_gene_expressions: k X n array
    k = number of bulk samples
    n = number of genes observed in bulk sample expressions (= number of genes observed in single cell reference data)
    
    num_iters: int
    number of iterations through the alternating optimization algorithm (default set to 5, but this is not necessarily optimal)
    """
    
    
    if check_convergence:
        alphas = defaultdict(int)
        Rs = defaultdict(int)
        Ns = defaultdict(int)
        alpha_results = defaultdict(int)
        R_results = defaultdict(int)
        
    
    random.seed(seed)
    np.random.seed(seed)
    
    #create random cluster profiles for initial guess of cell type fractions
    C = math.sqrt(np.sum(np.square(bulk_gene_expressions)))
    random_cluster_profiles = np.zeros((num_missing_cluster_refs,len(cluster_avg_expressions[0])))
    
    #populate random_cluster_profiles with entries from known cluster profiles
    for i in range(num_missing_cluster_refs):
        for j in range(len(cluster_avg_expressions[0])):
            k = random.randint(0,len(cluster_avg_expressions)-1)
            random_cluster_profiles[i,j] = cluster_avg_expressions[k,j]
    
    #incorporate random_cluster_profiles into known data
    pseudo_cluster_profiles = np.vstack([random_cluster_profiles, cluster_avg_expressions])

    
    alpha = np.zeros((1,len(pseudo_cluster_profiles)))
                     
    for i in range(num_iters):
        
        #guess for cell type fractions
        
        """Make this block more modular, e.g make this block:
        
        x = cp.Variable((1,len(pseudo_cluster_profiles)), nonneg = True)
        objective = cp.sum_squares(np.ones((len(bulk_gene_expressions),1))@x@pseudo_cluster_profiles-np.array(bulk_gene_expressions)/C) + alpha_weight*cp.sum(x) + R
        
        constraints = []
        if alpha_constraints and i > 1:
            constraints = [cp.abs(cp.sum(x)) <= 700]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        
        alpha_result = prob.solve(solver = cp.GUROBI)
        N = sum(x.value[0])
        #print(x.value)
        #print('population size: ', N)
        alpha = x.value
        
        look like this:
        
        alpha, N, result = input_prediction_algorithm(pseudo_cluster_profiles, bulk_gene_expressions, etc...) 
        
        
        
        """

        if i + 1 ==num_iters:
            if check_convergence:
                return alphas, Rs, Ns, alpha_results, R_results
            else:
                return sum(x.value[0]), x.value[0]/sum(x.value[0]), pseudo_cluster_profiles
        
        #alternating algorithm to solve for cluster type profiles one at a time
        """R should be renamed 'missing_cluster_profiles_estimates'"""
        for j in range(num_missing_cluster_refs):
            
            #R_pseudo_cluster_profiles has a variable in it
            #create cluster_j profile variable
            R = cp.Variable((1,len(cluster_avg_expressions[0])), nonneg= nonneg)
            if j == 0:
                B = pseudo_cluster_profiles[1:]
                R_pseudo_cluster_profiles = cp.vstack([R,B])
            else:
                A = pseudo_cluster_profiles[:j]
                B = pseudo_cluster_profiles[j+1:]
                R_pseudo_cluster_profiles = cp.vstack([A,R,B])
                
            #optimize cluster_j profile   
            objective = \
                cp.sum_squares(np.ones((len(bulk_gene_expressions),1))@alpha@R_pseudo_cluster_profiles-np.array(bulk_gene_expressions)/C) + R_weight*cp.sum_squares(R)
            constraints = [cp.sum_squares(R) <= max([np.sum(np.square(profile)) for profile in cluster_avg_expressions])]
            prob = cp.Problem(cp.Minimize(objective), constraints)
            R_result = prob.solve(solver = solver)
            
            
            #update reference after optimization
            R = R.value
            
            if j == 0:
                B = pseudo_cluster_profiles[1:]
                pseudo_cluster_profiles = np.vstack([R,B])
            else:
                A = pseudo_cluster_profiles[:j]
                B = pseudo_cluster_profiles[j+1:]
                pseudo_cluster_profiles = np.vstack([A,R,B])
          
            
            
            if check_convergence:
                alphas[i] = alpha
                Rs[i] = R
                Ns[i] = N
                alpha_results[i] = alpha_result
                R_results[i] = R_result