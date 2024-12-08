
import scipy as sp
import numpy as np
import itertools
def likelihood_score(signal_true, signal_est, true_paras = {'mean': 0, 'scale': 1}, normalize = True):
    """the lower the better"""
    if normalize:
        signal_est = signal_est/np.std(signal_est, axis = 0)
        signal_true = signal_true/np.std(signal_true, axis = 0)
    # true score 
    ll = sp.stats.laplace.logpdf(signal_true, loc = true_paras['mean'], scale = true_paras['scale'])
    ll = np.sum(ll)
    # estimated score
    ll_est = sp.stats.laplace.logpdf(signal_est, loc = true_paras['mean'], scale = true_paras['scale'])
    ll_est = np.sum(ll_est)
    return -(ll_est-ll)



def f_score(A_true, A_hat):
    # go through all permutations of the columns of A_hat and calculate the f_score for each permutation normalize the columns
    # return the best f_score and the corresponding permutation 
    smallest_f_score = np.inf
    best_permutation = None
    A_hat = A_hat/np.linalg.norm(A_hat, axis = 0)
    A_true = A_true/np.linalg.norm(A_true, axis = 0)
    for perm in itertools.permutations(range(A_hat.shape[1])):
        A_hat_perm = A_hat[:,perm]
        f_score = sp.linalg.norm((A_true - A_hat_perm), 'fro')**2
        if f_score < smallest_f_score:
            smallest_f_score = f_score
            best_permutation = perm
    return best_permutation, smallest_f_score/np.linalg.norm(A_true, 'fro')**2

def mean_squared_error(signal_true, signal_est, normalize = True):
    # set the scale of the estimated signal to the true signal column wise
    if normalize:
        signal_est = signal_est/np.std(signal_est, axis = 0)
        signal_true = signal_true/np.std(signal_true, axis = 0)
    return np.mean((signal_true - signal_est)**2)