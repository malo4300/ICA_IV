import numpy as np
from time import time
import scipy.sparse.linalg as scl
from scipy.sparse import diags
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

# Define the default options in a dictionary
def set_default_opts():
    opts = {
        'sub': 'quad',
        'sdp': 'clust'
    }
    return opts

def check_opts(opts):
  
    if type(opts) != dict:
        opts = set_default_opts()

    if 'sub' not in opts:
        opts['sub'] = 'gencov'

    if 'sdp' not in opts:
        opts['sdp'] = 'semiada'

    if 'ctype' in opts:
        ctype = opts['ctype']
        if ctype != 'h' and ctype != 'km':
            print('The opts.ctype value is changed to h')
            opts['ctype'] = 'h'

    if 'sub' in opts:
        sub = opts['sub']
        if sub != 'quad' and sub != 'gencov':
            print('The opts.sub value is changed to gencov')
            opts['sub'] = 'gencov'

    if 'sdp' in opts:
        sdp = opts['sdp']
        if sdp != 'ada' and sdp != 'semiada' and sdp != 'clust':
            print('The opts.sdp value is changed to semiada')
            opts['sdp'] = 'semiada'

    return opts

# Main function
def overica(X, k, opts=None):
    """ X is the p-times-n data matrix with observations in columns
%   k is the desired latent dimension (k < p^2/4 for our guarantees to hold)"""
    ds_est, Ds_est = None, None   # initial values

    if opts is None:
        opts = set_default_opts()
    else:
        opts = check_opts(opts)

    globtt = time()

    if opts['sub'] == 'quad':
        print('Computing quadricovariance')
        tt = time()
        C = quadricov(X)
        print(time() - tt)

    if opts['sub'] == 'gencov':
        print('Computing generalized covariances')
        tt = time()
        s = 5 * k
        t = 0.05 / np.sqrt(np.max(np.abs(np.cov(X, rowvar=False))))
        if 's' in opts:
            s = opts['s'] * k
        if 't' in opts:
            t = opts['t']
        C = estimate_gencovs(X, s, t)
        print(time() - tt)

    print(time() - globtt)
    cum_time = time() - globtt

    print('Computing SVD')
    CU, _, _ = scl.svds(C, k)
    Hs = CU[:, :k]
    print(time() - globtt)
    svd_time = time() - globtt - cum_time

    if opts['sdp'] == 'clust':
        print('Deflation via clustering')
        if 'ctype' in opts:
            ds_est, Ds_est = sdp_cluster(Hs, k, opts['ctype'])
        else:
            ds_est, Ds_est = sdp_cluster(Hs, k)
    #elif opts['sdp'] == 'ada':
    #    print('Adaptive deflation')
    #    ds_est, Ds_est = sdp_adaptive(Hs, k)
    #elif opts['sdp'] == 'semiada':
    #    print('Semiadaptive deflation')
    #    if 'ctype' in opts:
    #        ds_est, Ds_est = sdp_semiada(Hs, k, opts['ctype'])
    #    else:
    #        ds_est, Ds_est = sdp_semiada(Hs, k)

    print(time() - globtt)
    sdp_time = time() - globtt - cum_time - svd_time

    total_time = time() - globtt

    times = {
        'total_time': total_time,
        'cum_time': cum_time,
        'svd_time': svd_time,
        'sdp_time': sdp_time
    }

    return ds_est, Ds_est, times, Hs

def quadricov(X):
    # Zero-mean
    X = X - np.mean(X, axis=1, keepdims=True)
    
    Q = quadricov_in(X)

    # Subtracts the diagonal and adds the result to Q
    Q = (Q - np.diag(np.diag(Q))) + Q.T

    return Q

import numpy as np

def quadricov_in(X):
    # shape of X is p x n
    p,n = X.shape

    # Initialize C and Q
    C = np.zeros((p, p))
    Q = np.zeros((p*p, p*p))

    # Compute C and fill the temporary matrix
    for i in range(n):
        tau = np.outer(X[:,i], X[:,i])
        C += tau
        temp = tau.flatten()

        # Add the product of the corresponding elements to Q
        for a in range(p*p):
            for b in range(a, p*p):
                Q[a, b] += temp[a]*temp[b]

    # Compute average
    C /= n
    for a in range(p*p):
        for b in range(a, p*p):
            Q[a, b] /= n
    
    # Final step
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    if c + d*p >= a + b*p:
                        Q[a + b*p, c + d*p] -= C[a, b]*C[c, d] + C[a, c]*C[b, d] + C[a, d]*C[b, c]

    return Q


# Make sure to define 'gencov' and 'choose_t' functions in Python

def estimate_gencovs(X, s, t0):
    p, n = X.shape
    t = t0
    C = np.zeros((p**2, s))
    G0 = gencov(X, np.zeros((p, 1))).flatten()

    for i in range(s):
        omega = np.random.randn(p, 1)
        if t0 == -1:
            t = 0.05 / np.sqrt(np.max(np.abs(np.cov(X, rowvar=False)))) # overwritten missing choose_t function
        omega = t * omega
        Gi = gencov(X, omega)
        C[:, i] = Gi.flatten() - G0

    return C


def gencov(X, omega):
    if np.size(omega) == 1:
        p = X.shape[0]
        omega = omega * np.ones(p)/p

    n = X.shape[1]
    proj = np.dot(X.T, omega)
    eproj = np.exp(proj)
    
    # genexp
    Eomega = np.dot(X, eproj) / np.sum(eproj)
    C = np.dot(X, diags(eproj.flatten()).dot(X.T))
    C = C / np.sum(eproj)
    C = C - np.outer(Eomega, Eomega)
    C = C.flatten()

    return C

def sdp_cluster(Hs, k, ctype=None):
    if ctype is None:
        ctype = 'h'
    elif ctype != 'h' and ctype != 'km':
        ctype = 'h'

    p = int(np.sqrt(Hs.shape[0]))
    nclust = 3 * k

    _, Fs = extract_basis(Hs, k)
    Dss = np.zeros((p**2, nclust))
    
    for irep in range(nclust):
        u = np.random.randn(p, 1)
        u = u / np.linalg.norm(u)
        G = np.outer(u, u.T)
        D = majorize_minimize(G, Fs)
        Dss[:, irep] = D.flatten()

    Ds_est = cluster_Dss(Dss, k, ctype)
    ds_est = approx_ds_from_Ds(Ds_est)

    return ds_est, Ds_est

import numpy as np

def extract_basis(Es, k):
    Q, _ = np.linalg.qr(Es)
    Esbasis = Q[:, :k]
    Fsbasis = Q[:, k:]

    return Esbasis, Fsbasis

def extract_largest_eigenvector(D):
    # Copyright: Anastasia Podosinnikova 2019
    D = (D + np.transpose(D)) / 2
    e, u = np.linalg.eig(D)
    max_index = np.argmax(np.abs(e))
    u = np.real(u[:, max_index])
    e = e[max_index]
    return u, e

def majorize_minimize(G, Fs):
    p = int(np.sqrt(Fs.shape[0]))
    Dinit = np.eye(p) / p
    D = Dinit
    mu = 5
    maxiter = 100
    tolerance = 1e-3
    nmmmax = 100
  
    iter = 1
    while np.linalg.norm(D, 'fro') < 1:
        u = extract_largest_eigenvector(G)[0]
        Ginit = np.outer(u, u)
        D = solve_relaxation_mezcal_approx_fista(Fs, Ginit, mu, Dinit, maxiter, tolerance)
        G = (D + np.transpose(D))/2
        iter = iter + 1
        if iter > nmmmax:
            break
    return D


def solve_relaxation_mezcal_approx_fista(Fsbasis, G, mu, Dinit, maxiter, tolerance):
    # Copyright: Anastasia Podosinnikova 2019

    d, k = Fsbasis.shape
    d = int(np.sqrt(d))

    D = Dinit
    E = D
    L = mu
    t = 1
    primal_vals = np.zeros(maxiter)
    dual_vals = np.zeros(maxiter)
    
    for iter in range(maxiter):
        temp = Fsbasis.T @ E.flatten()
        grad = -G + mu * np.reshape(Fsbasis @ temp, (d, d))
        E = E - (1/L) * (grad)
        tnew = .5 * (1 + np.sqrt(1 + 4 * t ** 2))
        e, u = np.linalg.eig((E + E.T) / 2)
        u = np.real(u)
        e = np.real(np.diag(e))
      
        #eproj = proj_simplex(e.flatten()) # fixed format
        eproj = proj_simplex(np.diag(e))
        Dnew = u @ np.diag(eproj) @ u.T
        D = (D + D.T) / 2
        E = Dnew + (t - 1) / tnew * (Dnew - D)
        D = Dnew
        t = tnew

        if iter % 10 == 0:
            temp = Fsbasis.T @ D.flatten()
            grad = -G + mu * np.reshape(Fsbasis @ temp, (d, d))
            primal_vals[iter] = -np.sum(G * D) + mu/2 * np.sum(temp ** 2)
            dual_vals[iter] = np.min(np.real(np.linalg.eigvals(grad))) - (mu/2) * np.sum(temp ** 2)

            if (primal_vals[iter] - np.max(dual_vals[::10])) < tolerance:
                break

    return D


def proj_simplex(v):
    # PROJECTONTOSIMPLEX Projects point onto simplex of specified radius.
    #
    # w = ProjectOntoSimplex(v, b) returns the vector w which is the solution
    #   to the following constrained minimization problem:
    #
    #    min   ||w - v||_2
    #    s.t.  sum(w) <= b, w >= 0.
    #
    #   That is, performs Euclidean projection of v to the positive simplex of
    #   radius b.
    #
    # Author: John Duchi (jduchi@cs.berkeley.edu)

    v = v * (v > 0)
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.where(u > (sv - 1) / np.arange(1, len(u) + 1))[0][-1]
    theta = np.maximum(0, (sv[rho] - 1) / (rho + 1)) # Adding 1 to rho as Python's indexing starts from 0
    w = np.maximum(v - theta, 0)

    return w



def cluster_Dss(Dss, k, ctype):
    # Copyright: Anastasia Podosinnikova 2019

    # since the atoms are scaling (hence, sign) invariant
    # we first align them all to point to the same direction
  
    D1 = Dss[:, 0]
    for i in range(1, Dss.shape[1]):
        Di = Dss[:, i]
        Dss[:, i] = np.sign(np.dot(D1.T, Di)) * Di

    Ds_est = extract_clusters(Dss, k, ctype)
    return Ds_est
  
def extract_clusters(DD, nclust, ctype):
    p = int(np.sqrt(DD.shape[0]))
  
    if ctype == 'h': # hierarchical clustering
        cc = fcluster(linkage(DD.T, method='average'), nclust, criterion='maxclust')
    elif ctype == 'km': # k-means++
        kmeans = KMeans(n_clusters=nclust, random_state=0).fit(DD.T)
        cc = kmeans.labels_
  
    Ds_temp = np.zeros((p**2, nclust))
   
    for i in range(1, nclust+1):
  
        DDi = DD[:, cc==i]
        Ds_temp[:, i-1] = DDi[:, 0] #mean(DDi,2);
  
    return Ds_temp




def approx_ds_from_Ds(Ds): 
    k = Ds.shape[1]
    p = int(np.sqrt(Ds.shape[0]))
    ds = np.zeros((p, k))
    eigmaxes = np.zeros((k, 1))
  
    for i in range(k):
        D = np.reshape(Ds[:, i], (p, p))
        u, e = extract_largest_eigenvector(D)
        eigmaxes[i] = e
        ds[:, i] = u

    return ds, eigmaxes