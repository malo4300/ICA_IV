import numpy as np

def fourier_pca1(X, k):
    """
    X: pxN, p is dimension, N # samples
    ***********************************************************************
    """
    p = X.shape[0]

    u1 = np.random.randn(p, 1) / np.sqrt(p)
    u2 = np.random.randn(p, 1) / np.sqrt(p)

    Q1 = genquadricov(X, u1)
    Q2 = genquadricov(X, u2)

    W, S, _ = np.linalg.svd(Q1)
    inds = np.argsort(S)[::-1]  # Descending order sort

    W = W[:, inds]


    W = W[:, :k]
  
    q1 = W.T @ Q1 @ W
    q2 = W.T @ Q2 @ W

    M = np.linalg.solve(q2, q1)
    _, V = np.linalg.eig(M)

    C = W @ V
    # THEIR DEFINITION OF C IS NOT UNIQUE -> only up to sign, but that's not important
    for j in range(k):
        c = C[:, j]

        a = c.real
        b = c.imag
        theta = np.arctan(-(2 * np.sum(a * b)) / np.sum(a ** 2 - b ** 2)) / 2
        while theta < 0:
            theta += np.pi
        while theta > 2 * np.pi:
            theta -= np.pi

        temp = (np.exp(1j * theta) * c).real
        C[:, j] = temp / np.linalg.norm(temp)

    ds_est = np.zeros((p, k))
    for j in range(k):
        c = C[:, j]
        v, s, _ = np.linalg.svd(c.reshape(p, p))
        inds = np.argsort(s)[::-1]
        v = v[:, inds]
        ds_est[:, j] = v[:, 0]
    return ds_est


def genquadricov(X, u):
    """
    X: pxN, p dimension, N number of samples
    u: p, processing point
    Q: p*p x p*p, matricized 4-th order generalized cumulant

    The matricization rule:
    Q( (i1-1)*p + i2, (i3-1)*p + i4 ) = CUM(i1,i2,i3,i4);
    (i.e. row-wise for the the first two and the last two dimensions)

    THE OUTPUT IS A COMPLEX MATRIX
    ************************************************%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    p, N = X.shape

    Xu = X.T @ u
    expXu = np.exp(1j*Xu)
    M0 = np.sum(expXu) / N

    M1 = (X @ expXu) / (N*M0)

    XC = X - np.outer(M1, np.ones(N))

    M2 = np.zeros((p,p), dtype=complex)
    M4 = np.zeros((p*p, p*p), dtype=complex)
    temp = np.zeros((p,p), dtype=complex)

    for n in range(N):
        xn = XC[:, n]
        temp = np.outer(xn, np.conj(xn.T))
        M2 += expXu[n] * temp
  
        M4 += np.outer(expXu[n] * temp.flatten(), np.conj(temp.flatten()))

    M2 = M2 / (N*M0)
    M4 = M4 / (N*M0)
    Q = np.zeros((p*p, p*p), dtype=complex)

    for i3 in range(p):
        for i4 in range(p):
            icol = i3*p + i4
            temp = M4[:, icol].reshape(p, p)
            temp = temp - M2[i3, i4]*M2 - np.outer(M2[:, i3], M2[i4, :]) - np.outer(M2[:, i4], M2[i3, :])
            Q[:, icol] = temp.flatten()

    return Q

import numpy as np

def fourier_pca2(X, k):
    """
    X: pxN, p is dimension, N # samples
    ***********************************************************************
    """
    p = X.shape[0]

    u = np.random.randn(p, 1)
    u = u / np.linalg.norm(u)

    Q = genquadricov(X, u)
    Q1 = Q.real
    Q2 = Q.imag

    W, S, _ = np.linalg.svd(Q1)
    inds = np.argsort(S)[::-1]  # Descend sorting
    W = W[:, inds]
    W = W[:, :k]
    
    q1 = W.T @ Q1 @ W
    q2 = W.T @ Q2 @ W
    
    M = np.linalg.solve(q2, q1)
    _, V = np.linalg.eig(M)

    C = W @ V
    
    for j in range(k):
        c = C[:, j]
        a = c.real; b = c.imag
        theta = np.arctan(-(2 * np.sum(a * b)) / np.sum(a ** 2 - b ** 2)) / 2
        while theta < 0: theta += np.pi
        while theta > 2 * np.pi: theta -= np.pi
        
        temp = (np.exp(1j * theta) * c).real
        C[:, j] = temp / np.linalg.norm(temp)
    
    ds = np.zeros((p, k))
    for j in range(k):
        print(j)
        c = C[:, j]
        v, s, _ = np.linalg.svd(c.reshape(p, p))
        inds = np.argsort(s)[::-1] 
        v = v[:, inds]
        ds[:, j] = v[:, 0]
    
    return ds