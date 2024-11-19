import numpy as np
import tqdm
class OverICA_EM(): # after "An EM algorithm for learning sparse and overcomplete representations".
    # TODO: needs to be adjusted if the source distribution changes
    def __init__(self, iterations= 100, tolerance= 1e-3, learning_rate= 1e-3, beta = 10, random_seed = 42, true_A = None, init_range = [-3,3]):
        self.iterations = iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.beta = beta
        self.is_converged = False
        self.random_seed = random_seed
        self.true_A = true_A
        self.init_range = init_range
      
    def fit(self, X, J, noise_params = {'mean': 0, 'std': .1}):
        "X is observed data matrix n * I. J is the number of independent components"
        np.random.seed(self.random_seed)
        self.noise_mean = noise_params['mean']
        self.noise_std = noise_params['std']
        self.sigma_matrix = np.eye(X.shape[1]) * self.noise_std**2
        self.sigma_matrix_inv = np.linalg.inv(self.sigma_matrix)
        self.X = X
        self.J = J
        self.I = X.shape[1]
        if self.true_A is not None:
            print("Initializing A with true A + noise")
            self.A = self.true_A + np.random.normal(0, 1, (self.I, self.J))
        else:
            print("Initializing A randomly")
            self.A = np.random.uniform(low = self.init_range[0], high = self.init_range[1], size = (self.I, self.J))
        # Initilize the source matrix whit pseudo inverse
        self.Signals = (np.linalg.pinv(self.A) @ X.T).T
        self._normalize_columns()
        for i in tqdm.tqdm(range(self.iterations)):
            
            self._update_signals()
            self._update_mixing_matrix()
            self._normalize_columns()
      

    def _update_mixing_matrix(self):
        temp1 = np.zeros((self.I, self.J))
        temp2 = np.zeros((self.J, self.J))
        for i in range(self.X.shape[0]):
            temp1 += np.outer(self.X[i,:], self.Signals[i,:])
            hes = self._hess_prior(self.Signals[i,:])
            hes_inv = np.linalg.inv(hes)    
            M = self._M_mat(hes_inv)
            temp2 += hes_inv - hes_inv @ self.A.T @ M @ self.A @ hes_inv + np.outer(self.Signals[i,:] ,self.Signals[i,:])

        self.A = temp1 @ np.linalg.inv(temp2)

    
    def _normalize_columns(self):
        self.A =  self.A / np.linalg.norm(self.A, axis= 0)
    
    def _update_signals(self, ):
        update_step = self.learning_rate * self._gradient_s()
        if np.linalg.norm(update_step) < self.tolerance:
            self.is_converged = True
        else:
            self.Signals += update_step
    
    def _gradient_s(self):
        dev =  self.A.T @ self.sigma_matrix_inv @ (self.X.T- self.A @ self.Signals.T)
        grad_prior = -np.tanh(self.beta*self.Signals) # this is an approximation, see "Learning Overcomplete Representations"  
        return dev.T + grad_prior
    
    def _hess_prior(self, s):
        hes = np.eye(self.J)
        hes *= -self.beta * self._sech(self.beta*s)**2
        return -hes

    def _M_mat(self, hes_inv):
        return np.linalg.inv(self.sigma_matrix + self.A @ hes_inv @ self.A.T)
    
    def _sech(self, x):
        temp=  1/np.cosh(x)
        return temp
    

class VarEM():
    def __init__(self, max_iter=100, update_sigma = False, random_seed = 42, true_A = None, init_range = [-3,3], tol = .02):
        self.max_iter = max_iter
        self.update_sigma = update_sigma
        self.random_seed = random_seed
        self.true_A = true_A
        self.init_range = init_range
        self.tol = tol

    def fit(self, X, J,   noise_params = {'mean': 0, 'std': 1}):
        np.random.seed(self.random_seed)
        self.X = X
        self.J = J
        self.n = X.shape[0]
        self.I = X.shape[1]
        if self.true_A is not None:
            print("Initializing A with true A + noise")
            self.A = self.true_A + np.random.normal(0, 1, (self.I, self.J))
        else:
            print("Initializing A randomly")
            self.A = np.random.uniform(low = self.init_range[0] , high = self.init_range[1], size = (self.I, self.J))
        self.data_cov = np.cov(X.T, bias=True)
        self.xi = np.random.rand(self.n, self.J)
        self.noise_mean = noise_params['mean']
        self.noise_std = noise_params['std']
        if self.update_sigma:
            self.sigma_matrix = np.cov(X.T, bias=True)
        else:
            self.sigma_matrix = np.eye(X.shape[1]) * self.noise_std**2
        self.sigma_matrix_inv = np.linalg.inv(self.sigma_matrix)
        self.Signals = np.zeros((self.n, self.J))
        progress_bar = tqdm.tqdm(range(self.max_iter))

        for i in progress_bar:
            diff = self.update_A()
            if diff < self.tol:
                print(f"Converged after {i} iterations with diff = {diff:.4f}")
                break
            progress_bar.set_description(f"Diff: {diff:.4f}")
            

        
        print("Estimating the signals")

        self._estimate_signals()
        
    def _update_sigma(self, A_new):
        temp = np.zeros((self.J, self.I))
        for i in range(self.n):
            omega_i = self._omega_mat(i)
            M_i = self._M_mat(omega_i)
            x_outer = np.outer(self.X[i,:], self.X[i,:])
            temp += omega_i @ self.A.T @ M_i @ x_outer
        self.sigma_matrix = self.data_cov  - A_new @ temp/self.n
        self.sigma_matrix_inv = np.linalg.inv(self.sigma_matrix)
        
        

    def update_A(self):
        temp1 = np.zeros((self.I, self.J))
        temp2 = np.zeros((self.J, self.J))
        for i in range(self.n):
            omega_i = self._omega_mat(i)
            M_i = self._M_mat(omega_i)
            x_outer = np.outer(self.X[i,:], self.X[i,:])
            temp1 += x_outer @ M_i.T @ self.A @ omega_i
            temp2 += omega_i @ (np.eye(self.J) - self.A.T @ M_i @ (np.eye(self.I) - x_outer @ M_i.T) @ self.A @ omega_i)
            self._update_xi(i, M_i, omega_i, x_outer)
        A_new = temp1 @ np.linalg.inv(temp2)
        if self.update_sigma:
            self._update_sigma(A_new)
        # calculate the difference between the old and new A
        diff = np.linalg.norm(self.A - A_new, ord='fro')
        self.A = A_new
        return diff
    
    def _M_mat(self, omega_i): # dim I x I

        return np.linalg.inv(self.A @ omega_i @ self.A.T + self.sigma_matrix)
    
    def _omega_mat(self, i): # dim J x J
        return np.abs(np.diag(self.xi[i,:]))

    def _update_xi(self, i, M_i, omega_i, x_outer):
        self.xi[i] = np.diag(omega_i @ (np.eye(self.J) - self.A.T @ M_i @ (np.eye(self.I) - x_outer @ M_i.T) @ self.A @ omega_i))

    def _estimate_signals(self):
        for i in  tqdm.tqdm(range(self.n)):
            omega_i = self._omega_mat(i)
            temp1 = np.linalg.inv(self.A.T @ self.sigma_matrix_inv @ self.A +np.linalg.inv(omega_i))
            temp2 = self.A.T @ self.sigma_matrix_inv @ self.X[i,:]
            self.Signals[i,:] = temp1 @ temp2

