import typing
import numpy as np

class dgp():
    def __init__(self, noise_dict = {"loc" : 0, "scale" : 0}, prior = False):
        if prior:
            self.loc = prior['loc']
            self.scale = prior['scale']
        else:
            self.loc = 0
            self.scale = 1/np.sqrt(2)
        self.noise = noise_dict

    def generate_data(self, n: int, I: int, J: int, random_state: int = 0,):
        np.random.seed(random_state)
       
        self._generate_dag(J)
        self._generate_coefficients_matrix(J)
        self._generate_signals(n, J)

        self._generate_mixing_matrix()

        # generate data
        self.signals = np.zeros((n, J))
        self.signals[:, 0] = self.signal_T
        self.signals[:, 1] = self.signal_Y
        self.signals[:, 2] = self.signal_U
        if self.no_controls > 0:
            self.signals[:, 3:] = self.signal_X
        self.data = np.zeros((n, J))
        for j in range(J):
            self.data[:, j] = self.mixing_matrix[j, :] @ self.signals.T

        if self.noise:
            self.data += np.random.normal(loc=self.noise['loc'], scale=self.noise["scale"], size=(n, J))
        
        # the treatment effect is in 1,0
        self.treatment_effect = self.coef_mat[1, 0]
        # observed data is missing the confounder
        obs_indices = [i for i in range(J) if i != 2]
        self.data_observed = self.data[:, obs_indices]
        self.mixing_matrix_observed = self.mixing_matrix[obs_indices, :]

    
    def _generate_dag(self, J: int):
        self.adj_matrx = np.zeros((J, J)) # adjacency matrix at least 3 nodes
        # TODO: make this more general and allow for edges between controls 
        # w.l.o.g. we can assume that the first node is the treatment and the second node is the outcome
        self.adj_matrx[0,2] = 1 # confounder -> treatment
        self.adj_matrx[1,2] = 1 # confounder -> outcome
        self.adj_matrx[1,0] = 1 # treatment -> outcome
        # build the adjacency matrix for controls
        self.no_controls = J - 3
        for i in range(3, J):
            self.adj_matrx[1, i] = 1
        
        # introduce additional edges
        if J > 3:
            self.adj_matrx[0, 3] = 1 # control -> treatment
        if J > 4:
            self.adj_matrx[4, 3] = 1 # control1 -> control
            self.adj_matrx[J-2, J-1] = 1 # control1 -> outcome



    def _generate_coefficients_matrix(self, J: int):
        coef_of_edges = np.random.uniform(low=-3, high=3, size=(J, J))
        self.coef_mat = np.multiply(self.adj_matrx, coef_of_edges) 

    def _generate_signals(self, n: int, J):
        self.signal_T = np.random.laplace(loc=self.loc, scale=self.scale, size=n)
        self.signal_Y = np.random.laplace(loc=self.loc, scale=self.scale, size=n)
        self.signal_U = np.random.laplace(loc=self.loc, scale=self.scale, size=n)
        #TODO: make this more general and with controls
        if self.no_controls > 0:
            self.signal_X = np.random.laplace(loc=self.loc, scale=self.scale, size=(n, self.no_controls))

    def _generate_mixing_matrix(self):
        # by construction, the noise has coefficient 1 for its own node
        I = np.eye(self.coef_mat.shape[0])  # Identity matrix of size J
        self.mixing_matrix =  np.linalg.inv(I - self.coef_mat)
