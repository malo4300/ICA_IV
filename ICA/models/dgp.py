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

    def generate_data(self, n: int, I: int, J: int, random_state: int = 0, init_range = [-3,3]):
        np.random.seed(random_state)
        self.no_controls = J - 3
        self._generate_dag(J)
        self._generate_coefficients_matrix(J, init_range = init_range)
        self._generate_signals(n, J)

        self._generate_mixing_matrix()

        # generate data
        self.data = np.zeros((n, J))
        self.signals = np.zeros((n, J))


        self.signals[:, 0] = self.signal_U
        if self.no_controls > 0:
            self.signals[:, 1:(self.no_controls+1)] = self.signal_X
        self.signals[:, J-2] = self.signal_T
        self.signals[:, J-1] = self.signal_Y

        for j in range(J):
            self.data[:, j] = self.mixing_matrix[j, :] @ self.signals.T

        if self.noise:
            self.data += np.random.normal(loc=self.noise['loc'], scale=self.noise["scale"], size=(n, J))
        
        self.treatment_effect = self.coef_mat[-1, J-2]        # observed data is missing the confounder
        self.mixing_matrix_observed = self.mixing_matrix[1:J, :]
        self.data_observed = self.data[:, 1:J]
    
    def _generate_dag(self, J: int):
        self.adj_matrx = np.zeros((J, J)) # adjacency matrix at least 3 nodes
        # TODO: make this more general and allow for edges between controls 

        # w.l.o.g. we can assume that the matrix is in topological order meaning the confounder is the first node then controls then treatment (J-2) and outcome (J-1)
        self.adj_matrx[J-2,0] = 1 # confounder -> treatment
        self.adj_matrx[J-1,0] = 1 # confounder -> outcome
        self.adj_matrx[J-1,J-2] = 1 # treatment -> outcome

        
        if self.no_controls > 0:
            for i in range(1,(self.no_controls+1)):
                self.adj_matrx[J-1, i] = 1
            self.adj_matrx[J-2, 2] = 1 # control2 -> treatment
        if self.no_controls > 2:
            self.adj_matrx[2,1] = 1 # control1 -> control2
            self.adj_matrx[3,1] = 1 # control1 -> control3
        



    def _generate_coefficients_matrix(self, J: int, init_range = [-3,3]):
        coef_of_edges = np.random.uniform(low=init_range[0], high=init_range[1], size=(J, J))
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


class dgp_extended():# build a simple class with enough controll that the IV tgest should work and that has the most simple independence testing 
    def __init__(self, noise_dict = {"loc" : 0, "scale" : 0}, prior = False, level_of_confounding = 1):
        if prior:
            self.loc = prior['loc']
            self.scale = prior['scale']
        else:
            self.loc = 0
            self.scale = 1/np.sqrt(2)
        self.noise = noise_dict
        self.level_of_confounding = level_of_confounding

    def generate_data(self, n: int, I: int, J: int, random_state: int = 0, init_range = [-3,3]):
        np.random.seed(random_state)
        self.no_controls = J - 3
        self._generate_dag(J)
        self._generate_coefficients_matrix(J, init_range = init_range)
        self._generate_signals(n, J)

        self._generate_mixing_matrix()

        # generate data
        self.data = np.zeros((n, J))
        self.signals = np.zeros((n, J))


        self.signals[:, 0] = self.signal_U
        if self.no_controls > 0:
            self.signals[:, 1:(self.no_controls+1)] = self.signal_X
        self.signals[:, J-2] = self.signal_T
        self.signals[:, J-1] = self.signal_Y

        for j in range(J):
            self.data[:, j] = self.mixing_matrix[j, :] @ self.signals.T

        if self.noise:
            self.data += np.random.normal(loc=self.noise['loc'], scale=self.noise["scale"], size=(n, J))
        
        self.treatment_effect = self.coef_mat[-1, J-2]        # observed data is missing the confounder
        self.mixing_matrix_observed = self.mixing_matrix[1:J, :]
        self.data_observed = self.data[:, 1:J]
    
    def _generate_dag(self, J: int):
        self.adj_matrx = np.zeros((J, J)) 

        self.adj_matrx[J-2,0] = 1 # confounder -> treatment
        self.adj_matrx[J-1,0] = 1 # confounder -> outcome
        self.adj_matrx[J-1,J-2] = 1 # treatment -> outcome
        
        if self.no_controls > 0:
            for i in range(1,(self.no_controls+1)):
                self.adj_matrx[J-1, i] = 1 # controll -> outcome
                self.adj_matrx[J-2, i] = 1 # controll  -> treatment
        
        



    def _generate_coefficients_matrix(self, J: int, init_range = [-3,3]):
        coef_of_edges = np.random.uniform(low=init_range[0], high=init_range[1], size=(J, J))
        self.coef_mat = np.multiply(self.adj_matrx, coef_of_edges) 
        self.coef_mat[J-2,0] = self.level_of_confounding*self.coef_mat[J-2,0] 
        self.coef_mat[J-1,0] = self.level_of_confounding*self.coef_mat[J-2,0] 

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