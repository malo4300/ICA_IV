from models.ICA_EM import *
from models.dgp import * 
import pandas as pd


def sim(mode, B, iter, dgp_):
    
    data = dgp_(noise_dict= {"loc" : 0, "scale" : 0}, prior= {"loc" : 0, "scale" : 1/np.sqrt(2)})
    for i in tqdm.tqdm(range(B)):
        data.generate_data(n,I, J, random_state=i)
        #est = VarEM(update_sigma=False, true_A=None, max_iter = iter, random_seed= i)
        #est.fit(data.data_observed,J, noise_params= {"mean" : 0, "std" : 1}, progress_bar=False)
        pd.DataFrame(data.mixing_matrix_observed).to_csv(f"sim_data/mixing_matrix/true_mixing_{i}.csv", index = False)
        #pd.DataFrame(est.A).to_csv(f"sim_data_VarEM/mixing_matrix/estimated_mixing_{i}.csv", index = False)
        #pd.DataFrame(est.Signals).to_csv(f"sim_data_VarEM/signals/estimated_signals_{i}.csv", index = False)
        pd.DataFrame(data.signals).to_csv(f"sim_data/signals/true_signals_{i}.csv", index = False)
        pd.DataFrame(data.data_observed).to_csv(f"sim_data/data/data_obs_{i}.csv", index = False)




class dgp_extended():# build a simple class with enough controll that the IV tgest should work and that has the most simple independence testing 
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


if __name__ == "__main__":
    J = 6
    I = J-1
    n = 10000
    sim("lower_triangular", 100, 100, dgp)