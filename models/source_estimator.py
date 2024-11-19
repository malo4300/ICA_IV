import numpy as np
import cvxpy as cp
class SourceEstimator():
    """Estimate the sources based on paramteric assumption and the noisless mixing model"""
    
    def __init__(self, paras = None, noise = False):
        self.paras = paras
        self.noisy_case = noise
        # TODO: use paraemters to generalize 

    def fit(self, data, A_hat):
        """Fit the model to the data matrix by estimating the signals.
        
        Parameters:
        - data: a numpy array with n rows and I columns representing the data.
        - A_hat: an (I x J) matrix, the estimated mixing matrix.
        """
        if self.noisy_case:
            print("Fitting the model to the data in the noisy case")
            self._fit_noisy(data, A_hat)
        else:
            print("Fitting the model to the data in the noiseless case")
            self._fit_noiseless(data, A_hat)

    def _fit_noisy(self, data, A_hat):
        signals = cp.Variable((data.shape[0], A_hat.shape[1]))  # Signal matrix
        noise = cp.Variable(data.shape)  # Noise matrix

        # Define the objective to minimize the L1 norm of signals and the Frobenius norm of noise
        objective = cp.Minimize(cp.norm1(signals) + (1/2)*cp.norm(noise, 'fro')**2)

        # Define constraints
        constraints = [A_hat @ signals.T + noise.T == data.T]

        # Formulate and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Retrieve the solution
        self.Signals = signals.value 
       

    def _fit_noiseless(self, data, A_hat):
         # Define the variable for all signals as a matrix (shape: n x J)
        signals = cp.Variable((data.shape[0], A_hat.shape[1]))
         
        # Define the objective to minimize the L1 norm across all rows
        objective = cp.Minimize(cp.norm1(signals))

        # Define the equality constraint for all rows: A_hat * signal_row == data_row
        constraints = [A_hat @ signals.T == data.T]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Store the result
        self.Signals = signals.value

  