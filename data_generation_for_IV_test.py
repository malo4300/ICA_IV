from models.ICA_EM import *
from models.dgp import * 
import pandas as pd

J = 6
I = 5
n = 10000

def sim(mode, B, iter):
    B = B
   


    # fifth should correspond to the true treatment
    data = dgp(noise_dict= {"loc" : 0, "scale" : 0}, prior= {"loc" : 0, "scale" : 1/np.sqrt(2)})
    for i in tqdm.tqdm(range(B)):
        data.generate_data(n,I, J, random_state=i)
        est = CausalVarEM(update_sigma=False, true_A=None, max_iter = iter, random_seed= i, mode=mode)
        est.fit(data.data_observed,J, noise_params= {"mean" : 0, "std" : 1}, progress_bar=False)
        pd.DataFrame(data.mixing_matrix_observed).to_csv(f"sim_data/mixing_matrix/true_mixing_{i}.csv", index = False)
        pd.DataFrame(est.A).to_csv(f"sim_data/mixing_matrix/estimated_mixing_{i}.csv", index = False)
        pd.DataFrame(est.Signals).to_csv(f"sim_data/signals/estimated_signals_{i}.csv", index = False)
        pd.DataFrame(data.signals).to_csv(f"sim_data/signals/true_signals_{i}.csv", index = False)
        pd.DataFrame(data.data_observed).to_csv(f"sim_data/data/data_obs_{i}.csv", index = False)

if __name__ == "__main__":
    sim("lower_triangular", 100, 100)