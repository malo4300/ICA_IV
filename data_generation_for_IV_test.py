from ICA.models.ICA_EM import *
from ICA.models.dgp import * 
import pandas as pd


def sim(mode, B, iter, dgp_, name):
    data = dgp_(noise_dict= {"loc" : 0, "scale" : 0}, prior= {"loc" : 0, "scale" : 1/np.sqrt(2)}, level_of_confounding = 1)
    for i in tqdm.tqdm(range(B)):
        data.generate_data(n,I, J, random_state=i, init_range = [2,4])
        est = CausalVarEM(update_sigma=False, true_A=None, max_iter = iter, random_seed= i,  mode = mode) 
        est.fit(data.data_observed,J, noise_params= {"mean" : 0, "std" : 1}, progress_bar=False)
        pd.DataFrame(data.mixing_matrix_observed).to_csv(f"extended_dgp/mixing_matrix/true_mixing_{name}_{i}.csv", index = False)
        pd.DataFrame(est.A).to_csv(f"extended_dgp/mixing_matrix/estimated_mixing_CausalVarEM_{name}_{i}.csv", index = False)
        pd.DataFrame(est.Signals).to_csv(f"extended_dgp/signals/estimated_signals_CausalVarEM_{name}_{i}.csv", index = False)
        pd.DataFrame(data.signals).to_csv(f"extended_dgp/signals/true_signals_{name}_{i}.csv", index = False)
        pd.DataFrame(data.data_observed).to_csv(f"extended_dgp/data/data_obs_{name}_{i}.csv", index = False)




if __name__ == "__main__":
    J = 9
    I = J-1
    n = 10000
    sim("lower_triangular", 100, 100, dgp_extended, "init24")