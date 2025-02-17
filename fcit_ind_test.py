
import numpy as np
from fcit import fcit
import pandas as pd
from tqdm import tqdm

def p_vals(data, signals, n, J):
    T = data[:,-2]
    controlls = data[:, 0:(J-3)]
    p_val = []
    for j in range(J):
        p_val.append(fcit.test(T.reshape(n,1), signals[:,j].reshape(n,1),controlls))
    return p_val


if __name__ == "__main__":
    J = 6
    I = J-1
    n = 1000
    conf_levels = [6]
   
    for j in range(len(conf_levels)):
        p_values = np.ones((100,J))
        conf = conf_levels[j]
        for i in tqdm(range(100)):
                data_obs = pd.read_csv(f"increase_conf/data/data_obs_conf_{conf}_{i}.csv", header=0).values
                signals = pd.read_csv(f"increase_conf/signals/estimated_signals_CausalVarEM_conf_{conf}_{i}.csv", header=0).values
                p_values[i,:] = p_vals(data_obs, signals, n, J)

        pd.DataFrame(p_values).reset_index().to_csv(f"ind_tests/increase_conf/SDAG_CausalVarEM_conf_{conf}.csv", header=False, index = False)

