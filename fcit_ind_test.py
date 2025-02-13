
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
    J = 9
    n = 1000
    p_values = np.ones((100,J))
   


    for i in tqdm(range(100)):
            data_obs = pd.read_csv(f"extended_dgp/data/data_obs_init14_{i}.csv", header=0).values
            signals = pd.read_csv(f"extended_dgp/signals/estimated_signals_VarEM_init14_{i}.csv", header=0).values
            p_values[i,:] = p_vals(data_obs, signals, n, J)

    pd.DataFrame(p_values).reset_index().to_csv(f"ind_tests/FCIT_VarEM_init14.csv", header=False, index = False)

