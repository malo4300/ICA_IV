from fcit_ind_test import fcit_ind_test
import pandas as pd
import numpy as np
from tqdm import tqdm

J = 9
n = 10000
p_values = np.zeros((n,J))


for i in tqdm(range(100)):
    data_obs = pd.read_csv(f"extended_dgp/data/data_obs_{i}.csv", header=0).values
    signals = pd.read_csv(f"extended_dgp/signals/true_signals_{i}.csv", header=0).values
    T = data_obs[:,-2]
    p_val = []
    controlls = data_obs[:, 1:(J-2)]
    for j in range(J):
         p_val.append(fcit_ind_test.test(T.reshape(n,1), signals[:,i].reshape(n,1) ,controlls))
    p_values[i,:] = p_val

pd.DataFrame(p_values).to_csv("ind_tests/true_signals_fcit.csv", header=False)

