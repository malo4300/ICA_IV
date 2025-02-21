library("bnlearn")
library(abjutils)


J= 9# number of sources. We assume the treatment is the second last column
n = 1000
p_vals = matrix(NA,100,J)
pb = txtProgressBar(min = 0, max = 99, initial = 0, style = 3) 

for (i in 0:99) {
  data_obs = read.csv(paste("extended_dgp/bounded_coef/data/data_obs_init13_", i, ".csv", sep = ""), header = 1)
  T = data_obs[,ncol(data_obs)-1] 
  signals = read.csv(paste("extended_dgp/bounded_coef/signals/estimated_signals_VarEM_init13_", i, ".csv", sep = ""), header = 1)
  #signal_est = read.csv(paste("sim_data_VarEM/signals/estimated_signals_", i, ".csv", sep = ""), header = 1)
  p_val = rep(NA,J)
  controlls = data_obs[,1:(ncol(data_obs)-2)] 
  for (j in 1:J) {
    p_val[j] = ci.test(T,signals[,j], controlls)$p.value # when using true sources add rnorm(n, 0, .1)
  }
  p_vals[i+1,] = p_val
  setTxtProgressBar(pb,i)
  rm(data_obs,signals, controlls ) # helps to avoid crashes
}

write.csv(p_vals, "ind_tests/increase_conf/LDAG_VarEM_init13_old.csv")

p_vals = read.csv("ind_tests/increase_conf/CausalVarEM_large_conf_3_old.csv", row.names = NULL)[-1]
p_vals = read.csv("ind_tests/increase_conf/VarEM_large_conf_3_old.csv", row.names = NULL)[-1]

#p_vals = read.csv("ind_tests/FCIT_Causal_Var_EM.csv", row.names = 1)[-1]
p_vals_noise = read.csv("ind_tests/increase_conf/LDAG_conf_3.csv", row.names = NULL)
# how many p values are essentially zero
table(apply(p_vals, 1, function(x) sum(x == 0)))

# find two lowest p values
two_lowest = apply(p_vals, 1, function(x) sort.list(x, decreasing = T)[c(ncol(p_vals)-1, ncol(p_vals))])
two_lowest = t(apply(t(two_lowest), 1, sort))

two_lowest_noise= apply(p_val_noise, 1, function(x) sort.list(x, decreasing = T)[c(ncol(p_val_noise)-1, ncol(p_val_noise))])
two_lowest_noise = t(apply(t(two_lowest_noise), 1, sort))


# how often are the two lowest values unique, they are unique when the second and third lowest value are not the same

sum(apply(p_vals, 1, 
          function(x) 
            as.numeric(x[order(as.numeric(x), decreasing = T)])[ncol(p_vals)-2] !=
            as.numeric(x[order(as.numeric(x), decreasing = T)])[ncol(p_vals)-1]))




# only relevant if one uses the true sources ------------------
true_index = c(1,ncol(p_vals)-1)
sum(apply(two_lowest, 1, function(x) all(x == true_index)))

# 

table(apply(p_vals, 1, function(x) which.min(x)))

