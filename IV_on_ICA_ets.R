
####################################################################
# implementation of instrument validity test developed in 
# Burauel, Patrick F. "Evaluating Instrument Validity using the Principle of Independent Mechanisms." 
# Journal of Machine Learning Research 24.176 (2023): 1-56.
# https://jmlr.org/papers/v24/20-1287.html
# abbreviated to B23 in the following
####################################################################

rm(list = ls()) # To clear all


# load required packages
packages <- c("boot",
              "corpcor",
              "MASS",
              "parallel",
              "pracma",
              "knockoff")

lapply(packages,require,character.only=TRUE)

# load necessary functions
source('IV_test_code/fn_2SLS.R')
source('IV_test_code/estimate_kappa_bootstrap_wrapper.R')
source('IV_test_code/fn_complement.R')
source('IV_test_code/fn_complement_knockoff.R')
source('IV_test_code/fn_test_instrument_validity.R')
source('IV_test_code/estimate_confounding_via_kernel_smoothing.R')
source('IV_test_code/estimate_confounding_sigmas.R')
source('IV_test_code/estimate_confounding_via_kernel_smoothing.R')
source('IV_test_code/estimate_confounding_sigmas.R')

B = 500 # number of bootstrap draws
ncpus = 10
synthetic_D_method = 'standard'
kappa_method = 'sigmas'

order_data = function(data){
  ordered_data = matrix(0, nrow = nrow(data), ncol = ncol(data))
  ordered_data[,1] = data[, ncol(data)] # Y has to be the first column, in data it is the last
  ordered_data[,2:(ncol(data)-1)] = as.matrix(data[,1:3]) # controlls in the middle
  ordered_data[,ncol(data)] = data[, ncol(data)-1] # Treatment as last
  return(ordered_data)
}

get_path = function(str, i){
  return(paste(str,i, ".csv", sep = "" ))
}

p_values = function(ordered_data, signals){
  p_val = rep(0,6)
  for (i in 1:length(names(signals))) {
    p_val[i] = fn_test_instrument_validity(ordered_data, signals[[names(signals)[i]]], B, 
                                       ncpus, 
                                       kappa_method,
                                       synthetic_D_method)$pseudo_p
  }
  return(p_val)
}


results = matrix(0, nrow = 100, ncol = 7)
pb = txtProgressBar(min = 0, max = 99, initial = 0, style = 3) 
for (i in 0:99) {
  #sg_path = get_path("sim_data/signals/estimated_signals_", i)
  sg_path = get_path("sim_data_separate_signels/estimated_signals_separate_CausalVarEM_", i)
  dt_path = get_path("sim_data/data/data_obs_", i)
  data =  read.csv(dt_path, header = T)
  signals = read.csv(sg_path, header = T)
  ordered_data = order_data(data)
  results[i+1,1] = i
  results[i+1,2:7] = p_values(ordered_data, signals)
  setTxtProgressBar(pb,i)
}

write.csv(results, file = "IV_test_results/p_values_CausalVarEM_separate_pen_100.csv")
