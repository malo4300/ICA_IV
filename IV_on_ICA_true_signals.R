
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
library(progress)

B = 500 # number of bootstrap draws
ncpus = 12
synthetic_D_method = 'standard'
kappa_method = 'sigmas'

order_data = function(data){
  ordered_data = matrix(0, nrow = nrow(data), ncol = ncol(data))
  ordered_data[,1] = data[, ncol(data)] # Y has to be the first column, in data it is the last
  ordered_data[,2:(ncol(data)-1)] = as.matrix(data[,1:(ncol(data)-2)]) # controls in the middle
  ordered_data[,ncol(data)] = data[, ncol(data)-1] # Treatment as last
  return(ordered_data)
}

get_path = function(str, i){
  return(paste(str,i, ".csv", sep = "" ))
}

p_values = function(ordered_data, signals){
  p_val = rep(0,ncol(signals))
  for (i in 1:ncol(signals)) {
    p_val[i] = fn_test_instrument_validity(ordered_data, signals[[names(signals)[i]]], B, 
                                       ncpus, 
                                       kappa_method,
                                       synthetic_D_method)$pseudo_p
  }
  return(p_val)
}

J = 9
n = 1000
results = matrix(0, nrow = 100, ncol = J+1)
cand <- vector("list", 100)
pb <- progress_bar$new(
  format = "[:bar] :percent | ETA: :eta",
  total = 100, 
  clear = TRUE, 
  width = 60
)
i = 0
for (i in 0:99) {
  sg_path = get_path("extended_dgp/signals/true_signals_", i)
  dt_path = get_path("extended_dgp/data/data_obs_", i)
  data =  read.csv(dt_path, header = 1)
  signals = read.csv(sg_path, header = 1) + matrix(rnorm(n*J,0,.1), n,J)
  ordered_data = order_data(data)
  results[i+1,1] = i
  results[i+1,2:ncol(results)] = p_values(ordered_data, signals)
  pb$tick()
}

#write.csv(results, file = "IV_test_results/p_values_true_signals_LDAG_conf6.csv")
