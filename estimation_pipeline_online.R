source("treatment_effect_estimators.R")
library(progress)

get_path_seed = function(str, i, conf){
  return(paste(str,conf, "_",i, ".csv", sep = "" ))
}


get_path= function(str, i){
  return(paste(str,i, ".csv", sep = "" ))
}

p_values_indp = read.csv("ind_tests/increase_conf/LDAG_VarEM_non_causal_bounded.csv", header =1)[-1]


cand <- vector("list", 100)
pb <- progress_bar$new(
  format = "[:bar] :percent | ETA: :eta",
  total = 100, 
  clear = TRUE, 
  width = 60
)

conf = 1

for (i in 1:100) {
  seed = i - 1
  #sg_path = get_path_seed("increase_conf/signals/estimated_signals_CausalVarEM_conf_", seed, conf)
  sg_path = get_path("extended_dgp/non_causal/signals/estimated_signals_VarEM_VarEM_non_causal_bounded_", seed)
  #dt_path = get_path_seed("increase_conf/data/data_obs_conf_", seed,conf)data_obs_init13_
  dt_path = get_path("extended_dgp/non_causal/data/data_obs_VarEM_non_causal_bounded_", seed)
  data = read.csv(dt_path, header = TRUE)
  signals = read.csv(sg_path, header = TRUE)
  candidates <- online_extraction(p_values_indp[i, ], data, signals)
  cand[[i]] <- candidates
  pb$tick()
}

#saveRDS(cand, file="IV_test_results/LDAG_VarEM_non_causal_bounded.RData")
cand = readRDS(file="IV_test_results/VarEM_init14.RData")

# unique outputs, these are the only ones on which we can safely to the fitting
print(paste("we can run a regression in", sum(sapply(cand, function(x) length(x) == 1 && !is.na(x))), "cases"))
print(paste("Number of NAs", sum(sapply(cand, function(x) any(is.na(x))))))
print(paste("Number of non-unique outputs", sum(sapply(cand, function(x) length(x)>1))))


# find index of unique values ----

ind = which(sapply(cand, function(x) length(x) == 1 && !is.na(x)))

l = length(ind)


true_treatment_effect = rep(0,l)
estimated_treatment_efect = rep(NA,l)
ols_biased = rep(NA, l)

J = 9
I = J-1
i = 1
treatment_col =I-1# by construction
for (i in 1:l){
  seed = ind[i] -1 
  indx = ind[i]
  #signals = read.csv(get_path_seed("increase_conf/signals/estimated_signals_CausalVarEM_large_conf_", seed,conf ))
  signals = read.csv(get_path("extended_dgp/signals/estimated_signals_VarEM_init14_", seed))
  #true_signals = read.csv(get_path("extended_dgp/bounded_coef/signals/data_obs_init13_", seed))
  #data = read.csv(get_path_seed("increase_conf/data/data_obs_large_conf_", seed,conf))
  data = read.csv(get_path("extended_dgp/data/data_obs_init14_", seed))
  confounder_source_col = as.numeric(cand[indx])
  df = data.frame(y = data[,I], treatment = data[, treatment_col] , 
                  cont = data[,-c(treatment_col, I)], 
                  con = signals[,1])
  fit = lm(y~.-1,df)  
  ols_biased[i] = classic_ols(data)
  estimated_treatment_efect[i] = coef(fit)["treatment"] 
  #true_treatment_effect[i] = get_true_treatment_from_mixing_matrix(path = paste("extended_dgp/non_causal/mixing_matrix/true_mixing_non_causal_bounded_", conf ,"_",sep= ""),i = seed)
  true_treatment_effect[i] = get_true_treatment_from_mixing_matrix( path = "extended_dgp/mixing_matrix/true_mixing_init14_", i = seed)
}  

plot(true_treatment_effect,estimated_treatment_efect ,xlab = "True treatment", ylab = "Estimated treatment via OLS",
       main= "CausalVarEM and adjusted scheme for finding confounder source", col = "black")
points(true_treatment_effect, ols_biased, col = "red")





abline(a = 0, b = 1)
# load treatment effects for the column extraction
estimated_treatment_efect_column_extraction = rep(NA,l)
for (i in 1:l) {
  seed = ind[i]-1
  indx = ind[i]
  #mm = read.csv(get_path("increase_conf/mixing_matrix/estimated_mixing_CausalVarEM_conf_6_", seed))
  mm = read.csv(get_path("extended_dgp/non_causal/mixing_matrix/estimated_mixing_CausalVarEM_non_causal_bounded_", seed))
  estimated_treatment_efect_column_extraction[i] = mm[nrow(mm), ncol(mm)-1]
}

plot(true_treatment_effect, estimated_treatment_efect_column_extraction, col = "red",  xlab = "True treatment", ylab = "Estimated treatment")
points(true_treatment_effect,estimated_treatment_efect ,xlab = "True treatment", ylab = "Estimated treatment via OLS",
       main= "CausalVarEM and adjusted scheme for finding confounder source")
points(true_treatment_effect, ols_biased, col = "blue")
abline(a = 0, b = 1)


# rmse
sqrt(mean((true_treatment_effect-estimated_treatment_efect)^2))
sqrt(mean((true_treatment_effect-estimated_treatment_efect_column_extraction)^2))
sqrt(mean((true_treatment_effect-ols_biased)^2))

