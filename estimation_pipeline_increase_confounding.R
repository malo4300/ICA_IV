source("treatment_effect_estimators.R")

conf = 6
get_signals = function(seed){
  read.csv(get_path("increase_conf/signals/estimated_signals_CausalVarEM_large_conf_6_", seed))
}
get_data = function(seed){
  read.csv(get_path("increase_conf/data/data_obs_large_conf_6_",  seed))
}
get_true_mixing_matrix= function(seed){
  read.csv(get_path("increase_conf/mixing_matrix/true_mixing_large_conf_6_", seed))
}
get_estimated_mixing_matrix = function(seed){
  read.csv(paste("increase_conf/mixing_matrix/estimated_mixing_CausalVarEM_large_conf_6_", seed, ".csv", sep = ""))
}
p_values_iv = read.csv(paste("IV_test_results/increased_confounding/LDAG_CausalVarEM_conf_",conf,".csv", sep = ""), row.names = 1)[-1]
#p_values_iv = read.csv("IV_test_results/increased_confounding/LDAG_CausalVarEM_conf_1.csv",  row.names = 1)[-1]
p_values_indp = read.csv(paste("ind_tests/increase_conf/LDAG_conf_",conf,".csv", sep = ""), row.names = NULL, header= FALSE)[-1]
#p_values_indp = read.csv("ind_tests/increase_conf/LDAG_conf_1.csv", row.names = NULL, header=FALSE)[-1]
p_values_indp_undcond = read.csv(paste("ind_tests_uncon/LDAG_CausalVarEM_conf_", conf, ".csv", sep = "") ,header= FALSE)[-1]


table(apply(p_values_indp,1, function(x) sum(x<.05)))

cand_confounder_idx <- vector("list", 100)
cand_source_idx = vector("list", 100)

i = 1
for (i in 1:100) {
  candidates <- estimated_confounder_index_v2(p_values_iv[i,], p_values_indp[i,])
  cand_confounder_idx[[i]] <- candidates
  candidates <- estimated_treatmet_and_outcome_ind(p_values_iv[i,], p_values_indp_undcond[i,])
  cand_source_idx[[i]] <- candidates
}



# confounder source -----

ind = which(sapply(cand_confounder_idx, function(x) length(x) == 1 && !is.na(x)))

l = length(ind)
print(l)

true_treatment_effect_confounder_idx = rep(0,l)
estimated_treatment_efect_confounder_idx = rep(NA,l)
ols_biased = rep(NA,l)
estimated_treatment_efect_column_extraction = rep(NA,l)

J = 9
I = J-1
treatment_col = I-1 # by construction
for (i in 1:l){
  seed = ind[i]-1
  indx = ind[i]
  signals = get_signals(seed)
  data = get_data(seed)
  confounder_source_col = as.numeric(cand_confounder_idx[indx])
  df = data.frame(y = data[,I], treatment = data[, treatment_col], data[,c(-treatment_col, -I)], conf = signals[,confounder_source_col])
  fit = lm(y~.-1,df)  
  estimated_treatment_efect_confounder_idx[i] = coef(fit)["treatment"] 
  true_mm = get_true_mixing_matrix(seed)
  true_treatment_effect_confounder_idx[i] = column_extraction(true_mm)
  ols_biased[i] = classic_ols(data)
  mm = get_estimated_mixing_matrix(seed)
  estimated_treatment_efect_column_extraction[i] = column_extraction(mm)
  
}  


plot(true_treatment_effect_confounder_idx,estimated_treatment_efect_confounder_idx ,xlab = "True treatment", ylab = "Estimated treatment via OLS")
points(true_treatment_effect_confounder_idx, ols_biased, col ="red")
points(true_treatment_effect_confounder_idx, estimated_treatment_efect_column_extraction, col ="blue")
abline(a = 0, b = 1)

rmse(true_treatment_effect_confounder_idx, estimated_treatment_efect_confounder_idx)
rmse(true_treatment_effect_confounder_idx, ols_biased)
rmse(true_treatment_effect_confounder_idx, estimated_treatment_efect_column_extraction)

##############################
# estimation on 7 sources -----


ind = which(sapply(cand_source_idx, function(x) x[1] != x[2]))

l = length(ind)
print(l)

true_treatment_effect_confounder_idx = rep(0,l)
estimated_treatment_efect_source_idx = rep(NA,l)
ols_biased = rep(NA,l)
estimated_treatment_efect_column_extraction = rep(NA,l)


J = 9
I = J-1
treatment_col = I-1 # by construction
for (i in 1:l){
  seed = ind[i]-1
 indx = ind[i]
  signals = get_signals(seed)
  data = get_data(seed)
  remove = unlist(cand_source_idx[indx])
  df = data.frame(y = data[,I], treatment = data[, treatment_col], sign = signals[,-remove])
  fit = lm(y~.-1,df)  
  estimated_treatment_efect_source_idx[i] = coef(fit)["treatment"] 
  true_mm = get_true_mixing_matrix(seed)
  true_treatment_effect_confounder_idx[i] = column_extraction(true_mm)
  ols_biased[i] = classic_ols(data)
  mm = get_estimated_mixing_matrix(seed)
  estimated_treatment_efect_column_extraction[i] = column_extraction(mm)
}  


plot(true_treatment_effect_confounder_idx,estimated_treatment_efect_source_idx ,xlab = "True treatment", ylab = "Estimated treatment via OLS")
points(true_treatment_effect_confounder_idx, ols_biased, col ="red")
points(true_treatment_effect_confounder_idx, estimated_treatment_efect_column_extraction, col ="blue")
abline(a = 0, b = 1)


rmse(true_treatment_effect_confounder_idx, estimated_treatment_efect_source_idx)
rmse(true_treatment_effect_confounder_idx, ols_biased)
rmse(true_treatment_effect_confounder_idx, estimated_treatment_efect_column_extraction)

# base lines, ols and column extraction




