source("extract_confounder_source.R")

conf = 6
p_values_iv = read.csv(paste("IV_test_results/increased_confounding/LDAG_CausalVarEM_conf_",conf,".csv", sep = ""), row.names = 1)[-1]
p_values_indp = read.csv(paste("ind_tests/increase_conf/LDAG_conf_",conf,".csv", sep = ""), row.names = NULL, header= FALSE)[-1]
mask =p_values_indp ==.5

sum(mask, na.rm = T)
#p_values_indp[mask] = 0
#p_values_indp = read.csv("ind_tests/increase_conf/CausalVarEM_large_conf_3_old.csv")[-1]


cand <- vector("list", 100)
i = 1
for (i in 1:100) {
  candidates <- estimated_confounder_index_v2(p_values_iv[i,], p_values_indp[i,])
  cand[[i]] <- candidates
}

 # unique outputs, these are the only ones on which we can safely to the fitting
print(paste("we can run a regression in", sum(sapply(cand, function(x) length(x) == 1 && !is.na(x))), "cases"))
print(paste("Number of NAs", sum(sapply(cand, function(x) any(is.na(x))))))
print(paste("Number of non-unique outputs", sum(sapply(cand, function(x) length(x)>1))))

# find index of unique values ----

ind = which(sapply(cand, function(x) length(x) == 1 && !is.na(x)))

l = length(ind)


true_treatment_effect = rep(0,l)
estimated_treatment_efect = rep(NA,l)


J = 9
I = J-1
treatment_col = I-1 # by construction
for (i in 1:l){
  seed = ind[i]-1
  indx = ind[i]
  signals = read.csv(get_path(paste("increase_conf/signals/estimated_signals_CausalVarEM_large_conf_", conf, "_", sep= ""), seed))
  data = read.csv(get_path(paste("increase_conf/data/data_obs_large_conf_", conf, "_", sep= ""), seed))
  confounder_source_col = as.numeric(cand[indx])
  df = data.frame(y = data[,I], treatment = data[, treatment_col], data[,c(-treatment_col, -I)], conf = signals[,confounder_source_col])
  fit = lm(y~.,df)  
  estimated_treatment_efect[i] = coef(fit)["treatment"] 
  true_treatment_effect[i] = get_true_treatment_from_mixing_matrix(path = paste("increase_conf/mixing_matrix/true_mixing_large_conf_", conf, "_", sep= ""),i = seed)
}  


plot(true_treatment_effect,estimated_treatment_efect ,xlab = "True treatment", ylab = "Estimated treatment via OLS")
abline(a = 0, b = 1)

# load treatment effects for the column extraction
estimated_treatment_efect_column_extraction = rep(NA,l)
for (i in 1:l) {
  seed = ind[i]-1
  indx = ind[i]
  mm = read.csv(get_path(paste("increase_conf/mixing_matrix/estimated_mixing_CausalVarEM_large_conf_", conf, "_", sep= ""), seed))
  estimated_treatment_efect_column_extraction[i] = mm[nrow(mm), ncol(mm)-1]
}

plot(true_treatment_effect, estimated_treatment_efect_column_extraction, col = "red")
points(true_treatment_effect,estimated_treatment_efect ,xlab = "True treatment", ylab = "Estimated treatment via OLS",
     main= "CausalVarEM and adjusted scheme for finding confounder source")

abline(a = 0, b = 1)


# rmse
sqrt(mean((true_treatment_effect-estimated_treatment_efect)^2))
sqrt(mean((true_treatment_effect-estimated_treatment_efect_column_extraction)^2))



