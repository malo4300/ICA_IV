source("extract_confounder_source.R")
p_values_iv = read.csv("IV_test_results/p_values_CausalVarEM_extended_DGP.csv", row.names = 1)[-1]
p_values_indp = read.csv("ind_tests/large_DAG_CausalVarEM.csv", row.names = NULL)


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
  signals = read.csv(get_path("extended_dgp/signals/estimated_signals_lower_triangular_", seed))
  data = read.csv(get_path("extended_dgp/data/data_obs_", seed))
  confounder_source_col = as.numeric(cand[indx])
  df = data.frame(y = data[,I], treatment = data[, treatment_col], data[,c(-treatment_col, -I)], conf = signals[,confounder_source_col])
  fit = lm(y~.,df)  
  estimated_treatment_efect[i] = coef(fit)["treatment"] 
  true_treatment_effect[i] = get_true_treatment_from_mixing_matrix(path = "extended_dgp/mixing_matrix/true_mixing_",i = seed)
}  

plot(true_treatment_effect,estimated_treatment_efect ,xlab = "True treatment", ylab = "Estimated treatment via OLS",
     main= "Using CausalVarEM sources")
abline(a = 0, b = 1)
