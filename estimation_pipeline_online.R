source("extract_confounder_source.R")
library(progress)

get_path = function(str, i){
  return(paste(str,i, ".csv", sep = "" ))
}

#p_values_indp = read.csv("ind_tests/FCIT_Causal_Var_EM.csv", row.names = 1, header = FALSE)[-1]
p_values_indp = read.csv("ind_tests/increase_conf/VarEM_large_conf_3_old.csv", header =0)[-1]


cand <- vector("list", 100)
pb <- progress_bar$new(
  format = "[:bar] :percent | ETA: :eta",
  total = 100, 
  clear = TRUE, 
  width = 60
)

for (i in 1:100) {
  seed = i - 1
  sg_path = get_path("increase_conf/signals/estimated_signals_VarEM_large_conf_3_", seed)
  dt_path = get_path("increase_conf/data/data_obs_large_conf_1_", seed)
  data = read.csv(dt_path, header = TRUE)
  signals = read.csv(sg_path, header = TRUE)
  candidates <- online_extraction(p_values_indp[i, ], data, signals)
  cand[[i]] <- candidates
  pb$tick()
}

saveRDS(cand, file="IV_test_results/increased_confounding/LDAG_VarEM_conf_3_old.RData")
cand = readRDS(file="IV_test_results/increased_confounding/LDAG_VarEM_conf_3_old.RData")

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
  signals = read.csv(get_path("increase_conf/signals/estimated_signals_CausalVarEM_large_conf_6_", seed))
  data = read.csv(get_path("increase_conf/data/data_obs_large_conf_6_", seed))
  confounder_source_col = as.numeric(cand[indx])
  df = data.frame(y = data[,I], treatment = data[, treatment_col], data[,c(-treatment_col, -I)], conf = signals[,confounder_source_col])
  fit = lm(y~.-1,df)  
  estimated_treatment_efect[i] = coef(fit)["treatment"] 
  true_treatment_effect[i] = get_true_treatment_from_mixing_matrix(path = "increase_conf/mixing_matrix/true_mixing_large_conf_6_",i = seed)
}  

plot(true_treatment_effect,estimated_treatment_efect ,xlab = "True treatment", ylab = "Estimated treatment via OLS",
       main= "CausalVarEM and adjusted scheme for finding confounder source")
abline(a = 0, b = 1)
# load treatment effects for the column extraction
estimated_treatment_efect_column_extraction = rep(NA,l)
for (i in 1:l) {
  seed = ind[i]-1
  indx = ind[i]
  mm = read.csv(get_path("increase_conf/mixing_matrix/estimated_mixing_CausalVarEM_large_conf_6_", seed))
  estimated_treatment_efect_column_extraction[i] = mm[nrow(mm), ncol(mm)-1]
}

plot(true_treatment_effect, estimated_treatment_efect_column_extraction, col = "red")
points(true_treatment_effect,estimated_treatment_efect ,xlab = "True treatment", ylab = "Estimated treatment via OLS",
       main= "CausalVarEM and adjusted scheme for finding confounder source")
abline(a = 0, b = 1)


# rmse
sqrt(mean((true_treatment_effect-estimated_treatment_efect)^2))
sd(true_treatment_effect-estimated_treatment_efect)
sqrt(mean((true_treatment_effect-estimated_treatment_efect_column_extraction)^2))
sd(true_treatment_effect)

