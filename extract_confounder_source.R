# the point is the combine the p-values of the IV test and of the independence test

get_path = function(str, i){
  return(paste(str,i, ".csv", sep = "" ))
}




estimated_confounder_index = function(p_values_iv,p_values_indp){
  # IV test should reject all sources as potential instruments leaving only the treatment source 
  # in practice pick the largest value if it is unique
  mx = max(p_values_iv)
  ind_iv_max = which(p_values_iv == mx)
  if(length(ind_iv_max)>1){
    warning("IV test has selcted non-unique candidate for treatment source")
  }
  
  # for the independence test, the only not independent sources should be the treatment and confounder source, pick the smallest two p-values if unique
  
  ordered_p_values =  sort(as.numeric(p_values_indp))
  if(ordered_p_values[2] == ordered_p_values[3]){
    warning("Independence test return non-unique candidates for treatment and confounde source")
  }
  
  candidates = which(ordered_p_values[2]  >= as.numeric(p_values_indp))

  
  final_candidate = ind_iv_max[ind_iv_max %in% candidates]
  
  if(length(final_candidate)>1){
    warning("Final candidate not unique")
  }
  if(length(final_candidate) == 0){
    warning("No candidate: return NA")
    return(NA)
  }
  return(final_candidate)
}




get_path = function(str, i){
  return(paste(str,i, ".csv", sep = "" ))
}

get_true_treatment_from_mixing_matrix = function(path = "extended_dgp/mixing_matrix/true_mixing_", i){
  path_complete = get_path(path,i)
  mm = read.csv(path_complete)
  return(mm[nrow(mm), ncol(mm)-1])
}
