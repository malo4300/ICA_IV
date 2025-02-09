
results = read.csv("IV_test_results/p_values_using_true_signals_large_DAG.csv")

p_val = results[,3:ncol(results)]

# number of non significant test we would like to have exactly J-1
table(apply(p_val, 1, function(x) sum(x > .05)))
# unique maxima we would like to have exactly one
table(apply(p_val, 1, function(x) sum(x == max(x))))


# only applicable when we know the true source permutation
# how often is the correct column the unique maximum
sum(apply(p_val, 1, function(x) sum(x == max(x)) == 1 & which.max(x) == ncol(p_val)-1 ))

# how often is the correct column one of the maximum value

sum(apply(p_val, 1, function(x) max(x) == x[ncol(p_val)-1]))

# how often would one reject the correct columns as candidate
sum(p_val$V9 == 0 )

      