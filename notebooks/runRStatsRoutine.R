#!/usr/bin/env Rscript
# Run R script from command line using ```Rscript runRStatsRoutine.R``` from the containing folder.
print("Checkpoint 1")
tryCatch(
  {library("globaltest")}, 
  error = function(error_message){
    print("Package Not Found. Stopping Code Execution.")
    stop(error_message)
    q()
  }, 
  warnings = function(warning_message){"not good"})
print("Checkpoint 2")

tryCatch(
  {library("gobbledigook")}, 
  error = function(error_message){
    print("Package Not Found. Stopping Code Execution.")
    stop(error_message)
    q()
    }, 
  warnings = function(warning_message){"not good"})
print("Checkpoint 3")

runGroupGlobalTest <- function(responseColumn, featureFrame){
  # Runs globaltest for a single two-sample contrast for a single feature set.
  # Input data is basically a response vector (the group ids for a reference and a trt group in our case),
  # a featureFrame containing all features relevant to the current group (all features used here)
  # Returns:
  # output results in a named list
  return()
}

runGroupDescriptivesComputation <-  function(responseColumn, groupColumn, featureColumn){
   # Runs globaltest for a single two-sample contrast for a single feature set.
  # Input data is basically a response vector (the group ids for a reference and a trt group in our case),
  # a featureFrame containing all features relevant to the current group (all features used here)
  # this function is similar to global test, but computes group-based descriptives that can be comptued using just
  # data from one single group like average or median fold change 
  return()
}

runFeatureDescriptivesComputation <-  function(responseColumn, groupColumn, featureColumn){
  # Runs descriptive statistics retrieval for a two-sample contrast for a single feature
  # this function is similar to global test, but only gets a single featureColumn and computes the descriptive from there
  return()
}

contrast_reference_category = "treatment1"
treatmeants = c("treatment1","treatment2","treatment3")
unique_groups <- c("group0", "group1", "group2") # etc...
group_assignments <- c("group0" = c("feature_4","feature_10","feature_271","feature_78","feature_49","feature_21"), ...) # and so on for all other groups