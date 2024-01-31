rm(list = ls())

#' generateConfigurations
#' 
#' Creates configuration data frame (list) from inputs.
#'
#' @param measures List of measures c("globalTest", "log2FoldChange") are supported. Must be a list even if scalar.
#' @param contrasts List of contrasts. Named list, where names are contrast names, and sub-list elements are treatment identifiers. For example: list("contrast_ctrl_vs_trt" = list(reference="ctrl", treatment="trt"))
#' @param feature_ids List of feature identifiers (character)
#' @param feature_sets List of feature sets, where names in the list are feature set ids, and list elements are lists with feature identifiers.
#'
#' @return
#' @export
#'
#' @examples
generateConfigurations <- function(measures, contrasts, feature_ids, feature_sets){
  # generates configurations for msfeast linear situation run
  print(feature_sets)
  print(contrasts)
  print(feature_ids)
  print(measures) # is a list, should be character vector!
  feature_set_names <- names(feature_sets)
  configurations <- data.frame()
  for (current_measure in measures){
    configurations = switch(
      current_measure,
      "globalTest" = rbind(
        configurations, 
        expand.grid(
          measure = current_measure, 
          feature_set = feature_set_names,
          feature_id = NA,
          contrast = contrasts, 
          stringsAsFactors = FALSE)
      ),
      "log2FoldChange" = rbind(
        configurations, 
        expand.grid(
          measure = current_measure, 
          feature_set = NA,
          feature_id = feature_ids, 
          contrast = contrasts, 
          stringsAsFactors = FALSE)
      ),
    )
  }
  configurations["feature_set_members"] <- NA
  # for all set entries, add the particular feature_ids list to the feaure_id columns
  for (row_number in 1:nrow(configurations)){
    current_set_entry <- configurations[["feature_set"]][[row_number]]
    if (!is.na(current_set_entry)){
      feature_id_list <- feature_sets[current_set_entry]
      # Accounting for possible scalar sets
      if ( typeof(feature_id_list) != "list") {
        feature_id_list <- list(feature_id_list)
      }
      configurations[["feature_set_members"]][[row_number]] <- feature_id_list
    }
  }
  return (configurations)
}

constructEmptyNamedList <- function(entryNamesList){
  # Construct empty named list with provided names list. Each name will be associated with an empty list entry.
  # To create a list of certain length, the vector constructor needs to be used
  emptyList <- sapply(entryNamesList, function(x) list())
  return (emptyList)
}

extractFeatureSetData <- function(){
  # Function Extracts data for specified contrast and feature set  from the full data tibble
  # The obvious thing to do for univariate methods is to have a set of one features
}

runHandlerLog2FoldChange <- function(resultsListEnv, feature_id, contrast, contrast_name, quantification_table, metadata_table){
  # Extract contrast specific data
  contrast <- contrasts[[contrast_name]]
  
  # get sample_ids
  treatment_reference_id <- contrast[[1]]
  treatment_treatment_id <- contrast[[2]]
  tmpMetaData <- metadata_table[which(metadata_table$treatment %in% c(treatment_reference_id, treatment_treatment_id)),]
  sample_ids <- pull(tmpMetaData, sample_id)
  tmpAllData <- quantification_table %>% 
    filter(., sample_id %in% sample_ids) %>%
    select(., all_of( c("sample_id", feature_id))) %>%
    left_join(., tmpMetaData, by= "sample_id") %>%
    mutate(treatment = factor(
      treatment, 
      levels = c(treatment_reference_id, treatment_treatment_id))) %>%
    mutate(treatment = as.numeric(treatment)-1)
  
  print(tmpAllData)
  referenceIntensities <- tmpAllData %>%
    filter(., treatment == 0) %>%
    select(., -c(treatment, sample_id)) %>%
    pull(.)
  
  treatmentIntensities <- tmpAllData %>%
    filter(., treatment == 1) %>%
    select(., -c(treatment, sample_id)) %>%
    pull(.)
  
  print(referenceIntensities)
  ratio <- mean(treatmentIntensities) / mean(referenceIntensities)
  log2ratio <- log2(ratio)
  
  tmpFeatureData <- tmpAllData %>% 
    select(-c("sample_id", "treatment")) %>%
    as.matrix(.)
  
  # get the actual value
  value = runif(1, 0.001, 100)
  
  # Attach to output
  resultsListEnv$"feature_specific"[[feature_id]][[contrast_name]]["log2FoldChange"] <- list(log2ratio)
}

runHandlerGlobalTest <- function(resultsListEnv, feature_set_name, feature_set_members, contrast, contrast_name, quantification_table, metadata_table){
  # Extract contrast specific data
  contrast <- contrasts[[contrast_name]]
  feature_ids <- feature_sets[[feature_set_name]]
  
  # get sample_ids
  treatment_reference_id <- contrast[[1]]
  treatment_treatment_id <- contrast[[2]]
  tmpMetaData <- metadata_table[which(metadata_table$treatment %in% c(treatment_reference_id, treatment_treatment_id)),]
  sample_ids <- pull(tmpMetaData, sample_id)
  tmpAllData <- quantification_table %>% 
    filter(., sample_id %in% sample_ids) %>%
    select(., all_of( c("sample_id", feature_ids))) %>%
    left_join(., tmpMetaData, by= "sample_id")
  tmpResponse <- tmpAllData %>% select(treatment) %>%
    mutate(treatment = factor(
      treatment, 
      levels = c(treatment_reference_id, treatment_treatment_id))) %>%
    mutate(treatment = as.numeric(treatment)-1) %>%
    as.matrix(.)
  tmpFeatureData <- tmpAllData %>% 
    select(-c("sample_id", "treatment")) %>%
    as.matrix(.)
  
  # get the actual values
  model_output <- globaltest::gt(tmpResponse, tmpFeatureData, model = "linear")
  p_value <- model_output@result[[1]] # extracts the p-value
  
  value1 = runif(1,0,1)
  value2 = sample(1:100, 1)
  
  # Attach to output
  resultsListEnv$"set_specific"[[feature_set_name]][[contrast_name]]["globalTestPValue"] <- list(p_value) 
  for (feature_id in feature_set_members){
    resultsListEnv$"feature_specific"[[feature_id]][[contrast_name]]["globalTestFeatureContribution"] <- list(value2)
  }
}


#' run_msfeast
#' 
#' Main interface function for msFeaST. This function provides a convenience wrapper to the globaltest (v5.50.0) package.
#' Given the quantification table, metadata_table, feature_sets, contrasts and measures, the function
#'  1. Initializes configurations for each test
#'  2. Initialize output data structures (hierarchical named lists for json export) 
#'  3. For each configuration:
#'    -> extract and construct relevant tables
#'    -> run the test handler -> add results to respective output structures
#'  4. Return results output structures as nested named lists and json strings
#'
#' Input Data assumptions:
#' --> all feature_id, sample_id, feature_set_id, condition_id values must be unique in their respective columns.
#' --> column names must exactly match expectations posited
#'
#' @param quantification_table A tibble (data frame) with a sample_id column (character) and a column for each feature_id containing the respective measurement values (float)
#' @param metadata_table A tibble with a sample_id column (character) and a condition_id column (character). The condition_id is expected to contain both the control/reference group and treatment/comparison groups. It is used to extract relevant samples for each contrast.
#' @param feature_sets A named list with keys representing feature_set_id (character) and sub-list elements representing feature_id (character)
#' @param feature_ids A list with feature_ids (character)
#' @param contrasts A list of 2-tuples containing the control/reference category and the treatment/comparison category (character). 
#' @param measures A list of measures to use for comparative purposes. These must match one or more of the following supported measures: c("globalTest", "log2FoldChange")
#'
#' @return
#' @export
#'
#' @examples
run_msfeast <- function(
    quantification_table, 
    metadata_table,
    feature_sets,
    feature_ids,
    contrasts,
    measures = list("globalTest", "log2FoldChange")
    ){
  
  # Create configuration table used for looping over all measure configurations
  # configurations is a data_frame type of list (named columns, nrow and ncol attributes)
  configurations <- generateConfigurations(measures, contrasts, feature_ids, feature_sets)
  n_configurations <- nrow(configurations)
  
  # Create empty output data container.
  # A named list with two entries, each named lists with keys for feature and set
  # identifiers Code Assumption: at least one group and one feature specific measure 
  # is computed (no fully empty list). 
  resultsListEnv <- listenv::listenv(
    feature_specific = constructEmptyNamedList(feature_ids),
    set_specific = constructEmptyNamedList(names(feature_sets)))
  
  # Loop through all configurations and attach relevant metadata to output container
  # Each row in configurations is dealt with separately, where the required information
  # is accessed using named column / list entries. Configurations are measure specific,
  # accounting for differences in inputs that are accessed. For example, globalTest
  # is a set specific measure, and thus requires feature_set_name and feature_set_members
  # to be available while log2foldChange is feature specific and doesn't require this
  # information.
  for (row_number in 1:n_configurations){
    current_measure <- configurations$measure[[row_number]]
    # NOTE THE INPLACE MODIFICATION OF resultsListEnv WITHIN THE HANDLERS!
    switch(
      current_measure,
      "globalTest" = runHandlerGlobalTest(
        resultsListEnv = resultsListEnv,
        feature_set_name = configurations$feature_set[[row_number]],
        feature_set_members = configurations$feature_set_members[[row_number]][[1]],
        contrast = configurations$contrast[[row_number]],
        contrast_name = names(configurations$contrast[row_number]),
        quantification_table = quantification_table, 
        metadata_table = metadata_table
      ),
      "log2FoldChange" = runHandlerLog2FoldChange(
        resultsListEnv = resultsListEnv,
        feature_id = configurations$feature_id[[row_number]],
        contrast = configurations$contrast[[row_number]],
        contrast_name = names(configurations$contrast[row_number]),
        quantification_table = quantification_table, 
        metadata_table = metadata_table
      ),
    )
  }
  # Avoids unexpected modify in place behavior for output of msfeast after return
  resultsList <- as.list(resultsListEnv)
  return(resultsList)
}

if (TRUE){
  library(dplyr)
  n_samples <- 100
  n_treatments <- 4
  n_features <- 9
  n_sets <- 3
  
  # Basic Input Expected ---------------------------------------------------------
  treatment_ids <- paste0("treatment_", 1:n_treatments)
  metadata_table <- tibble(
    sample_id = paste0("sample_", seq(1, n_samples, 1)),
    treatment = rep(treatment_ids, length.out = n_samples)
  )
  feature_ids <- paste0("feature_", seq(1:n_features))
  set_ids <- paste0("set_", seq(1:n_sets))
  quantification_table <- tibble(
    sample_id = paste0("sample_", seq(1, n_samples, 1)),
  )
  
  for (feature in feature_ids){
    quantification_table[feature] <- rnorm(n = n_samples, mean = 1000, sd = 100)
  }
  
  feature_groupings <- tibble(
    feature_id = feature_ids,
    set_id = rep(set_ids, length.out = n_features)
  )
  feature_sets <-constructEmptyNamedList(set_ids)
  for (set in set_ids){
    feature_sets[[set]] <- feature_groupings %>% filter(set_id == set) %>% pull(feature_id)
  }
  
  measures <- c("globalTest", "log2FoldChange")
  
  contrasts = list()
  ref_treat <- treatment_ids[1]
  for (iloc in 1:(length(treatment_ids)-1)){
    current_treatment <- treatment_ids[iloc+1]
    contrast_name <- paste0(ref_treat, "_vs_", current_treatment)
    contrasts[[contrast_name]] <- c(reference = ref_treat, treatment = current_treatment)
  }
  out <- run_msfeast(
    quantification_table = quantification_table, 
    metadata_table = metadata_table, 
    feature_sets = feature_sets, 
    feature_ids = feature_ids,
    contrasts = contrasts)
  #str(out)
  print("Results Inspection")
  #print(typeof(out$feature_specific$feature_1$treatment_1_vs_treatment_2))
  #print(out$set_specific$set_1)
  jsonlite::toJSON(out, pretty = T, simplifyVector = F, flatten = TRUE, auto_unbox = T)
}

