#!/usr/bin/env Rscript

#' @title generate_configurations
#' @description Creates configuration data frame (list) from inputs.
#' @param measures List of measures c("globalTest", "log2FoldChange") are supported. Must be a list even if scalar.
#' @param contrasts List of contrasts. Named list, where names are contrast names, and sub-list elements are treatment 
#' identifiers. For example: list("contrast_ctrl_vs_trt" = list(reference="ctrl", treatment="trt"))
#' @param feature_ids List of feature identifiers (character). For example: list("feature_1", "feature_2").
#' @param feature_sets List of feature sets, where names in the list are feature set ids (character), and list elements 
#' are lists with feature identifiers (character). For example: list("set_1" = list("feature_1", "feature_2"), ...).
#' @details 
#' generates configurations for msfeast linear situation run
#' Configurations takes the form of a data frame with columns: 
#' measure, feature_set, feature_id, contrast, feature_set_members
#' --> measure is a character string, with globalTest or log2FoldChange as an entry
#' --> feature_set is a character string or NA, with the feature_set id entry that is NA if the config measure is 
#'     log2foldChange (feature level)
#' --> feature_id is a character string or NA, with a feature_id entry that is NA if the config measure is globalTest 
#'     (set level)
#' --> contrast is a list of character strings, with the first character string indicating the reference treatment, and 
#'     the second character string to other treatment
#' @examples 
#' \dontrun{
#' ...
#' }
generate_configurations <- function(measures, contrasts, feature_ids, feature_sets){
  feature_set_names <- names(feature_sets) # Names from named list
  configurations <- data.frame()
  # globalTest & log2FoldChange are currently accepted measures.
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



#' @title run_and_attach_log2foldchange_on_feature
#' @description Runs log2foldchange scenario on specified feature and attaches results to resultsListEnv.
#' @param ...
#' @return ...
#' @details
#' Extracts relevant data to construct new scenario specific tables. Runs fold change computations. Attaches results
#' to resultsListEnv in place! (no copy behavior since env is used)
#' @examples 
#' \dontrun{
#' ...
#' }
#' 
run_and_attach_log2foldchange_on_feature <- function(
  resultsListEnv, feature_id, contrast, contrast_name, quantification_table, metadata_table){
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
  
  referenceIntensities <- tmpAllData %>%
    filter(., treatment == 0) %>%
    select(., -c(treatment, sample_id)) %>%
    pull(.)
  
  treatmentIntensities <- tmpAllData %>%
    filter(., treatment == 1) %>%
    select(., -c(treatment, sample_id)) %>%
    pull(.)
  
  ratio <- mean(treatmentIntensities) / mean(referenceIntensities)
  log2ratio <- log2(ratio)

  tmpFeatureData <- tmpAllData %>% 
    select(-c("sample_id", "treatment")) %>%
    as.matrix(.)
  
  # Attach to output
  resultsListEnv$"feature_specific"[[feature_id]][[contrast_name]]["log2FoldChange"] <- list(log2ratio)
}



#' @title run_and_attach_global_test_on_feature_set
#' @description Runs globaltest scenario on specified feature set and attaches results to resultsListEnv.
#' @param ...
#' @return ...
#' @details
#' Extracts relevant data to construct new scenario specific tables. Runs globaltest computations. Attaches results
#' to resultsListEnv in place! (no copy behavior since env is used)
#' @examples 
#' \dontrun{
#' ...
#' }
#' 
run_and_attach_global_test_on_feature_set <- function(
  resultsListEnv, 
  feature_set_name, 
  feature_set_members, 
  contrast, 
  contrast_name, 
  quantification_table, 
  metadata_table
  ){
  # Extract contrast specific data
  contrast <- contrasts[[contrast_name]]
  feature_ids <- feature_sets[[feature_set_name]]
  # get sample_ids
  treatment_reference_id <- contrast[[1]]
  treatment_treatment_id <- contrast[[2]]

  tmpMetaData <- metadata_table[
    which(metadata_table$treatment %in% c(treatment_reference_id, treatment_treatment_id)),
  ]
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

  # extraction of results leads to the construction of an RPlots.pdf
  table <- extract(covariates(model_output))
  extras_table <- table@extra
  results_table <- table@result

  # assert output tables  as expected
  if (! is.data.frame(extras_table)){stop(paste("Expected data frame but received", typeof(extras_table)))}
  if (! is.matrix(results_table)){stop(paste("Expected matrix but received", typeof(results_table)))}
  # Attach to output
  resultsListEnv$"set_specific"[[feature_set_name]][[contrast_name]]["globalTestPValue"] <- list(p_value) 

  # Feature specific global test results are mocked!
  for (feature_id in feature_set_members){
    # Extract effect direction for individual feature as pos or neg character string
    # rownames persist for arbitrary table size, alias column of globaltest disappears when nrow == 1
    tmp_index <- which(rownames(extras_table) == feature_id)
    effect_direction <- substr(
      extras_table[tmp_index, , drop = FALSE]$direction,
      start = 1, stop = 3
    )
    # Extract pvalue and statistic
    tmp_index <- which(rownames(results_table) == feature_id)
    feature_specific_p_value <- results_table[tmp_index, ]["p-value"] # matrix & named vector accessing
    feature_specific_statistic <- results_table[tmp_index, ]["Statistic"] # matrix & named vector accessing
    # Attach data 
    resultsListEnv$"feature_specific"[[feature_id]][[contrast_name]]["globalTestFeaturePValue"] <- list(
      feature_specific_p_value
    )
    resultsListEnv$"feature_specific"[[feature_id]][[contrast_name]]["globalTestFeatureStatistic"] <- list(
      feature_specific_statistic
    )
    resultsListEnv$"feature_specific"[[feature_id]][[contrast_name]]["globalTestFeatureEffectDirection"] <- list(
      effect_direction
    )
  }
}



#' @title run_msfeast
#' @description Main interface function for msFeaST. This function provides a convenience wrapper to the globaltest 
#' (v5.50.0) package.
#' Given the quantification table, metadata_table, feature_sets, contrasts and measures, the function
#'  1. Initialize configurations for each test
#'  2. Initialize output data structures (hierarchical named lists for json export (based on env)) 
#'  3. For each configuration:
#'    -> extract and construct relevant tables
#'    -> run the test handler -> add results to respective output structures
#'  4. Return results output structures as nested named lists and json strings
#' @param quantification_table A tibble (data frame) with a sample_id column (character) and a column for each 
#' feature_id containing the respective measurement values (float)
#' @param metadata_table A tibble with a sample_id column (character) and a condition_id column (character). The 
#' condition_id is expected to contain both the control/reference group and treatment/comparison groups. It is used to 
#' extract relevant samples for each contrast.
#' @param feature_sets A named list with keys representing feature_set_id (character) and sub-list elements representing 
#' feature_id (character)
#' @param feature_ids A list with feature_ids (character)
#' @param contrasts A list of 2-tuples containing the control/reference category and the treatment/comparison category 
#' (character). 
#' @param measures A list of measures to use for comparative purposes. These must match one or more of the following 
#' supported measures: c("globalTest", "log2FoldChange")
#' @return
#' @Details Data assumptions:
#' --> all feature_id, sample_id, feature_set_id, condition_id values must be unique in their respective columns.
#' --> column names must exactly match expectations posited
#' \dontrun{
#' ...
#' }
run_msfeast <- function( quantification_table, metadata_table, feature_sets, feature_ids, contrasts){
  
  measures = list("globalTest", "log2FoldChange")

  # Create configuration table used for looping over all measure configurations
  # configurations is a data_frame type of list (named columns, nrow and ncol attributes)
  configurations <- generate_configurations(measures, contrasts, feature_ids, feature_sets)
  n_configurations <- nrow(configurations)
  # Create empty output data container.
  # A named list with two entries, each named lists with keys for feature and set
  # identifiers Code Assumption: at least one group and one feature specific measure 
  # is computed (no fully empty list). 
  resultsListEnv <- listenv::listenv(
    feature_specific = construct_empty_named_list(feature_ids),
    set_specific = construct_empty_named_list(names(feature_sets)))


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
      "globalTest" = run_and_attach_global_test_on_feature_set(
        resultsListEnv = resultsListEnv,
        feature_set_name = configurations$feature_set[[row_number]],
        feature_set_members = configurations$feature_set_members[[row_number]][[1]],
        contrast = configurations$contrast[[row_number]],
        contrast_name = names(configurations$contrast[row_number]),
        quantification_table = quantification_table, 
        metadata_table = metadata_table
      ),
      "log2FoldChange" = run_and_attach_log2foldchange_on_feature(
        resultsListEnv = resultsListEnv,
        feature_id = configurations$feature_id[[row_number]],
        contrast = configurations$contrast[[row_number]],
        contrast_name = names(configurations$contrast[row_number]),
        quantification_table = quantification_table, 
        metadata_table = metadata_table
      ),
    )
  }
  # quantification_table, metadata_table, feature_sets, feature_ids, contrasts, 
  # measures = list("globalTest", "log2FoldChange")

  # Attach easy of parsing variables
  
  resultsListEnv$feature_id_keys <- feature_ids
  tmp_contrasts <- names(contrasts)
  if (! is.list(tmp_contrasts)){
    tmp_contrasts <- list(tmp_contrasts)
  }
  resultsListEnv$contrast_keys <- tmp_contrasts
  resultsListEnv$set_id_keys <- names(feature_sets)
  resultsListEnv$feature_specific_measure_keys <- c(
    "globalTestFeaturePValue",  "globalTestFeatureStatistic", "globalTestFeatureEffectDirection", "log2FoldChange"
  )
  resultsListEnv$set_specific_measure_keys <- list("globalTestPValue")

  
  # Avoids unexpected modify in place behavior for output of msfeast after return
  resultsList <- as.list(resultsListEnv)
  return(resultsList)
}



#' @title validate_input_filepaths
#' @description Function ensures that provided input aligns with msfeast.R expectations
#' @param input_filepaths list of character string entries with filepaths to use in msfeast.
#'    quantification_table, treatment_table, assignment_table, and the R output json location.
#' @return ...
#' @details ...
#' @examples 
#' \dontrun{
#' ...
#' }
validate_input_arguments <- function(input_filepaths){
  # Function checks whether provided input filepaths (list of character) points to existing files. Stops if not.
  if (length(input_filepaths) != 4){stop(sprintf("Expected 4 arguments, but received %s", length(input_filepaths)))}
  for (filepath in input_filepaths[1:3]){
    if (!file.exists(filepath)){stop(paste("Provided filepath ", filepath, "does not exist!"))}
  }
  attempt_file_creation(input_filepaths[4])
}



#' @title attempt_file_creation
#' @description Attempts creating a file in specified filepath.
#' @param filepath A character specifying a filepath.
#' @return TRUE if successful, stops if not.
#' @details ...
#' @examples 
#' \dontrun{
#' ...
#' }
attempt_file_creation <- function(filepath) {
  # Attempt to open the file for writing
  if (file.exists(filepath)){
    stop(
      paste(
        "File with path", 
        filepath, 
        "already exists! Remove, rename, or specify alternative output filepath to run msfeast."
      )
    )
  }
  if (!file.exists(filepath)){
    tryCatch({
      write("", file = filepath)
    }, error = function(e) {
      stop("Error in file creation attempt.")
    })
  }
  return(TRUE)
}



#' @title construct_empty_named_list
#' @description Creates a named list using names from given list of entry names. Each entry contains an empty list. This
#' is used as a generic data container.
#' @param param A character vector; cannot be scalar.
#' @return Returns list with specified entries containing empty lists each.
#' @details 
#' @examples 
#' \dontrun{
#' ...
#' }
construct_empty_named_list <- function(names_list){
  # Assert correct inputs
  if(! (is.character(names_list) && length(names_list) >= 1)){
    stop(paste("Expected character vector input but received", typeof(names_list)))
  }
  if(!all(sapply(names_list, is.character))){ 
    stop(paste(
        "Expected list of character entries but received other entry types:", 
        paste(sapply(names_list, typeof), collapse = " ")
      )
    )
  }
  # Create the named list
  emptyList <- sapply(names_list, function(x) list())
  return (emptyList)
}



generateFeatureSetList <- function(feature_groupings_df){
  # Assert input is tibble
  if(!is_tibble(feature_groupings_df)){
    stop(paste("Expected tibble, but received ", typeof(feature_groupings_df)))
  }
  set_ids <- unique(feature_groupings_df$set_id)
  feature_sets <- construct_empty_named_list(set_ids)
  for (set in set_ids){
    feature_sets[[set]] <- feature_groupings_df %>% filter(set_id == set) %>% pull(feature_id)
  }
  # 
  if(! length(feature_sets) == length(set_ids)) {
    stop("Expected that length(feature_sets) == length(set_ids).")
  }
  return(feature_sets)
}



#' @title load_libraries
#' @description Function attempts to load libraries and returns simple error message if not found.
load_libraries <- function(){
  tryCatch(
    {suppressPackageStartupMessages(library("readr"))}, 
    error = function(error_message){
      print("R Routine: ERROR: readr package Not Found. Stopping Code Execution."); 
      q();
    }
  )
  tryCatch(
    {suppressPackageStartupMessages(library("dplyr"))}, 
    error = function(error_message){
      print("R Routine: ERROR: dplyr package Not Found. Stopping Code Execution.");
      q();
    }
  )
  tryCatch(
    {suppressPackageStartupMessages(library("tibble"))}, 
    error = function(error_message){
      print("R Routine: ERROR: tibble package Not Found. Stopping Code Execution."); 
      q();
    }
  )
  tryCatch(
    {suppressPackageStartupMessages(library("globaltest"))}, 
    error = function(error_message){
      print("R Routine: ERROR: globaltest package Not Found. Stopping Code Execution."); 
      q();
    }
  )
  return (TRUE)
}



#' @title create_contrasts_list
#' @description Creates a named list with contrast names as keys, and character vector containing the respective
#' contrast specific character strings for accessing.
#' @param treatment_ids character vector with treatment identifiers
#' @param ref_treat  a single treatment identifier character to be used as reference
#' @return Named list with contrast name and constituent treatment ids, e.g. : 
#' list("treat_1_vs_treat_2" = c("reference" = "treat_1", "treatment" = "treat_2"), ...)
#' @details ...
#' @examples 
#' \dontrun{
#' ...
#' }
create_contrasts_list <- function(treatment_ids, ref_treat){
  # check input requirement number of treatments 2 or larger.
  if (length(treatment_ids) < 2){stop(paste("Expected 2 treatments or more but received", length(treatment_ids)))}
  contrasts = list()
  for (iloc in 1:(length(treatment_ids)-1)){
    current_treatment <- treatment_ids[iloc+1]
    contrast_name <- paste0(ref_treat, "_vs_", current_treatment)
    contrasts[[contrast_name]] <- c(reference = ref_treat, treatment = current_treatment)
  }
  return(contrasts)
}



run_integration_test <- function(){ 
  # test package loading
  # test single globaltest instance with known result
  # test single foldchange instance with known result
  # test a mock demo scenario with all elements checking out 
  return(TRUE)
}



########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# runs only if script called by itself, equivalent to python if __name__ == __main__
if (sys.nframe() == 0){
  pdf(NULL) # prevent pdf creation because of globaltest unsuppressable png creation
  print(c("Starting Routine log at ", as.character(Sys.time())))
  measures <- c( "log2FoldChange", "globalTest") # constant

  print("R Routin: run integration test...")
  run_integration_test() # <-- currently empty

  print("R Routine: Validating input file paths...")
  input_filepaths <- commandArgs(trailingOnly=TRUE)
  # file.remove(input_filepaths[4])
  validate_input_arguments(input_filepaths)
  
  print("R Routine: Loading required packages...")
  load_libraries()

  ######################################################################################################################
  print("R Routine: Reading input files...")
  quantification_table <- read_delim(input_filepaths[1], delim = ",", show_col_types = FALSE) %>%
    rename_all(~ as.character(.)) %>% # ensure that feature_id column names are character!
    mutate(sample_id = as.character(sample_id)) # enfore character entries

  treatment_table <- read_delim(input_filepaths[2], delim = ",", show_col_types = FALSE) %>%
    mutate_all(as.character) # enfore character entries
  treatment_ids = unique(treatment_table$treatment)
  ref_treat <- treatment_ids[1]
  contrasts <- create_contrasts_list(treatment_ids, ref_treat)

  assignment_table <- read_delim(input_filepaths[3], delim = ",", show_col_types = FALSE) %>%
    mutate_all(as.character) # enfore character entries
  feature_ids <- assignment_table$feature_id
  feature_sets <- generateFeatureSetList(assignment_table)
  
  joint_validate_input_tables <- function(quantification_table, treatment_table, assignment_table){
    # NOT IMPLEMENTED
  }

  ######################################################################################################################
  print("R Routine: running global test and fold change computations...")
  
  out <- run_msfeast(
    quantification_table = quantification_table, 
    metadata_table = treatment_table, 
    feature_sets = feature_sets, 
    feature_ids = feature_ids,
    contrasts = contrasts)

  print("R Routine: exporting globaltest and log fold change computations...")

  json_output <- jsonlite::toJSON(out, pretty = T, simplifyVector = F, flatten = TRUE, auto_unbox = T)
  writeLines(json_output, input_filepaths[4])

  print("R Routine: complete, file saved, exiting R session." )
}


