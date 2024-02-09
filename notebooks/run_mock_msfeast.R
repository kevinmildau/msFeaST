#!/usr/bin/env Rscript

file = "r-log-file.txt"
print(c("Starting Routine log at ", as.character(Sys.time())))

print("R Routine: Reading Python Input File Paths...")

# Read input arguments
input_arguments <- commandArgs(trailingOnly=TRUE) # returns list of arguments? 

# Assert all expected file paths provided
if (length(input_arguments) != 3){stop(sprintf("Expected 3 arguments, but received %s", length(input_arguments)))}



print("R Routine: arguments read. Filepaths are: ")
print(input_arguments)


print("R Routine: Attempting to load readr library...")

tryCatch(
  {
    library("readr")
  }, 
  error = function(error_message){
    print("R Routine: ERROR: readr package Not Found. Stopping Code Execution.")
    print(as.character(error_message))
    q()
  }
)

quantification_table <- read_delim(input_arguments[1], delim = ",", )

print(head(quantification_table))

treatment_table <- read_delim(input_arguments[2], delim = ",")

print(head(treatment_table))

assignment_table <- read_delim(input_arguments[2], delim = ",")

print(head(assignment_table))

print("R Routine: Reading Python Input File Paths...")

print("R Routine: Attempting to load globaltest library...")

tryCatch(
  {
    library("globaltest")
  }, 
  error = function(error_message){
    print("R Routine: ERROR: globaltest package Not Found. Stopping Code Execution.")
    print(as.character(error_message))
    q()
  }
)

print("R Routine: globaltest loading complete...")

print("R Routine: loading files needed for R code executions...")

print("R Routine: running global test and fold change computations...")


print("R Routine: exporting globaltest and log fold change computations...")

print("R Routine: complete, exiting R session." )


