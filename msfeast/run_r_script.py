import os
import subprocess
from r_output_parsing import load_and_validate_r_output

def run_and_attach_r_testing_routine(self, directory : str, r_filename : str = "r_output.json", top_k = 20, overwrite = False):
    """
    INCOMPLETE
    Function writes r inputs to file, writes r script to file, tests r availabiltiy, runs test, and imports r results.
    directory: folder name for r output
    r_filename: filename for r output

    """
    # assess all data required available
    # construct python structures suitable for file writing
    
    self.assignment_table # -->
    self.quantification_table # --> 
    self.treatment_table # --> turn into contrasts
    
    # json filenames input for R; the one argument to pass to R so it can find all the rest
    # could also contain all info, but that makes r parsing more difficult than reading csvs...

    filepath_assignment_table = str(os.path.join(directory, "assignment_table.csv"))
    filepath_quantification_table = str(os.path.join(directory, "test_quant_table.csv"))
    filepath_treatment_table = str(os.path.join(directory, "test_treat_table.csv"))
    filepath_r_output_json = str(os.path.join(directory, r_filename))

    # TODO check for directory exist, if not, create it
    # TODO check for existing r routine output, remove if force = true


    # Write R input data to file
    self.quantification_table = self.quantification_table.reset_index(drop=True)
    self.treatment_table = self.treatment_table.reset_index(drop=True)
    self.assignment_table = self.assignment_table.reset_index(drop=True)
    
    # Required to remove the unnamed = 0 index column that is created somewhere
    self.quantification_table.drop(
      self.quantification_table.columns[
          self.quantification_table.columns.str.contains('Unnamed', case=False)], 
      axis=1, inplace=True
    )
    self.treatment_table.drop(
      self.treatment_table.columns[
        self.treatment_table.columns.str.contains('Unnamed', case=False)], 
      axis=1, inplace=True
    )
    self.assignment_table.drop(
      self.assignment_table.columns[
        self.assignment_table.columns.str.contains('Unnamed', case=False)], 
      axis=1, inplace=True
    )

    # Setting index to false is often not enough for pandas to remove it as the index is sometimes added as an unnamed 
    # column
    self.quantification_table.to_csv(filepath_quantification_table, index = False)
    self.treatment_table.to_csv(filepath_treatment_table, index = False)
    self.assignment_table.to_csv(filepath_assignment_table, index = False)

    # Fetch r script filepath
    r_script_path = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), # module directory after pip install
      "runStats.R" # filename included as package data
    )
    
    # Run R Code
    subprocess.run((
        f"Rscript {r_script_path} {filepath_quantification_table} " 
        f"{filepath_treatment_table} " 
        f"{filepath_assignment_table} "
        f"{filepath_r_output_json}"
      ), 
      shell=True
    )
    # load r data
    r_json_data = load_and_validate_r_output(filepath_r_output_json)
    # construct derived variables and attach
    self._generate_and_attach_long_format_r_data(r_json_data)
    self._generate_and_attach_json_dict(r_json_data, top_k)
    return None