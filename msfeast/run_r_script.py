import os
import subprocess
import json

def run_statistics_routine(
    directory : str, 
    r_filename : str,
    quantification_table, 
    treatment_table, 
    assignment_table,
  ):
  """
  R interface function that calls R shell script doing statistical comparisons.

  Function writes r inputs to file, writes r script to file, tests r availabiltiy, runs test, and imports r results.
  
  directory: folder name for r output
  r_filename: filename for r output
  """
  filepath_quantification_table = str(os.path.join(directory, "msfeast_r_quant_table.csv"))
  filepath_treatment_table = str(os.path.join(directory, "msfeast_r_treat_table.csv"))
  filepath_assignment_table = str(os.path.join(directory, "msfeast_r_assignment_table.csv"))
  write_table_to_file(quantification_table, filepath_quantification_table)
  write_table_to_file(treatment_table, filepath_treatment_table)
  write_table_to_file(assignment_table, filepath_assignment_table)
  
  filepath_r_output_json = str(os.path.join(directory, r_filename))

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
    shell = True
  )
  # load r data
  r_json_data = load_r_output(filepath_r_output_json)
  # construct derived variables and attach
  return r_json_data

def write_table_to_file(table, filepath) -> None:
  """ Function applies tabular processing for csv writing compatible with R. """
  table = table.reset_index(drop=True)
  table.drop(
    table.columns[
      table.columns.str.contains('Unnamed', case=False)
    ], 
    axis=1, 
    inplace=True
  )
  table.to_csv(filepath, index = False)
  return None

def load_r_output(filepath : str) -> dict:
  """ Function loads and validates r output file.
  Returns the r output json data as a python dictionary. First level entries are:
  
  feature_specific
  --> feature id specific data, subdivided into contrast specific, measure specific, and finally value. I.e. for each
  feature id, for each contrast key, for each measure key, there will be a corresponding value in a nested dict
  of hierarchy [feature_identifier][contrast_key][measure_key] --> value. Feature_identifier, contrast_key, and
  measure keys are data dependent strings. The hierarchy gives the type of entry.
  
  set_specific
  --> set id specific data, subdivided into contrast specific, measure specific, and finally value
  feature_id_keys. Similar to feature_id.
  
  set_id_keys
  --> list of set identifiers
  
  contrast_keys
  --> list of contrast keys
  
  feature_specific_measure_keys
  --> list of measure keys for the feature specific entry
  
  set_specific_measure_keys
  --> list of measure keys for the set specific entries
  """
  json_data = json.load(open(filepath, mode = "r"))
  # Assert that the top level keys are all populated (partial input assertion testing only!)
  assert json_data["feature_specific"]is not None, "ERROR: Expected feature_specific  entry to not be empty."
  assert json_data["feature_specific"].keys() is not None, "ERROR: Expected feature specific keys."
  assert json_data["set_specific"] is not None, "ERROR: Expected set_specific  entry to not be empty."
  assert json_data["set_specific"].keys() is not None, "ERROR: Expected set specific keys."
  assert json_data["feature_id_keys"] is not None, "ERROR: Expected feature id keys entry to not be empty."
  assert json_data["set_id_keys"] is not None, "ERROR: Expected feature id keys entry to not be empty."
  assert json_data["contrast_keys"] is not None, "ERROR: Expected contrast_keys entry to not be empty."
  assert json_data["feature_specific_measure_keys"] is not None, "ERROR: Expected feature_specific_measure_keys to not be empty."
  assert json_data["set_specific_measure_keys"] is not None, "ERROR: Expected set_specific_measure_keys entry to not be empty."
  # TODO: for robustness, Cross compare R entries against python data from pipeline (contrasts, setids, fids)
  # TODO: for robustness, Validate each feature_id and set_id entry
  # return the validated data
  return json_data