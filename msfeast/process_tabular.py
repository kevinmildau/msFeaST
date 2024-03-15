import pandas as pd

def validate_quantification_table(self, table : pd.DataFrame) -> None:
  """
  NOT IMPLEMENTED. 
  Function validates quantification table input against spectral data.
  """
  # Use self.spectra_matchms to get feature_ids to compare against quantification table

  # Make sure there are no NA
  # Make sure there is numeric information
  # Make sure feature_id columns are available
  # Make sure sample_id column is available
  # Make sure no non-feature_id or non-sample_id columns are there to confuse downstream functions. 
  assert True
  return None

def validate_treatment_data(self, table : pd.DataFrame):
  """
  NOT IMPLEMENTED
  Function validates treatment information input. Expects treatment information to be a pandas.DataFrame with
  sample_id and treatment_id columns. In addition, sample_ids should be identical to those used in quantification
  table (no missing, no type differences) and treatment_ids should each contain multiple instances (statistical 
  units.)
  """
  # Make sure treatment infor is pandas Data Frame, has sample_id and treatment_id columns
  # Make sure sample_id corresponds to sample_ids in quantification table, 
  # Make sure each sample has a corresponding treatment
  # Make srue treatment_id are type string
  # Make sure there are three instances of each treatment id at least, otherwise warn of possible issues in stats
  assert True
  return None