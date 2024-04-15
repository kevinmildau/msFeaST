from typing import List, Union, Tuple
import matchms
import copy
import pandas as pd
import numpy as np

def apply_default_spectral_processing(
    spectra: List[matchms.Spectrum], 
    feature_identifier : str = "scans",
    minimum_number_of_fragments : int = 5,
    maximum_number_of_fragments : int = 200,
    ion_mode : Union[str, None] = None, 
    verbose = True
  ) -> List[matchms.Spectrum]:
  """ 
  Function applies default spectral processing of msFeaST using matchms. 
  
  Runs:
    1. matchms default filters, 
    2. intensity normalization, 
    3. minimum_number_of_fragments filter,
    4. low intensity fragments removal if the spectrum exceeds maximum_number_of_fragments.

  Parameters
    spectra: list of matchms Spectrum objects.
    feature_identifier: the feature identifier name used in the spectral data file, defaults to "scans" 
    minimum_number_of_fragments: minimum number of fragments required for a spectrum to be kept, defaults to 5.
    maximum_number_of_fragments: maximum number of fragments allowed in a spectrum, defaults to 200 (any more
    are assumed noise; the lowest intensity fragments are removed until the spectrum has 200 fragments)
    ion_mode : str specifying the ion mode of the spectrum. If None (default) ionmode is not added to the spectra.
  Returns 
    The processed list of matchms spectrum objects. If none, an assertion error is caused.
  """
  tmp_spectra = copy.deepcopy(spectra)
  if verbose:
    print("Number of spectral features provided: ", len(tmp_spectra))
  tmp_spectra = [matchms.filtering.default_filters(spectrum) for spectrum in tmp_spectra] 
  tmp_spectra = [matchms.filtering.normalize_intensities(spectrum) for spectrum in tmp_spectra]
  tmp_spectra = [
    matchms.filtering.reduce_to_number_of_peaks(
      spectrum, 
      n_required= minimum_number_of_fragments, 
      n_max = maximum_number_of_fragments
    ) 
    for spectrum in tmp_spectra
  ]
  tmp_spectra = [spectrum for spectrum in tmp_spectra if spectrum is not None]
  if ion_mode is not None:
    tmp_spectra = [spectrum.set("ionmode", ion_mode) for spectrum in tmp_spectra]
  # spectrum.set uses modify in place
  [spectrum.set("feature_id", spectrum.get(feature_identifier)) for spectrum in tmp_spectra]
  if verbose:
    print("Number of spectral features which passed pre-processing: ", len(tmp_spectra))
  assert tmp_spectra != [None], "Error: no spectra left after default spectral processing!"
  return tmp_spectra

def extract_treatment_table(
    metadata_table : pd.DataFrame,
    treatment_column_name : str,
    treatment_identifiers : List[str], 
    sample_column_name : str = "filename",
    reference_category : Union[str, None] = None
    ) -> pd.DataFrame:
  """ 
  Extracts treatment table from gnps metadata table 
  
  Parameters
    metadata_table : pd.DataFrame - gnps export metadata table
    treatment_column_name : str - the column name for the treatment identifying entries (case and whitespace sensitive!)
    treatment_identifiers : List[str] - list of strings for the treatment identifiers to be extracted (case and whitespace sensitive!)
    sample_column_name : str = "filename"  - string for the sample id containing columns (case and whitespace sensitive!), defaults to 'filename'
    reference_category : Union[str, None] = None - string for the reference treatment identifier (case and whitespace sensitive!)

  Returns
    pd.DataFrame object with treatment_id and sample_id columns.
  """
  tmpdf = copy.deepcopy(metadata_table) # avoid accidental modification
  # extract relevant columns
  treatment_table = tmpdf[[sample_column_name, treatment_column_name]]
  # rename to default names
  treatment_table.columns = ["sample_id", "treatment"] # renaming columns; order given in column extraction
  # extract relevant rows
  selection_mask = treatment_table["treatment"].isin(treatment_identifiers)
  treatment_table = treatment_table[selection_mask] # extract relevant treatment entrys (row subselection)
  
  treatment_table.reset_index(drop = True, inplace=True) # remove whatever pandas index may exist inplace

  if reference_category is not None:
    # reorder df to include reference treatment in first row
    selection_mask = treatment_table["treatment"] == reference_category
    treatment_table = pd.concat(
      [treatment_table[selection_mask], treatment_table[~selection_mask]],
      ignore_index=True
    )
  treatment_table["sample_id"] = treatment_table["sample_id"].astype(dtype="string")
  return treatment_table

def subset_spectra_to_exclude(spectra : List[matchms.Spectrum], exclusion_list : List [str]):
  """ 
  Subsets spectra to exclude a particular set of feature identifiers. 

  Assumes feature_id present. Throws error if return subset is empty. Used to align spectra with quant table.
  """
  subset = [spec for spec in spectra if spec.get("feature_id") not in exclusion_list]
  assert len(subset) >= 1 and subset != [None], "Error: subsetting spectral list resulted in empty output."
  return subset

def subset_spectra_to_include(spectra : List[matchms.Spectrum], inclusion_list : List [str], verbose = True):
  """ 
  Subsets spectra to exclusively include a particular set of feature identifiers. 

  Assumes feature_id present. Throws error if return subset is empty. Used to align spectra with quant table.
  """
  subset = [spec for spec in spectra if spec.get("feature_id") in inclusion_list]
  assert len(subset) >= 1 and subset != [None], "Error: subsetting spectral list resulted in empty output."
  if verbose and len(inclusion_list) > len(subset):
    print("Warning: Not all features in inclusion list found in spectra.")
  return subset

def restructure_quantification_table(
    quantification_table : pd.DataFrame, 
    feature_id_column_name : str = "row ID", 
    sample_id_suffix : str = "Peak area") -> pd.DataFrame:
  """ 
  Parses quantification table into expected format for msFeaST. 

  Parameters
    quantification_table : pd.DataFrame, 
    feature_id_column_name : str = "row ID", 
    sample_id_suffix : str = "Peak area")

  Returns
    pd.DataFrame with sample_id and column and one column per feature_id.
  """
  ...
  quant_table = copy.deepcopy(quantification_table)
  # Extract feature id columns and any sample id columns via sample_id suffic 
  quant_table = quant_table.filter(
    regex=f"{feature_id_column_name}|{sample_id_suffix}", axis=1
  )
  quant_table = quant_table.rename(columns = {'row ID':'feature_id'})
  quant_table = quant_table.melt(id_vars="feature_id", var_name="sample_id").reset_index(drop=True)
  quant_table["feature_id"] = quant_table["feature_id"].astype(dtype="string")
  quant_table["sample_id"] = quant_table["sample_id"].astype(dtype="string")
  quant_table["sample_id"] = quant_table["sample_id"].str.replace(pat=" Peak area", repl="")
  # Pivot creates a hierarchical index, where the the columns are named feature_id, and the new index is added
  # to as a secondary column layer when resetting the index without dropping the column. This leads to the impression
  # (visually when printing) that the index is called feature_id. To avoid this, remove the name for the column index
  quant_table = pd.pivot(quant_table, columns="feature_id", index = "sample_id", values="value").reset_index()
  quant_table.columns.name = '' 
  return quant_table

def normalize_via_total_ion_current(quantification_table):
  """ 
  Apply total ion current normalization to quantification table

  For each sample (row) compute sum and divide each entry by this sum.
  """
  quant_table = copy.deepcopy(quantification_table)
  # TIC code adapted from 
  # https://github.com/Functional-Metabolomics-Lab/FBMN-STATS/blob/main/Python/Stats_Untargeted_Metabolomics_python.ipynb
  numeric_data = quant_table.drop("sample_id", axis = 1)
  numeric_data = numeric_data.apply(lambda x: x / np.sum(x), axis=1) # numeric_data.sum(axis= 1) --> all 1
  quant_table = pd.concat([quant_table["sample_id"], numeric_data], axis=1)
  return quant_table

def subset_quantification_table_to_samples(quantification_table :pd.DataFrame, sample_id_list : List[str]):
  """ 
  Subsets the quantification table to the sample identifiers provided.
  """
  quant_table = copy.deepcopy(quantification_table)
  quant_table = quant_table.query("sample_id in @sample_id_list")
  return quant_table

def generate_exclusion_list(quantification_table : pd.DataFrame) -> Union[List[str]]:
  """
  Function generates a list of feature_ids for which all data in the quantification table is 0 for exclusion.
  """
  tmp_quant = copy.deepcopy(quantification_table)
  tmp_quant = tmp_quant.drop("sample_id", axis = 1)
  zero_columns = tmp_quant.columns[(tmp_quant == 0).all()]
  if zero_columns.empty:
      print("No columns have only zero entries.")
  else:
      print(f"Number of columns with only zero entries: {len(zero_columns)}")
  all_zero_features = list(zero_columns)
  return all_zero_features

def align_feature_subsets(quantification_table : pd.DataFrame, spectra : List[matchms.Spectrum]):
  """ 
  Aligns quantification table and spectra in terms of feature_ids. Assumes spectra already filtered.
  """
  # Assumes:
  # --> treatment table is already subset, and only relevant samples are in quantification table
  # --> spectra were already subset and contain only suitable spectra (no further removal necessary)
  tmp_quant = copy.deepcopy(quantification_table)
  all_features = [spec.get("feature_id") for spec in spectra]
  exclusion_list = generate_exclusion_list(tmp_quant) # this is only the case after subsetting the samples sets.
  if exclusion_list != []:
    overlapping_features = list(set(all_features).difference(set(exclusion_list)))
  else:
    overlapping_features = all_features
  subset_spectra = subset_spectra_to_include(spectra, overlapping_features)
  column_mask = ["sample_id"] + overlapping_features
  subset_quant_table = tmp_quant[column_mask]
  return subset_quant_table, subset_spectra

def get_sample_ids_from_treatment_table(treatment_table : pd.DataFrame) -> List[str]:
  """ 
  Returns sample ids from treatment_table as a list of strings.

  Assumes the dtype of the sample_id column to be string. Does not enforce dtypes.
  """
  assert type(treatment_table) == pd.DataFrame, (
    f"Error: expected treatment_table to be type pd.DataFrame but received {type(treatment_table)}"
  )
  assert 'sample_id' in treatment_table.columns, "Error: sample_id column not found in provided data frame."
  sample_id_list = treatment_table["sample_id"].to_list()
  return sample_id_list

def align_treatment_and_quantification_table(
    treatment_table:pd.DataFrame, quantification_table:pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Function re-orders quantification table sample order to match treatment table order. 
  
  """
  treat_table = copy.deepcopy(treatment_table)
  quant_table = copy.deepcopy(quantification_table)
  treat_table.set_index("sample_id", drop=False, inplace=True)
  quant_table.set_index("sample_id", drop=False, inplace=True)
  treat_table, quant_table = treat_table.align(quant_table, join="inner", axis=0)
  treat_table.reset_index(inplace = True, drop = True)
  quant_table.reset_index(inplace = True, drop = True)
  return (treat_table, quant_table)