from typing import List, Union
import matchms
import copy
import pandas as pd

def apply_default_spectral_processing(
    spectra: List[matchms.Spectrum], 
    feature_identifier : str = "scans",
    minimum_number_of_fragments : int = 5,
    maximum_number_of_fragments : int = 200,
    verbose = True
  ):
  """ Function applies default spectral processing of msFeaST using matchms. 
  
  Runs:
    1. matchms default filters, 
    2. intensity normalization, 
    3. minimum_number_of_fragments filter,
    4. low intensity fragments removal if the spectrum exceeds maximum_number_of_fragments.

  Parameters
    spectra: List[matchms.Spectrum] - list of matchms spectrum objects.
    feature_identifier : str - the feature identifier name used in the data, defaults to "scans" 
    minimum_number_of_fragments : int - minimum number of fragments required for a spectrum to be kept, defaults to 5.
    maximum_number_of_fragments : int - maximum number of fragments allowed in a spectrum, defaults to 200 (any more
    are assumed noise; the lowest intensity fragments are removed until the spectrum has 200 fragments)

  Returns 
    List[matchms.Spectrum] - the cleaned list of matchms spectrum objects. If none, an assertion error is caused.
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
  """ Extracts treatment table from gnps metadata table 
  
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