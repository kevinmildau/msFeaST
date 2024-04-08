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
