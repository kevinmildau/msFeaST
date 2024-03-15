from typing import List
import matchms
from warnings import warn
from file_checking import assert_filepath_exists

def load_spectral_data(filepath : str, identifier_key : str = "feature_id") -> List[matchms.Spectrum]:
  """ Loads spectra from file and validates identifier availability """
  assert_filepath_exists(filepath)
  spectra_matchms = list(matchms.importing.load_from_mgf(filepath)) # this may cause its own assert errors. 
  assert isinstance(spectra_matchms, list), "Error: spectral input must be type list[matchms.Spectrum]"
  for spec in spectra_matchms:
    assert isinstance(spec, matchms.Spectrum), "Error: all entries in list must be type spectrum."
  return spectra_matchms

def validate_spectra(
    spectra : List[matchms.Spectrum], 
    identifier_key : str = "feature_id"
  ) -> None:
  """ 
  Function validates spectral data input to match expectations. A feature_id must be available for each spectrum. Aborts
  if spectra non-conforming.

  # Make sure spectrum is not None
  # Make sure spectrum is instance matchms
  # Make sure spectrum has peaks
  # Make sure spectrum has feature_id
  """
  empty_spectra_detected = False
  for spectrum in spectra:
    assert isinstance(spectrum, matchms.Spectrum), (
      f"Error: item in loaded spectrum list is not of type matchms.Spectrum!"
    )
    assert spectrum is not None, (
      "Error: None object detected in spectrum list. All spectra must be valid matchms.Spectrum instances."
    )
    assert spectrum.get(identifier_key) is not None, (
      "Error: All spectra must have valid feature_id entries."
    )
    assert spectrum.get("precursor_mz") is not None, (
      "Error: All spectra must have valid precursor_mz value."
    )
    if spectrum.intensities is None or spectrum.mz is None: # is none also true for empty list []
      empty_spectra_detected = True
  if empty_spectra_detected:
    warn((
        "At least one spectrum provided that does not contain peak information (empty). "
        "Spectral processing required!"
      )
    )
  return None  