from typing import List
import matchms
from warnings import warn
from msfeast.file_checking import assert_filepath_exists

def process_spectra(self, spectra):
  """
  NOT IMPLEMENTED
  Runs default matchms filters on spectra to improve spectrals similarity computations and derived processes.
  """
  # after processing: 
  # check feature_ids is not empty, validate spectra
  # get feature_id subset
  # restrict quantification table to feature_id set (extract only relevant features)
  # validate quantification table
  # update spectra and quantification table
  return None

def add_feature_id_key(spectra : List[matchms.Spectrum], identifier_key : str):
  """ Function add feature_id key to all spectra in list using entry from identifier_key. """
  for spectrum in spectra:
    assert spectrum.get(identifier_key) is not None, "Error provided identifier key does not point to valid id!"
    spectrum.set(key = "feature_id", value = spectrum.get(identifier_key))
  return spectra

def extract_feature_ids_from_spectra(spectra : List[matchms.Spectrum]) -> List[str]:
  """ Extract feature ids from list of matchms spectra in string format. """
  # Extract feature ids from matchms spectra. 
  feature_ids = [str(spec.get("feature_id")) for spec in spectra]
  assert_feature_ids_valid(feature_ids)
  return feature_ids

def assert_feature_ids_valid(feature_ids : List[str]) -> None:
  """ Makes sure that feature_id entries are valid and unique."""
  assert isinstance(feature_ids, list), f"Error: expect feature_ids to be type list but received {type(feature_ids)}!"
  assert not feature_ids == [], "Error: Expected feature ids but received empty list!"
  assert not any(feature_ids) is None and not all(feature_ids) is None, (
    "Error: None type feature ids detected! All feature ids must be type string."
  )
  assert all(isinstance(x, str) for x in feature_ids), (
    "Error: Non-string feature_ids detected. All feature_ids for spectra must be valid string type."
  )
  assert not (len(feature_ids) > len(set(feature_ids))), (
    "Error: Non-unique (duplicate) feature_ids detected. All feature_ids for spectra must be unique strings."
  )
  return None

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