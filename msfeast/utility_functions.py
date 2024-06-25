import os
from warnings import warn
import numpy as np

def assert_similarity_matrix(scores : np.ndarray, n_spectra : int) -> None:
  """ Function checks whether similarity matrix corresponds to expected formatting. Aborts code if not. """
  assert (isinstance(scores, np.ndarray)), "Error: input scores must be type np.ndarray."
  assert scores.shape[0] == scores.shape[1] == n_spectra, (
    "Error: score dimensions must be square & correspond to n_spectra"
  )
  assert np.logical_and(scores >= 0, scores <= 1).all(), "Error: all score values must be in range 0 to 1."
  return None

def assert_iloc_valid(iloc : int, iloc_range_max : int) -> None:
  """ Asserts that a provided iloc is a valid value in range (0, iloc_range_max) """
  assert iloc in [x for x in range(0, iloc_range_max)], (
    f"Error: must provide iloc in range of tsne grid 0 to {iloc_range_max}"
  )
  return None

def create_directory_if_not_exists(directory : str) -> None:
  """
  Creates a directory if it does not exist.

  Parameters:
    directory: str path to the directory.
  Returns:
    None
  """
  try:
    os.makedirs(directory, exist_ok=True)
    print(f"Directory '{directory}' created or already exists.")
  except OSError as error:
    warn(f"Directory could not be created. Following error encounted when attempting to create directory: {error}")
  return None