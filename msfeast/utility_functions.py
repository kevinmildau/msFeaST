import os
from warnings import warn

def assert_iloc_valid(iloc : int, iloc_range_max : int) -> None:
  """ Asserts that a provided iloc is a valid value in range (0, iloc_range_max) """
  assert iloc in [x for x in range(0, iloc_range_max)], (
    f"Error: must provide iloc in range of tsne grid 0 to {len(self.tsne_grid)}"
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