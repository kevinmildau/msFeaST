import os

def assert_filepath_valid(filepath : str) -> None:
  """ 
  Helper Function checks whether the provided filepath is valid (str, in existing folder etc.), the file doesn't need to 
  exist. The function raises an assert error if not.
  """
  assert isinstance(filepath, str), f"Error: expected filepath to be string but received {type(filepath)}"
  assert os.path.isfile(filepath), "Error: supplied filepath is not valid."
  return None

def assert_filepath_exists(filepath : str) -> None:
  """ 
  Helper Function checks whether the provided filepath is valid and exists. The function raises an assert error if not. 
  """
  assert isinstance(filepath, str), f"Error: expected filepath to be string but received {type(filepath)}"
  assert os.path.exists(filepath), "Error: supplied filepath does not point to existing file."
  return None

def assert_directory_exists(directory : str) -> None:
  """ 
  Helper Function checks whether the provided filepath is valid and exists. The function raises an assert error if not. 
  """
  assert isinstance(directory, str), f"Error: expected directory to be string but received {type(directory)}"
  assert os.path.isdir(directory), "Error: model directory path must point to an existing directory!"
  return None