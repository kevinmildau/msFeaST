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