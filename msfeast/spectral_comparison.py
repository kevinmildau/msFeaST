import matchms
from typing import Union, List
from warnings import warn
import numpy as np
import os

# Dev Note: imports for ms2deepscore and spec2vec are currently in calling functions to avoid dependency problems.

def compute_similarities_wrapper(
  spectra : List[matchms.Spectrum], 
  score_name : str = "ModifiedCosine",
  model_directory: Union[str, None] = None
  ) -> Union[np.ndarray, None]:
  """ 
  Functions delegates similarity computations to appropriate score specific function. 
  """
  if score_name in ["ms2deepscore", "spec2vec"] and model_directory is None:
    warn(
      "Warning: when using ML based scores a model directory with pre-trained models needs to be provided. Not run."
    )
    return None
  if score_name not in ["ms2deepscore", "spec2vec", "ModifiedCosine", "CosineHungarian", "CosineGreedy"]:
    warn(f"The provided score {score_name} is not supported.")
    return None
  if score_name == "ms2deepscore":
    similarity_array = _compute_similarities_ms2ds(spectra, model_directory)
  if score_name == "spec2vec":
    similarity_array = _compute_similarities_s2v(spectra, model_directory)
  if score_name in ["ModifiedCosine", "CosineHungarian", "CosineGreedy"]:
    similarity_array = _compute_similarities_cosine(spectra, cosine_type="ModifiedCosine")
    # if successful, upade object and attach similarities
  return similarity_array

def convert_similarity_to_distance(similarity_matrix : np.ndarray) -> np.ndarray:
  """ 
  Converts pairwise similarity matrix to distance matrix with values between 0 and 1. Assumes that the input is a
  similarity matrix with values in range 0 to 1 up to floating point error.
  """
  distance_matrix = 1.- similarity_matrix
  distance_matrix = np.round(distance_matrix, 6) # Round to deal with floating point issues
  distance_matrix = np.clip(distance_matrix, a_min = 0, a_max = 1) # Clip to deal with floating point issues
  return distance_matrix

def _compute_similarities_cosine(
    spectrum_list:List[matchms.Spectrum], 
    cosine_type : str = "ModifiedCosine"
    ) -> np.ndarray:
    """ Function computes pairwise similarity matrix for list of spectra using specified cosine score. 
    
    Parameters:
        spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
        cosine_type: String identifier of supported cosine metric, options: ["ModifiedCosine", "CosineHungarian", 
        "CosineGreedy"]
    Returns:
        ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
    """
    valid_types = ["ModifiedCosine", "CosineHungarian", "CosineGreedy"]
    assert cosine_type in valid_types, f"Cosine type specification invalid. Use one of: {str(valid_types)}"
    if cosine_type == "ModifiedCosine":
        similarity_measure = matchms.similarity.ModifiedCosine()
    elif cosine_type == "CosineHungarian":
        similarity_measure = matchms.similarity.CosineHungarian()
    elif cosine_type == "CosineGreedy":
        similarity_measure = matchms.similarity.CosineGreedy()
    tmp = matchms.calculate_scores(
      spectrum_list, spectrum_list, similarity_measure, is_symmetric=True, array_type = "numpy"
    )
    scores = _extract_similarity_scores_from_matchms_cosine_array(tmp.to_array())
    scores = np.clip(scores, a_min = 0, a_max = 1) 
    return scores

def assert_similarity_matrix(scores : np.ndarray, n_spectra : int) -> None:
  """ Function checks whether similarity matrix corresponds to expected formatting. Aborts code if not. """
  assert (isinstance(scores, np.ndarray)), "Error: input scores must be type np.ndarray."
  assert scores.shape[0] == scores.shape[1] == n_spectra, (
    "Error: score dimensions must be square & correspond to n_spectra"
  )
  assert np.logical_and(scores >= 0, scores <= 1).all(), "Error: all score values must be in range 0 to 1."
  return None

def _extract_similarity_scores_from_matchms_cosine_array(
  tuple_array : np.ndarray
  ) -> np.ndarray:
  """ 
  Function extracts similarity matrix from matchms cosine scores array.
  
  The cosine score similarity output of matchms stores output in a numpy array of pair-tuples, where each tuple 
  contains (sim, n_frag_overlap). This function extracts the sim scores, and returns a numpy array corresponding to 
  pairwise similarity matrix.

  Parameters:
      tuple_array: A single matchms spectrum object.
  Returns:  
      A np.ndarray with shape (n, n) where n is the number of spectra deduced from the dimensions of the input
      array. Each element of the ndarray contains the pairwise similarity value.
  """
  sim_data = [ ]
  for row in tuple_array:
    for elem in row:
      sim_data.append(float(elem[0]))
  return(np.array(sim_data).reshape(tuple_array.shape[0], tuple_array.shape[1]))

def _return_model_filepath(
  path : str, 
  model_suffix:str
  ) -> str:
  """ Function parses path input into a model filepath. If a model filepath is provided, it is returned unaltered , if 
  a directory path is provided, the model filepath is searched for and returned.

  :param path: File path or directory containing model file with provided model_suffix.
  :param model_suffix: Model file suffix (str)
  :returns: Filepath (str).
  :raises: Error if no model in file directory or filepath does not exist. Error if more than one model in directory.
  """
  filepath = []
  if path.endswith(model_suffix):
    # path provided is a model file, use the provided path
    filepath = path
    assert os.path.exists(filepath), "Provided filepath does not exist!"
  else:
    # path provided is not a model filepath, search for model file in provided directory
    for root, _, files in os.walk(path):
      for file in files:
        if file.endswith(model_suffix):
          filepath.append(os.path.join(root, file))
    assert len(filepath) > 0, f"No model file found in given path with suffix '{model_suffix}'!"
    assert len(filepath) == 1, (
    "More than one possible model file detected in directory! Please provide non-ambiguous model directory or"
    "filepath!")
  return filepath[0]

def _compute_similarities_ms2ds(
  spectrum_list:List[matchms.Spectrum], 
  model_path:str
  ) -> np.ndarray:
  """ Function computes pairwise similarity matrix for list of spectra using pretrained ms2deepscore model.
  
  Parameters
      spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
      model_path: Location of ms2deepscore pretrained model file path (filename ending in .hdf5 or file-directory)
  Returns: 
      ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
  """
  from ms2deepscore import MS2DeepScore
  from ms2deepscore.models import load_model  
  import ms2deepscore # This makes the dependency issue only a problem when this function is called
  filename = _return_model_filepath(model_path, ".hdf5")
  model = ms2deepscore.models.load_model(filename) # Load ms2ds model
  similarity_measure = ms2deepscore.MS2DeepScore(model)
  scores_matchms = matchms.calculate_scores(
    spectrum_list, spectrum_list, similarity_measure, is_symmetric=True, array_type="numpy"
  )
  scores_ndarray = scores_matchms.to_array()
  scores_ndarray = np.clip(scores_ndarray, a_min = 0, a_max = 1) # Clip to deal with floating point issues
  return scores_ndarray

def _compute_similarities_s2v(
  spectrum_list:List[matchms.Spectrum], 
  model_path:str,
  apply_linear_transform : bool = True
  ) -> np.ndarray:
  """ Function computes pairwise similarity matrix for list of spectra using pretrained spec2vec model. Uses linear
  transform to case similarities to range 0 to 1. 
  
  Parameters:
    spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
    model_path: Location of spec2vec pretrained model file path (filename ending in .model or file-directory)
    apply_linear_transform : Recasts spec2vec scores from -1 to 1 range to 0 to 1 range.
  Returns: 
    ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
  """
  from spec2vec import Spec2Vec
  import gensim
  filename = _return_model_filepath(model_path, ".model")
  model = gensim.models.Word2Vec.load(filename) # Load s2v model
  similarity_measure = Spec2Vec(model=model)
  scores_matchms = matchms.calculate_scores(
    spectrum_list, spectrum_list, similarity_measure, is_symmetric=True, array_type="numpy"
  )
  scores_ndarray = scores_matchms.to_array()
  if apply_linear_transform:
    scores_ndarray = (scores_ndarray + 1) / 2 # linear scaling from range -1 to 1 to 0 to 1
  scores_ndarray = np.clip(scores_ndarray, a_min = 0, a_max = 1) # Clip to deal with floating point issues
  return scores_ndarray