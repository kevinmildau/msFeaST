from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matchms
import typing
import os
from typing import List, TypedDict, Tuple, Dict, NamedTuple, Union
from warnings import warn
import copy # for safer get methods  pipeline internal variables

# spec2vec dependencies
from spec2vec import Spec2Vec
import gensim

# ms2deepscore currently not working with macos m1 processors. Only include if ready to test in windows / linux.
# ms2deepscore dependencies (does not appear to work for Python 3.10 ; check compatible python version)
# from ms2deepscore import MS2DeepScore
# from ms2deepscore.models import load_model



@dataclass
class GridEntryTsne:
  """ 
  Container Class for t-SNE embedding optimization results. Contains a single entry. A list of these containers can be 
  converted to pandas for easy display.

  Parameters:
    perplexity : int with perplexity value used in t-SNE optimization.
    x_coordinates : List[int] x coordinates produced by t-SNE
    y_coordinates:  List[int] y coordinates produced by t-SNE
    pearson_score : float representing the pearson correlation between pairwise distances in embedding and 
      high dimensional space.
    spearman_score : float representing the spearman correlation between pairwise distances in embedding and 
      high dimensional space.
    random_seed_used : int or float with the random seed used in k-medoid clustering.
  """
  perplexity : int
  x_coordinates : List[int]
  y_coordinates:  List[int]
  pearson_score : float
  spearman_score : float
  random_seed_used : Union[int, float]
  def __str__(self) -> str:
    custom_print = (
      f"Perplexity = {self.perplexity}," 
      f"Pearson Score = {self.pearson_score}, "
      f"Spearman Score = {self.spearman_score}, \n"
      f"x coordinates = {', '.join(self.x_coordinates[0:4])}...",
      f"y coordinates = {', '.join(self.y_coordinates[0:4])}...")
    return custom_print
@dataclass
class GridEntryUmap:
  """ 
  NOT IMPLEMENTED
  Results container for umap grid computation. A list of these containers can be converted to pandas for easy display.
  """
  ...
@dataclass
class GridEntryKmedoid:
    """ 
    Container Class for K medoid clustering results. Contains a single entry. A list of these containers can be
    converted to pandas for easy display.

    Parameters:
        k: the number of clusters set.
        cluster_assignments: List with cluster assignment for each observation.
        silhouette_score: float with clustering silhouette score.
        random_seed_used : int or float with the random seed used in k-medoid clustering.
    """
    k : int
    cluster_assignments : List[int]
    silhouette_score : float
    random_seed_used : Union[int, float]
    def __str__(self) -> str:
        """ Custom Print Method for kmedoid grid entry producing an easy readable string output. """
        custom_print = (
            f"k = {self.k}, silhoutte_score = {self.silhouette_score}, \n"
            f"cluster_assignment = {', '.join(self.cluster_assignments[0:7])}...")
        return custom_print

@dataclass
class Msfeast:
  """
  msFeaST pipeline api class for user interactions via python console or GUI (NOT IMPLEMENTED YET)

  msFeaST probvides a pipeline for feature set testing based on unknown spectra. This API deals with data loading,
  processing, classification, embedding, statistical testing, and constructing the suitable data structures for 
  interactive visualization in the msFeaST javascript app. The steps of the pipeline expected to be followed in order 
  are:

  1. Data importing: The required data for msFeaST is loaded into the class instance via from file loading or 
  provided as appropriate Python objects. Strict abidance by format requirements is crucial here to avoid downstream
  processing issues.
  2. Data cleaning: spectral data are filtered to conform with minimum spectral data requirements.
  3. Optional Data Normalization: total-sum scaling is applied to the quantification table if no alternative 
  normalization approaches were used in other software.
  4. Similarity measure computation
  5. k-medoid classification or class provision
  6. embedding or embedding provision (t-SNE and Umap implemented)
  7. global test run or provide statistical results in some suitable format (not defined for now) # TODO: define stats provision format
  8. Export data tables & dashboard data structures

  The msFeaST pipeline requires an interaction with the R programming language to run globaltest. For this to work, 
  a working R and globaltest installation, as well as the R R script itself need to be available.
  """
  # msFeaST instance variable are default set to None in constructor. The pipeline gradually builds them up.
  quantification_table: pd.DataFrame | None = None
  treatment_table: pd.DataFrame | None = None
  spectra_matchms: list[matchms.Spectrum] | None = None
  similarity_array : Union[None, np.ndarray] = None

  # settings used dictionary, initialized as empty
  _settings_used : dict = field(default_factory= lambda: {})

  def attach_spectral_data_from_file(self, filepath : str, identifier_key : str = "feature_id") -> None:
    """ 
    Loads and attaches spectra from provided filepath (pointing to compatible .mgf file). Does not run any pre-
    processing. While the function does not check spectral data integrity or performs any filtering, it does make 
    sure that unique feature identifiers are available for all spectra provided.

    Parameters
    filepath : str pointing to a .mgf or .MGF formatted file containing the spectral data. 
    identifier_key : str defaults to feature_id. Must be a valid key in spectral metadata pointing to unique feature_id.
      Note that identifiers should always be supplied in all lowercase letters; they will not be recognized if provided
      in uppercase even if the original entry is uppercase. This is because the matchms.Spectrum internal representation 
      makes use of lowercase only.
    
    Returns
    Attaches spectrum_matchms to pipeline instance. Returns None.
    """
    _assert_filepath_exists(filepath)
    spectra_matchms = _load_spectral_data(filepath, identifier_key)
    if identifier_key != "feature_id":
      spectra_matchms = _add_feature_id_key(spectra_matchms, identifier_key)
    _validate_spectra(spectra_matchms)
    _ = _extract_feature_ids_from_spectra(spectra_matchms) # loads feature_ids to check uniqueness of every entry
    self.spectra_matchms = spectra_matchms
    self._spectral_data_loading_complete = True
    return None

  def load_quantification_table_from_file(self, filepath : str) -> None :
    """
    NOT IMPLEMENTED
    Loads spectral data quantification table from csv / tsv file. Must contain sample id, and feature_id based columns.
    """
    # Implement file extension check to see whether a csv or tsv file is provided
    table = ...
    self._validate_quantification_table(table)
    return None
  
  def load_treatment_table_from_file(self, filepath : str) -> None:
    """
    NOT IMPLEMENTED
    Loads treatment data table from csv / tsv file. Must contain sample_id and treatment_id columns.
    """
    # Implement file extension check to see whether a csv or tsv file is provided
    table = ...
    self._validate_treatment_data(table)
    return None
  
  def attach_spectral_data(
      self, 
      spectra : List[matchms.Spectrum], 
      identifier_key : str = "feature_id"
      ) -> None:
    """
    NOT IMPLEMENTED

    Attaches spectral data from python list. Must have feature_id entry, or alternative identifier_key name to fetch. 

    Parameters
      spectra : Python object of type List[matchms.Spectrum] containing the spectral data. 
      identifier_key : str defaults to feature_id. Must be a valid key in spectral metadata pointing to unique feature_id.
      Note that identifiers should always be supplied in all lowercase letters; they will not be recognized if provided
      in uppercase even if the original entry is uppercase. This is because the matchms.Spectrum internal representation 
      makes use of lowercase only.
    Returns 

    """
    self._validate_spectra()
    return None
  
  def attach_quantification_table(self, table : pd.DataFrame) -> None :
    """
    Attaches spectral data quantification table from pandas data frame. Must contain sample id, and feature_id based 
    columns.
    """
    assert self.spectra_matchms is not None, (
      "Error: quantificaiton table validation requires spectral data. Please provide spectral data before attaching "
      "quantification table.")

    self._validate_quantification_table(table)
    self.quantification_table = table
    return None
  
  def attach_treatment_table(self, table : pd.DataFrame) -> None:
    """
    Attaches treatment data table in form of pandas data frame. Must contain sample_id and treatment_id columns.

    Parameters
      table: pd.DataFrame with sample_id and treatment_id columns. Exact name correspondance is expected. Assumes a 
        dtype of str.
    Returns
      None. Atacched treatment_table to self.
    """
    # Implement file extension check to see whether a csv or tsv file is provided
    self._validate_treatment_data(table) # validat
    self.treatment_table = table
    return None
  
  def _validate_quantification_table(self, table : pd.DataFrame) -> None:
    """
    NOT IMPLEMENTED. 
    Function validates quantification table input against spectral data.
    """
    # Use self.spectra_matchms to get feature_ids to compare against quantification table

    # Make sure there are no NA
    # Make sure there is numeric information
    # Make sure feature_id columns are available
    # Make sure sample_id column is available
    # Make sure no non-feature_id or non-sample_id columns are there to confuse downstream functions. 
    assert True
    return None
  
  def _validate_treatment_data(self, table : pd.DataFrame):
    """
    NOT IMPLEMENTED
    Function validates treatment information input. Expects treatment information to be a pandas.DataFrame with
    sample_id and treatment_id columns. In addition, sample_ids should be identical to those used in quantification
    table (no missing, no type differences) and treatment_ids should each contain multiple instances (statistical 
    units.)
    """
    # Make sure treatment infor is pandas Data Frame, has sample_id and treatment_id columns
    # Make sure sample_id corresponds to sample_ids in quantification table, 
    # Make sure each sample has a corresponding treatment
    # Make srue treatment_id are type string
    # Make sure there are three instances of each treatment id at least, otherwise warn of possible issues in stats
    assert True
    return None
  
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
  
  def attach_spectral_similarity_array(
      self, 
      similarity_array : np.ndarray, 
      score_name : str = "unspecified"
      ) -> None:
    """
    Function attaches pre-computed similarity matrix in format of square np.ndarray to pipeline. Requires spectra to be
    loaded for size agreement assessment.

    Parameters
      similarity_array : np.ndarray square form with shape n_spectra by n_spectra. Similarities in range 0 to 1. 
    Returns
      None. Attaches similarity_array to self.
    """
    assert self.spectra_matchms is not None, (
      "Error: pipeline requires spectra to be attached prior to similarity computation."
    )
    _assert_similarity_matrix(similarity_array, n_spectra=len(self.spectra_matchms))
    self.similarity_array = similarity_array
    self.similarity_score_name = score_name
    return None
  
  def run_spectral_similarity_computations(
      self, 
      method = "modified_cosine_score", 
      model_directory = "not_available"
      ) -> None:
    """ 
    NOT IMPLEMENTED
    DELEGATOR FUNCTION 
    --> refers to specific functions for computing the different measures
    --> specifies similarity matrix for use in the remainder of the tool.
    """
    # check that: spectra are available, check method and whether model available if required
    if method == "ms2deepscore" and model_directory == "not_available":
      # print warning message and indicate object unchanged
      ...
    if method == "spec2vec" and model_directory == "not_available":
      # print warning message and indicate object unchanged
      ...
    if method == "modified_cosine_score":
      # check input validity and run data
      values = _compute_similarities_cosine(self.spectra, cosine_type="ModifiedCosine")
      # if successful, upade object and attach similarities
      self.similarity_array = values
    self.similarity_score = method # record used similarity matrix approach 
    return None
  
  def get_spectral_similarity_array(self) -> Union[np.ndarray, None]:
    """
    Returns copy of the spectral similarity array from pipeline.
    """
    if self.similarity_array is not None:
      return copy.deepcopy(self.similarity_array)
    else: 
      warn("Similarity array not available. Returning None instead.")
      return None
  
  def run_kmedoid_grid(self, values_of_k = None):
    """
    NOT IMPLEMENTED
    """
    if not self._similarityMatrixAvailable:
      ...
      # warning, please attach similarity measures first
      return None
    if values_of_k == None:
      ...
      # determine number of features, set values_of_k to list of every value between 2 and min( n_features, 100 )
    # turn similarities to distance
    # run grid and attach results
    self._kmedoidGridComputed = True
    # print results as summary
    self.printKmedoidGridSummary()
    return None
  
  def _print_kmedoid_grid_results(self):
    """
    NOT IMPLEMENTED
    """
    # show the plot or textual summaries for the embedding grid
    return None

  def run_embedding_grid(self, method, method_settings):
    """
    NOT IMPLEMENTED
    """
    # run the tsne or umap grid
    # use a switch logic here to run either tsne or umap (replace separate if statements below)

    # turn similarities to distance for embedding

    if method == "tsne":
      self.method = "tsne"
      self.grid = self._run_tsne_grid(method_settings)
    if method == "umap":
      self.method = "umap"
      self.grid = self._run_umap_grid(method_settings)
    # method_settings are default encoded for either tsne or umap (may be different in naming, and number)
    # Finally, print the results
    self.printEmbeddingGridSummary()
    return None

  def _run_tsne_grid(self, inputs, settings):
    """
    NOT IMPLEMENTED
    """
    # run grid for t-SNE appropriate settings
    grid = ...
    return grid
  
  def _run_umap_grid(self, inputs, settings):
    """
    NOT IMPLEMENTED
    """
    # run grid for umap appropriate settings
    grid = ...
    return grid

  def printEmbeddingGridSummary(self, showplot = True, returnPlot = True):
    # show the plot or textual summaries for the embedding grid
    # the grid index needs to be clearly visible

    # May be method specific, assess self.method
    # plot the grid vs the values, include both grid parameters and iloc clearly
    
    # return the plot (for separate storage)
    # print the plot to visible
    plot = ...
    return plot
  
  def selectEmbeddingIndex(self):
    # make sure that the index is available in the visual overviews
    return None
  
  def run_r_testing_routine (self, filepath, overwrite = False):
    """
    NOT IMPLEMENTED
    Function writes r inputs to file, writes r script to file, tests r availabiltiy, runs test, and imports r results.
    """
    # assess all data required available
    # construct python structures suitable for file writing
    self._generate_r_script()
    self._export_r_input_data()
    self._try_r_connection()
    self._run_r_routine()
    self._import_r_results()
    # determine file_names; (date, time, run/file_prefix, default file suffix)
    # write files to directory
    # run R interface call & wait for run execution
    # --> if just waiting for bash feedback works, great, otherwise
    # --> wait until R output files appear in directory
    # load r data and attach to msfeast session data
    self.importGlobaltest(filepath)
    _statisticsDataComputed = True
    return None
  
  def _run_r_routine(self, file_directory, filepath, time_limit : int = 60):
    """
    NOT IMPLEMENTED
    Function calls the r script encoding the testing routine.
    The r-script will generate a json-file output at filepath. This function only returns when the r routine is complete
    or a time limit is exceeded. The default time limit is 60 seconds.
    """
    return None
  
  def _export_r_input_data(self, file_directory):
    """
    NOT IMPLEMENTED
    """
    return None
  
  def _try_r_connection(self, file_directory):
    """
    NOT IMPLEMENTED
    """
    return None
  
  def _generate_r_script(self, directory_path, filename):
    """
    NOT IMPLEMENTED
    Funciton creates the r script file required to run R from within python. 
    R code is a constant from the python module and standalone code.
    """
    return None
  
  def _import_r_results(self, directory_path : str):
    """
    NOT IMPLEMENTED
    Funciton imports the results of the R-based testing.
    """
    # read the r data
    # add default nodeSize conversions to the expected R outputs, deal with NA values by giving sensible defaults
    # if any NAs detected, print corresponding warning that this is unexpected
    return None
  
  def generate_json_dict(self) -> Dict: 
    """
    NOT IMPLEMENTED
    Function constructs the json representation required by the visualization app as a python dictionary,.
    """
    # this will involve create node list, 
    # edge list (including ordering and zero removal), 
    # stats data incorporation
    # Multiple steps creation various lists and or dicts of dicts
    return {}
  
  def export_to_json_file(self, filepath = None, force = False):
    """ 
    NOT IMPLEMENTED
    Can be split into many small routines, one to make the node lists, one to make the group stats values etc.
    exportToJson 
    """
    # validate the that all self object data available
    self.validate_complete()
    # validate the filepath does not exist or force setting to ensure everything works 
    assert True
    # construct json string for entire dataset
    output_dict = self.generate_json_dict()
    # write to file
    return None
  
  def validate_complete(self):
    """
    NOT IMPLEMENTED
    Runs a comprehensive check for lack of compliance or agreement between available data types, makes sure all data for
    export is available.
    """
    isValid = True
    isValid = False
    return isValid

########################################################################################################################
# Utility functions required by msFeaST that are be purely functions

def _add_feature_id_key(spectra : List[matchms.Spectrum], identifier_key : str):
  """ Function add feature_id key to all spectra in list using entry from identifier_key. """
  for spectrum in spectra:
    assert spectrum.get(identifier_key) is not None, "Error provided identifier key does not point to valid id!"
    spectrum.set(key = "feature_id", value = spectrum.get(identifier_key))
  return spectra

def _assert_feature_ids_valid(feature_ids : List[str]) -> None:
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

def _extract_feature_ids_from_spectra(spectra : List[matchms.Spectrum]) -> List[str]:
  """ Extract feature ids from list of matchms spectra in string format. """
  # Extract feature ids from matchms spectra. 
  feature_ids = [str(spec.get("feature_id")) for spec in spectra]
  _assert_feature_ids_valid(feature_ids)
  return feature_ids

def _validate_spectra(
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

def _assert_filepath_valid(filepath : str) -> None:
  """ 
  Helper Function checks whether the provided filepath is valid (str, in existing folder etc.), the file doesn't need to 
  exist. The function raises an assert error if not.
  """
  assert isinstance(filepath, str), f"Error: expected filepath to be string but received {type(filepath)}"
  assert os.path.isfile(filepath), "Error: supplied filepath is not valid."
  return None

def _assert_filepath_exists(filepath : str) -> None:
  """ 
  Helper Function checks whether the provided filepath is valid and exists. The function raises an assert error if not. 
  """
  assert isinstance(filepath, str), f"Error: expected filepath to be string but received {type(filepath)}"
  assert os.path.exists(filepath), "Error: supplied filepath does not point to existing file."
  return None

def _load_spectral_data(filepath : str, identifier_key : str = "feature_id") -> List[matchms.Spectrum]:
  """ Loads spectra from file and validates identifier availability """
  _assert_filepath_exists(filepath)
  spectra_matchms = list(matchms.importing.load_from_mgf(filepath)) # this may cause its own assert errors. 
  assert isinstance(spectra_matchms, list), "Error: spectral input must be type list[matchms.Spectrum]"
  for spec in spectra_matchms:
    assert isinstance(spec, matchms.Spectrum), "Error: all entries in list must be type spectrum."
  return spectra_matchms

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
    filename = _return_model_filepath(model_path, ".model")
    model = gensim.models.Word2Vec.load(filename) # Load s2v model
    similarity_measure = Spec2Vec(model=model)
    scores_matchms = matchms.calculate_scores(
        spectrum_list, spectrum_list, similarity_measure, is_symmetric=True, array_type="numpy")
    scores_ndarray = scores_matchms.to_array()
    if apply_linear_transform:
      scores_ndarray = (scores_ndarray + 1) / 2 # linear scaling from range -1 to 1 to 0 to 1
    scores_ndarray = np.clip(scores_ndarray, a_min = 0, a_max = 1) # Clip to deal with floating point issues
    return scores_ndarray

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
    tmp = matchms.calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True, array_type = "numpy")
    scores = _extract_similarity_scores_from_matchms_cosine_array(tmp.to_array())
    scores = np.clip(scores, a_min = 0, a_max = 1) 
    return scores

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