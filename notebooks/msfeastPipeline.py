from dataclasses import dataclass
import pandas as pd
import numpy as np
import matchms
import typing
from typing import List, Union

# spec2vec dependencies
from spec2vec import Spec2Vec
import gensim

# ms2deepscore dependencies (does not appear to work for Python 3.10 ; check compatible python version)
#from ms2deepscore import MS2DeepScore
#from ms2deepscore.models import load_model

import os
# from typing import List, TypedDict, Tuple, Dict, NamedTuple, Union

# ISSUE: msFeaST does not contain any imputation routines. Data is assumed to not contain any NA values.

@dataclass
class msfeast:
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
  spectra_list: list[matchms.Spectrum] | None = None

  # Instance tracking variables
  _dataLoaded: bool = False # _dataAttached / _dataUploaded / _dataAvailable
  _contrastSelected: bool = False # sounds accurate
  _referenceCategorySelected: bool = False # sounds accurate _referenceTreatmentSelected also an option
  # _spectralDataCleaningDone_: bool = False # this one might be unnecessary. If data cleaned elsewhere, may not be necessary here. Causes no blocking
  _similarityMatrixAvailable: bool = False # sounds accurate pairwiseSimilarityMatrixAvailable would be more explicit, technically numpy array...
  _kmedoidGridComputed: bool = False # 
  _kmedoidIndexSelected: bool = False
  _tnseGridComputed: bool = False
  _tsneIndexSelected: bool = False
  _statisticsDataComputed: bool = False

  def loadAndSelectDataFromFile(self, filepath_quantification_table, filepath_metadata, filepath_spectra):
    """ NOT IMPLEMENTED YET
    Deals with loading data from filepaths & selecting / subselecting the right data for 
    attaching to msfeast class instance. 

    Currently indicated as a single function call but actually a multiple step process, including load, expose, select in some gui, process, and finally attach
    Could be mocked using a "wait for console input approach" but that seems hardly useful without the visual underpinnings (maybe Jupyter?)

    """
    # Anticipated Steps
    # TODO: check path exist
    # TODO: try loading data from expected formats & convert to suitable python representation
    # TODO: expose and let user select appropriate columns & reference categories
    # TODO: subset data to selected and overlapping sub-parts (no unused feature_ids, no unused sample_ids etc.)
    # TODO: if selected, run cleanSpectralData with minimal exposed matchms functionality
    # TODO: validate data agreement (feature_id, sample_id, treatment_id overlaps and uniqueness)
    # TODO: attach spectral data
    return None
  
  def attachData(
      self, 
      quantification_table : pd.DataFrame, 
      treatment_table : pd.DataFrame, 
      spectra : List[matchms.Spectrum]
    ) -> None:
    """
    Method attaches data to the msfeast class object.
    
    This method does not deal with loading from file, but instead assumes the data were already loaded and processed for
    compatibility with msFeast (strict, including naming convention and file-types & sub-setting).
    """
    # TODO: ADD INPUT VALIDATION FUNCTION CALL
    self.quantification_table = quantification_table
    self.treatment_table = treatment_table
    self.spectra = spectra
    self._dataLoaded = True # take note of data being available (required for follow-up steps)
    return None
  
  def _validateAttachedData(
      self, 
      quantification_table : pd.DataFrame, 
      treatment_table : pd.DataFrame, 
      spectra : List[matchms.Spectrum]
    ) -> bool:
    """ NOT IMPLEMENTED """
    # TODO: define and implement the checks needed for the input data to pass all requirements
    ...
    return False

  def cleanSpectralData(self, spectra):
    """ NOT IMPLEMENTED 
    """
    # TODO: apply spectral pre-processing inside the pipeline. Currently: assumed done outside of the pipeline, no checks in place.
    
    # After processing
    # validate cleanded data output validity and id agreement, check non empty
    # update data
    self._dataSetCleaningRun = True
    return None

  def cleanInputData(self):
    """ NOT IMPLEMENTED 
    - private method that cleans input data to only contain relevant data to avoid bugs from unexpected data floating around.
    - no unused feature_id in spectra or quantification table
    - no unused sample_id in the metadata or quantification table
    """

  def attachSimilarities(self, similarities : np.ndarray, similarity_measure_name : str):
    """ NOT IMPLEMENTED
    --> attach similarities and check for agreement in dimensions with spectra list
    --> Require: at least order alignment between spectra list and similarity matrix
    --> Possibly: 
        implement the similarities in a non-ambiguous format with all respective feature_ids explicit (sim matrix + 
        feature_id list in order of the matrix). Then, make sure they are ordered correctly.
    """
    # validate similarities to match feature_id length (implied order agreement!)
    # validate similarities to be in range 0 to 1
    # specify that similarity matrix must follow feature_id ordering provided (uncheckable!). 
    # attach the similarity matrix
    self._similarityMatrixAvailable = True
    return None
  
  def computeSimilarities(self, method = "modified_cosine_score", model_directory = "not_available", output = False) -> Union[None, np.ndarray]:
    """ 
    NOT IMPLEMENTED
    computes and attaches similarities, by default returns None, but can be made to also return the array
    DELEGATOR FUNCTION 
    --> refers to specific functions for computing the different measures
    --> specified similarity matrix for use in the remainder of the tool.
    """
    # check that: spectra are available, check method and whether model available if required
    if method == "ms2deepscore" and model_directory == "not_available":
      # print warning message and indicate object unchanged
      return False
    if method == "spec2vec" and model_directory == "not_available":
      # print warning message and indicate object unchanged
      return False
    if method == "modified_cosine_score":
      # check input validity and run data
      values = compute_similarities_cosine(self.spectra, cosine_type="ModifiedCosine")
      # if successful, upade object and attach similarities
      self._similarities = values
      self._similarityMatrixAvailable = True
    if output:
      return values
    else:
      return None
  
  def runKmedoidGrid(self, values_of_k = None):
    if not self._similarityMatrixAvailable:
      ...
      # warning, please attach similarity measures first
      return None
    if values_of_k == None:
      ...
      # determine number of features, set values_of_k to list of every value between 2 and min( n_features, 100 )
    # run grid and attach results
    self._kmedoidGridComputed = True
    # print results as summary
    self.printKmedoidGridSummary()
    return None
  
  def printKmedoidGridSummary(self):
    # show the plot or textual summaries for the embedding grid
    return None

  def runEmbeddingGrid(self, method, method_settings):
    # run the tsne or umap grid
    # use a switch logic here to run either tsne or umap (replace separate if statements below)
    if method == "tsne":
      self.method = "tsne"
      self.grid = self.runTsneGrid(method_settings)
    if method == "umap":
      self.method = "umap"
      self.grid = self.runUmapGrid(method_settings)
    # method_settings are default encoded for either tsne or umap (may be different in naming, and number)
    # Finally, print the results
    self.printEmbeddingGridSummary()
    return None

  def runTsneGrid(self, settings):
    # run grid for t-SNE appropriate settings
    grid = ...
    return grid
  
  def runUmapGrid(self, settings):
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

  def runGlobalTest (self, file_directory, file_prefix = None, overwrite = False):
    # assess all data required available
    # construct python structures suitable for file writing
    # determine file_names; (date, time, run/file_prefix, default file suffix)
    # write files to directory
    # run R interface call & wait for run execution
    # --> if just waiting for bash feedback works, great, otherwise
    # --> wait until R output files appear in directory
    # load r data and attach to msfeast session data
    self.importGlobaltest(filepath)
    _statisticsDataComputed = True
    return None
  
  def importGlobalTest (self, filepath):
    # read the r data
    # add default nodeSize conversions to the expected R outputs, deal with NA values by giving sensible defaults
    # if any NAs detected, print corresponding warning that this is unexpected
    return None
  
  def exportToJson (self, filepath = None):
    """ 
    exportToJson 
    """
    # validate the self object data
    # construct json string for entire dataset
    # this will involve create node list, edge list (including ordering and zero removal), stats data incorporation

    if filepath is not None:
      # write to file
      # print export successful to console
      ...
    else:
      jsonstring = ...
      return jsonstring
    return None
  
  def validate(self):
    # Run a comprehensive check for lack of compliance or agreement between available data types
    isValid = True
    isValid = False
    return isValid



########################################################################################################################
# Utility functions required by msFeaST that can be purely functional (no self input needed)

@staticmethod
def compute_similarities_cosine(
  spectrum_list : List[matchms.Spectrum], 
  cosine_type : str = "ModifiedCosine" ) -> np.ndarray:
  """ Function computes pairwise similarity matrix for list of spectra using specified cosine score method. 
  
  Parameters:
      spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
      cosine_type: String identifier of supported cosine metric, options: ["ModifiedCosine", "CosineHungarian", 
      "CosineGreedy"]
  Returns:
      ndarray with shape (n, n) where n is the number of spectra (pairwise similarity matrix).
  """
  valid_types = ["ModifiedCosine", "CosineHungarian", "CosineGreedy"]
  assert cosine_type in valid_types, f"Cosine type specification invalid. Use one of: {str(valid_types)}"
  if cosine_type == "ModifiedCosine":
      similarity_measure = matchms.similarity.ModifiedCosine()
  elif cosine_type == "CosineHungarian":
      similarity_measure = matchms.similarity.CosineHungarian()
  elif cosine_type == "CosineGreedy":
      similarity_measure = matchms.similarity.CosineGreedy()
  # tmp variable contains both score and number of overlapping fragments; ndarray of tuples for each pair.
  tmp = matchms.calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
  scores = extract_similarity_scores_from_matchms_cosine_array(tmp.scores)
  return scores

@staticmethod
def extract_similarity_scores_from_matchms_cosine_array(tuple_array : np.ndarray) -> np.ndarray:
    """ 
    Function extracts similarity matrix from matchms cosine scores array.
    
    The cosine score similarity output of matchms stores output in a numpy array of pair-tuples, where each tuple 
    contains (sim, n_frag_overlap). This function extracts the sim scores, and returns a numpy array corresponding to 
    pairwise similarity matrix.

    Parameters:
        tuple_array: A np.ndarray of shape n by n, where each entry is a tuple containing either the score 
        (index iloc 0) or the number of overlapping fragments (index iloc 1). 
    
    Returns: 
        A np.ndarray with shape (n, n) where n is the number of spectra deduced from the dimensions of the input
        array. Each element of the ndarray contains the pairwise similarity value.
    """
    sim_data = [ ]
    for row in tuple_array:
        for elem in row:
            sim_data.append(
               float(elem[0])
            )
    similarity_scores = np.array(sim_data).reshape(tuple_array.shape[0], tuple_array.shape[1])
    return similarity_scores

@staticmethod
def compute_similarities_s2v(spectrum_list:List[matchms.Spectrum], model_path:str) -> np.ndarray:
    """ Function computes pairwise similarity matrix for list of spectra using pretrained spec2vec model.
    
    Parameters:
        spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
        model_path: Location of spec2vec pretrained model file path (filename ending in .model or file-directory)
    Returns: 
        ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
    """
    filename = return_model_filepath(model_path, ".model")
    model = gensim.models.Word2Vec.load(filename) # Load s2v model
    similarity_measure = Spec2Vec(model=model)
    scores_matchms = matchms.calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    scores_ndarray = scores_matchms.scores
    return scores_ndarray

@staticmethod
def return_model_filepath(path : str, model_suffix:str) -> str:
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

@staticmethod
def compute_similarities_ms2ds(spectrum_list:List[matchms.Spectrum], model_path:str) -> np.ndarray:
    """ 
    PORTED FROM SPECXPLORE - DEPENDENCIES NOT MET IN PYTHON 3.10 & HENCE DEACTIVATED
    Function computes pairwise similarity matrix for list of spectra using pretrained ms2deepscore model.
    
    Parameters
        spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
        model_path: Location of ms2deepscore pretrained model file path (filename ending in .hdf5 or file-directory)
    Returns: 
        ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
    """
    #filename = return_model_filepath(model_path, ".hdf5")
    #model = load_model(filename) # Load ms2ds model
    #similarity_measure = MS2DeepScore(model)
    #scores_matchms = calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    #scores_ndarray = scores_matchms.scores
    #return scores_ndarray
    ...
    return None


