from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matchms
import os
import subprocess
from typing import List, TypedDict, Tuple, Dict, NamedTuple, Union
from warnings import warn
import copy # for safer get methods  pipeline internal variables
import json

# kmedoid dependency
from kmedoids import KMedoids
from sklearn.metrics import silhouette_score

from grid_entry_classes import GridEntryTsne, GridEntryKmedoid
from r_output_parsing import load_and_validate_r_output
from spectral_comparison import compute_similarities_wrapper, convert_similarity_to_distance, assert_similarity_matrix
from file_checking import assert_filepath_exists
from process_spectra import validate_spectra, load_spectral_data

# tsne dependencies
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

# plotting functionalities
import plotly
import plotly.express
import plotly.graph_objects as go


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
  7. global test run or provide statistical results in some suitable format (not defined for now)
  8. Export data tables & dashboard data structures

  The msFeaST pipeline requires an interaction with the R programming language to run globaltest. For this to work, 
  a working R and globaltest installation, as well as the R R script itself need to be available.
  """
  # msFeaST instance variable are default set to None in constructor. The pipeline gradually builds them up.
  quantification_table: pd.DataFrame | None = None
  treatment_table: pd.DataFrame | None = None
  spectra_matchms: list[matchms.Spectrum] | None = None
  similarity_array : Union[None, np.ndarray] = None
  embedding_coordinates_table : Union[None, pd.DataFrame] = None
  kmedoid_grid : Union[List[GridEntryKmedoid], None] = None
  assignment_table : Union[None, pd.DataFrame] = None
  output_dictionary : Union[None, dict] = None
  r_data_long_df : Union[None, dict] = None
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
    assert_filepath_exists(filepath)
    spectra_matchms = load_spectral_data(filepath, identifier_key)
    
    if identifier_key != "feature_id":
      spectra_matchms = _add_feature_id_key(spectra_matchms, identifier_key)
    
    validate_spectra(spectra_matchms)
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
    self.validate_quantification_table(table)
    return None
  
  def load_treatment_table_from_file(self, filepath : str) -> None:
    """
    NOT IMPLEMENTED
    Loads treatment data table from csv / tsv file. Must contain sample_id and treatment_id columns.
    """
    # Implement file extension check to see whether a csv or tsv file is provided
    table = ...
    self.validate_treatment_data(table)
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
      identifier_key : str defaults to feature_id. Must be a valid key in spectral metadata pointing to unique 
      feature_id. Note that identifiers should always be supplied in all lowercase letters; they will not be recognized 
      if provided in uppercase even if the original entry is uppercase. This is because the matchms.Spectrum internal 
      representation makes use of lowercase only.
    Returns 

    """
    self.validate_spectra()
    return None
  
  def attach_quantification_table(self, table : pd.DataFrame) -> None :
    """
    Attaches spectral data quantification table from pandas data frame. Must contain sample id, and feature_id based 
    columns.
    """
    assert self.spectra_matchms is not None, (
      "Error: quantificaiton table validation requires spectral data. Please provide spectral data before attaching "
      "quantification table.")

    self.validate_quantification_table(table)
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
    self.validate_treatment_data(table) # validat
    self.treatment_table = table
    return None
  
  def validate_quantification_table(self, table : pd.DataFrame) -> None:
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
  
  def validate_treatment_data(self, table : pd.DataFrame):
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
    assert_similarity_matrix(similarity_array, n_spectra=len(self.spectra_matchms))
    self.similarity_array = similarity_array
    self._attach_settings_used(score_name = score_name)
    return None
  
  
  def run_spectral_similarity_computations(
      self, 
      score_name : str = "ModifiedCosine",
      model_directory: Union[str, None] = None, 
      force : bool = False) -> None:
    """ Runs and attaches spectral similarity measures using self.spectra. Requires model_directory_path as input.
    
    Parameters
        score_name : str indicating the score_name to use, can be "ModifiedCosine", "spec2vec", "ms2deepscore", 
          "CosineHungarian", or "CosineGreedy".
        model_directory_path : str path to directory containing model files. Assumes one set of model files only!
        force : bool defaulting to false indicating whether existing similarity matrices can be overwritten or not.
    Returns
        Attaches similarity matrices to self. Returns None.
    """
    if force is False:
      assert (self.similarity_array is None), (
        "Error: Similarities were already computed or set. "
        "To replace existing scores set Force to True or re-initialize the pipeline."
      )
    similarity_array = compute_similarities_wrapper(self.spectra_matchms, score_name, model_directory)




    self.similarity_array = similarity_array
    self._attach_settings_used(score_name = score_name)
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
  
  def run_r_testing_routine(self, directory : str, r_filename : str = "r_output.json", top_k = 20, overwrite = False):
    """
    INCOMPLETE
    Function writes r inputs to file, writes r script to file, tests r availabiltiy, runs test, and imports r results.
    directory: folder name for r output
    r_filename: filename for r output

    """
    # assess all data required available
    # construct python structures suitable for file writing
    
    self.assignment_table # -->
    self.quantification_table # --> 
    self.treatment_table # --> turn into contrasts
    
    # json filenames input for R; the one argument to pass to R so it can find all the rest
    # could also contain all info, but that makes r parsing more difficult than reading csvs...

    filepath_assignment_table = str(os.path.join(directory, "assignment_table.csv"))
    filepath_quantification_table = str(os.path.join(directory, "test_quant_table.csv"))
    filepath_treatment_table = str(os.path.join(directory, "test_treat_table.csv"))
    filepath_r_output_json = str(os.path.join(directory, r_filename))

    # TODO check for directory exist, if not, create it
    # TODO check for existing r routine output, remove if force = true

    # Write R input data to file
    self.quantification_table = self.quantification_table.reset_index(drop=True)
    self.treatment_table = self.treatment_table.reset_index(drop=True)
    self.assignment_table = self.assignment_table.reset_index(drop=True)
    
    # Required to remove the unnamed = 0 index column that is created somewhere
    self.quantification_table.drop(
      self.quantification_table.columns[
          self.quantification_table.columns.str.contains('Unnamed', case=False)], 
      axis=1, inplace=True
    )
    self.treatment_table.drop(
      self.treatment_table.columns[
        self.treatment_table.columns.str.contains('Unnamed', case=False)], 
      axis=1, inplace=True
    )
    self.assignment_table.drop(
      self.assignment_table.columns[
        self.assignment_table.columns.str.contains('Unnamed', case=False)], 
      axis=1, inplace=True
    )

    # Setting index to false is often not enough for pandas to remove it as the index is sometimes added as an unnamed 
    # column
    self.quantification_table.to_csv(filepath_quantification_table, index = False)
    self.treatment_table.to_csv(filepath_treatment_table, index = False)
    self.assignment_table.to_csv(filepath_assignment_table, index = False)

    # Fetch r script filepath
    r_script_path = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), # module directory after pip install
      "runStats.R" # filename included as package data
    )
    
    # Run R Code
    subprocess.run((
        f"Rscript {r_script_path} {filepath_quantification_table} " 
        f"{filepath_treatment_table} " 
        f"{filepath_assignment_table} "
        f"{filepath_r_output_json}"
      ), 
      shell=True
    )
    # load r data
    r_json_data = load_and_validate_r_output(filepath_r_output_json)
    # construct derived variables and attach
    self._generate_and_attach_long_format_r_data(r_json_data)
    self._generate_and_attach_json_dict(r_json_data, top_k)
    return None
  
  def _generate_and_attach_json_dict(self, r_json_data, top_k, alpha = 0.01) -> None: 
    """
    NOT IMPLEMENTED
    Function constructs the json representation required by the visualization app as a python dictionary,.
    """
    # this will involve create node list, 
    # edge list (including ordering and zero removal), 
    # stats data incorporation
    # Multiple steps creation various lists and or dicts of dicts
    nodes_list = _construct_nodes(r_json_data, self.assignment_table, self.embedding_coordinates_table)
    edge_list = _construct_edge_list(
      self.similarity_array, _extract_feature_ids_from_spectra(self.spectra_matchms), top_k
    )
    group_stats_list = _apply_bonferroni_correction_to_group_stats(r_json_data["set_specific"], alpha)
    output_dictionary = {
      "groupKeys": r_json_data["set_id_keys"],
      "univMeasureKeys": ["log2FoldChange", "globalTestFeaturePValue"],
      "groupMeasureKeys": ["globalTestPValue"],
      "contrastKeys": r_json_data["contrast_keys"],
      "groupStats": group_stats_list,
      "nodes": nodes_list,
      "edges": edge_list,
    }
    self.output_dictionary = output_dictionary
    return None

  def run_and_attach_kmedoid_grid(self, k_values : List[int] = [8, 10, 20, 30, 50]):
    """ 
    Run the k-medoid grid & attach the results to pipeline instance.
    
    Parameters
      k_values : List[int] of number of clusters to optimize for.
    Returns
      Attached kmedoid grid to self. Returns None.
    """
    # Subset k_values
    k_values = [value for value in k_values if value < len(self.spectra_matchms)]
    _check_k_values(k_values, len(self.spectra_matchms))
    distance_matrix = convert_similarity_to_distance(self.similarity_array)
    self.kmedoid_grid = _run_kmedoid_grid(distance_matrix, k_values)
    _print_kmedoid_grid(self.kmedoid_grid)
    return None
    
  def select_kmedoid_settings(self, iloc : int):
    """ 
    Select and attach particular k-medoid clustering assignments using entry iloc. Attaches assignment_table.

    Parameters:
        ilocs : int with kmedoid assignment entry to exctract from the tuning grid.
    Returns:
        Attaches cluster assignment table to self. Returns None.          
    """
    # asser provided iloc is valid
    assert isinstance(iloc, int), (
        f"Unsupported input type, iloc must be type int but {type(iloc)} provided!"
    )
    valid_ilocs = set([iloc for iloc in range(0, len(self.kmedoid_grid))])
    assert iloc in valid_ilocs, (
      "Error: iloc provided not in range of valid ilocs for kmedoid grid! Values must be in set: "
      f"{valid_ilocs}"
    )
    # Make sure an initiated classification_table is available
    feature_ids = _extract_feature_ids_from_spectra(self.spectra_matchms)
    self.assignment_table = pd.DataFrame(data = {
        "feature_id" : feature_ids,  
        "set_id" : [f"group_{clust}" for clust in self.kmedoid_grid[iloc].cluster_assignments]
      }
    )
    selected_k = self.kmedoid_grid[iloc].k
    self._attach_settings_used(kmedoid_n_clusters = selected_k)
    return None

  def _attach_settings_used(self, **kwargs) -> None:
    """Helper function attaches used settings to settings dictionary via key value pairs passed as kwargs. """
    for key, value in kwargs.items():
        if key is not None and value is not None:
            self._settings_used[key] = value
    return None
  
  def export_to_json_file(self, filepath : str, force = False):
    """ 
    INCOMPLETE
    Can be split into many small routines, one to make the node lists, one to make the group stats values etc.
    exportToJson 
    """
    # validate the that all self object data available
    self.validate_complete()
    # validate the filepath does not exist or force setting to ensure everything works 
    assert True
    
    # construct json string for entire dataset
    json_string = json.dumps(self.output_dictionary, indent=2)
    with open(filepath, 'w') as f:
        f.write(json_string)
    return None

  def validate_complete(self):
    """
    NOT IMPLEMENTED
    Runs a comprehensive check for lack of compliance or agreement between available data types, makes sure all data for
    export is available.
    """
    isValid = True
    return isValid

  def _generate_and_attach_long_format_r_data(self, r_json_data : dict) -> None:
    """ Converts json format data to long format data frame. Focuses on feature_specific and set_specific statistical
    data. The data frame will have columns type, id, contrast, measure, and value:
    --> type (feature or set, string indicating the type of entry)
    --> id (feature_id or set_id, string)
    --> contrast (contrast key, string)
    --> measure (measure key, from feature specific and set specific measures, string)
    --> value (number or string with the appropriate data for the measure)
    Function assumes all data to be available and correct. A validator should be run before, e.g 
    load_and_validate_r_output()
    """
    entries = list()
    for feature_key in r_json_data["feature_specific"].keys():
      for contrast_key in r_json_data["feature_specific"][feature_key].keys():
        for measure_key in r_json_data["feature_specific"][feature_key][contrast_key].keys():
          entries.append(
            {
              "type" : "feature",
              "id" : feature_key,
              "contrast" : contrast_key,
              "measure" : measure_key,
              "value" : r_json_data["feature_specific"][feature_key][contrast_key][measure_key]
            }
          )
    for feature_key in r_json_data["set_specific"].keys():
      for contrast_key in r_json_data["set_specific"][feature_key].keys():
        for measure_key in r_json_data["set_specific"][feature_key][contrast_key].keys():
          entries.append(
            {
              "type" : "set",
              "id" : feature_key,
              "contrast" : contrast_key,
              "measure" : measure_key,
              "value" : r_json_data["set_specific"][feature_key][contrast_key][measure_key]
            }
          )
    long_form_df = pd.DataFrame.from_records(entries)
    self.r_data_long_df = long_form_df
    return None

  def run_and_attach_tsne_grid(self, perplexity_values : List[int] = [10, 20, 30, 40, 50]) -> None:
    """ Run the t-SNE grid & attach the results to pipeline instance.

    Parameters:
      perplexity_values : List[int] with the preplexity values to run tsne optimization with.
    Returns:
      Attaches tsne optimization results to self. Returns None.
    """
    # Subset perplexity values
    perplexity_values = [perplexity for perplexity in perplexity_values if perplexity < len(self.spectra_matchms)]
    _check_perplexities(perplexity_values, len(self.spectra_matchms))
    distance_matrix = convert_similarity_to_distance(self.similarity_array)
    self.tsne_grid = _run_tsne_grid(distance_matrix, perplexity_values)
    _print_tsne_grid(self.tsne_grid)
    return None

  def select_tsne_settings(self, iloc : int) -> None:
    """ Select particular t-SNE coordinate setting using entry iloc. 
    
    Parameters:
      iloc : int pointing towards entry in tsne grid to select for coordinates extraction.
    Returns:
      Attaches t-sne coordinates table to self. Returns None.
    """
    # check iloc valid
    assert self.tsne_grid is not None, (
        "Error: tsne_grid is None. Please run 'run_and_attach_tsne_grid' before selecting a value."
    )
    assert iloc in [x for x in range(0, len(self.tsne_grid))], (
        f"Error: must provide iloc in range of tsne grid 0 to {len(self.tsne_grid)}"
    )
    embedding_coordinates_table = pd.DataFrame({"feature_id" : _extract_feature_ids_from_spectra(self.spectra_matchms)})
    embedding_coordinates_table["x"] = self.tsne_grid[iloc].x_coordinates
    embedding_coordinates_table["y"] = self.tsne_grid[iloc].y_coordinates
    self.embedding_coordinates_table = embedding_coordinates_table
    self._attach_settings_used(tsne_perplexity = self.tsne_grid[iloc].perplexity)
    return None

  def plot_selected_embedding(self) -> None:
    """ Plots the selected t-sne embedding. """
    assert self.embedding_coordinates_table is not None, "Error: tsne coordinates must be selected to plot embedding."
    data = self.embedding_coordinates_table
    fig  = plotly.express.scatter(
      data_frame= data, x = "x", y = "y", hover_data=["feature_id"],
      width=800, height=800
    )
    fig.show()

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

def _run_kmedoid_grid(
    distance_matrix : np.ndarray, 
    k_values : List[int], 
    random_states : Union[List, None] = None
    ) -> List[GridEntryKmedoid]:
  """ Runs k-medoid clustering for every value in k_values. 
  
  Parameters:
      distance_matrix: An np.ndarray containing pairwise distances.
      k_values: A list of k values to try in k-medoid clustering.
      random_states: None or a list of integers specifying the random state to use for each k-medoid run.
  Returns: 
      A list of GridEntryKmedoid objects containing grid results.
  """
  if random_states is None:
      random_states = [ 0 for _ in k_values ]
  output_list = []
  _check_k_values(k_values, max_k = distance_matrix.shape[0])
  for idx, k in enumerate(k_values):
      cluster = KMedoids(
          n_clusters=k, 
          metric='precomputed', 
          random_state=random_states[idx], 
          method = "fasterpam"
      )  
      cluster_assignments = cluster.fit_predict(distance_matrix)
      cluster_assignments_strings = [
          "km_" + str(elem) 
          for elem in cluster_assignments
      ]
      score = silhouette_score(
          X = distance_matrix, 
          labels = cluster_assignments_strings, 
          metric= "precomputed"
      )
      output_list.append(
          GridEntryKmedoid(
              k, 
              cluster_assignments_strings, 
              score, 
              random_states[idx]
          )
      )
  return output_list

def _check_k_values(k_values : List[int], max_k : int) -> None:
    """ Function checks whether k values match expected configuration. Aborts if not. """
    assert k_values is not [], (
        "Error: k_values list is empty! This may be a result of post-processing: there must be a "
        "k value below the number of features/spectra for optimization to work."
    )
    assert isinstance(k_values, list), (
        "Error: k_values must be a list. If only running one value, specify input as [value]."
    )
    for k_value in k_values: 
        assert isinstance(k_value, int) and k_value < max_k, (
            "Error: k_value must be numeric (int) and smaller than number of features/spectra." 
        )
    return None

def _print_kmedoid_grid(grid : List[GridEntryKmedoid]) -> None:
  """ Prints all values in kmedoid grid in readable format via pandas conversion """
  kmedoid_results = pd.DataFrame.from_dict(data = grid).loc[
      :, ["k", "silhouette_score", "random_seed_used"]
  ]
  kmedoid_results.insert(loc = 0, column = "iloc", value = [iloc for iloc in range(0, len(grid))])
  print("Kmedoid grid results. Use to inform kmedoid classification selection ilocs.")
  print(kmedoid_results)
  return None

def _plot_kmedoid_grid(
  kmedoid_list : List[GridEntryKmedoid]
  ) -> None:
  """ Plots Silhouette Score vs k for each entry in list of GridEntryKmedoid objects. """
  scores = [x.silhouette_score for x in kmedoid_list]
  ks = [f"k = {x.k} / iloc = {iloc}" for iloc, x in enumerate(kmedoid_list)]
  fig = plotly.express.scatter(x = ks, y = scores)
  fig.update_layout(
    xaxis_title="K (Number of Clusters) / iloc", 
    yaxis_title="Silhouette Score"
  )
  fig.show()
  return None

def _check_perplexities(perplexity_values : List[Union[float, int]], max_perplexity : Union[float, int]) -> None:
  """ Function checks whether perplexity values match expected configuration. Aborts if not. """
  assert perplexity_values is not [], (
    "Error: perplexity_values list is empty! This may be a result of post-processing: there must be a "
    "perplexity value below the number of spectra for optimization to work."
  )
  assert isinstance(perplexity_values, list), (
    "Error: perplexity values must be a list. If only running one value, specify input as [value]."
  )
  for perplexity_value in perplexity_values: 
    assert isinstance(perplexity_value, (int, float)) and perplexity_value < max_perplexity, (
      "Error: perplexity values must be numeric (int, float) and smaller than number of features." 
    )
  return None

def _run_tsne_grid(
  distance_matrix : np.ndarray,
  perplexity_values : List[int], 
  random_states : Union[List, None] = None
  ) -> List[GridEntryTsne]:
  """ Runs t-SNE embedding routine for every provided perplexity value in perplexity_values list.

  Parameters:
      distance_matrix: An np.ndarray containing pairwise distances.
      perplexity_values: A list of perplexity values to try for t-SNE embedding.
      random_states: None or a list of integers specifying the random state to use for each k-medoid run.
  Returns: 
      A list of GridEntryTsne objects containing grid results. 
  """
  _check_perplexities(perplexity_values, distance_matrix.shape[0])
  if random_states is None:
      random_states = [ 0 for _ in perplexity_values ]
  output_list = []
  for idx, perplexity in enumerate(perplexity_values):
    model = TSNE(
      metric="precomputed", 
      random_state = random_states[idx], 
      init = "random", 
      perplexity = perplexity
    )
    z = model.fit_transform(distance_matrix)
    # Compute embedding quality
    dist_tsne = squareform(pdist(z, 'seuclidean'))
    spearman_score = np.array(spearmanr(distance_matrix.flat, dist_tsne.flat))[0]
    pearson_score = np.array(pearsonr(distance_matrix.flat, dist_tsne.flat))[0]
    output_list.append(
      GridEntryTsne(
        perplexity, 
        z[:,0], 
        z[:,1], 
        pearson_score, 
        spearman_score, 
        random_states[idx]
      )
    )
  return output_list

def _plot_tsne_grid(tsne_list : List[GridEntryTsne]) -> None:
  """ Plots pearson and spearman scores vs perplexity for each entry in list of GridEntryTsne objects. """
  
  pearson_scores = [x.spearman_score for x in tsne_list]
  spearman_scores = [x.pearson_score for x in tsne_list]
  iloc_perplexity = [ f"{x.perplexity} / {iloc}" for iloc, x in enumerate(tsne_list)]

  trace_spearman = go.Scatter(x = iloc_perplexity, y = spearman_scores, name="spearman_score", mode = "markers")
  trace_pearson = go.Scatter(x = iloc_perplexity, y = pearson_scores, name="pearson_score", mode = "markers")
  fig = go.Figure([trace_pearson, trace_spearman])
  fig.update_layout(xaxis_title="Perplexity / iloc", yaxis_title="Score")
  fig.show()
  return None

def _print_tsne_grid(grid : List[GridEntryTsne]) -> None:   
  """ Prints all values in tsne grid in readable format via pandas conversion """
  tsne_results = pd.DataFrame.from_dict(data = grid).loc[
      :, ["perplexity", "pearson_score", "spearman_score", "random_seed_used"]
  ]
  tsne_results.insert(loc = 0, column = "iloc", value = [iloc for iloc in range(0, len(grid))])
  print("T-sne grid results. Use to inform t-sne embedding selection.")
  print(tsne_results)
  return None
