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
# from version import __version__ # for version indication in exports
from spectral_comparison import compute_similarities_wrapper, convert_similarity_to_distance, assert_similarity_matrix
from file_checking import assert_filepath_exists
from process_spectra import validate_spectra, load_spectral_data, add_feature_id_key, extract_feature_ids_from_spectra
from embedding import GridEntryTsne, run_tsne_grid, print_tsne_grid, check_perplexities
from kmedoid_clustering import GridEntryKmedoid, run_kmedoid_grid, check_k_values, print_kmedoid_grid
from utility_functions import assert_iloc_valid
from run_r_script import run_statistics_routine
from integrate import construct_node_list, construct_edge_list, apply_bonferroni_correction_to_group_stats

@dataclass
class Msfeast:
  """
  msFeaST pipeline api class for user interactions via python console or GUI (NOT IMPLEMENTED)

  msFeaST probvides a pipeline for feature set testing based on unknown spectra. This API deals with data loading,
  processing, classification, embedding, statistical testing, and constructing the suitable data structures for 
  interactive visualization in the msFeaST javascript app. The steps of the pipeline expected to be followed in order 
  are:

  1. Data importing: The required data for msFeaST is loaded into the class instance via from file loading or 
  provided as appropriate Python objects.
  2. [Optional] Data cleaning: spectral data are filtered to conform with minimum spectral data requirements.
  3. [Optional] Data Normalization: total-sum scaling is applied to the quantification table if no alternative 
  normalization approaches were used in other software.
  4. Similarity measure computation. 
  5. K-medoid clustering.
  6. Embedding.
  7. Statistical Testing at cluster level and computation of descriptive statistics on feature level.
  8. Export dashboard data structure (json)

  The msFeaST pipeline requires an interaction with the R programming language to run globaltest. For this to work, 
  a working R and globaltest installation need to be available (auto-install along python NOT IMPLEMENTED)

  """
  # msFeaST instance variable are default set to None in constructor. The pipeline gradually builds them up.
  quantification_table: pd.DataFrame | None = None
  treatment_table: pd.DataFrame | None = None
  spectra_matchms: list[matchms.Spectrum] | None = None
  similarity_array : Union[np.ndarray, None] = None
  embedding_coordinates_table : Union[pd.DataFrame, None] = None
  kmedoid_grid : Union[List[GridEntryKmedoid], None] = None
  assignment_table : Union[pd.DataFrame, None] = None
  r_json_results: Union[dict, None] = None
  dashboard_json_dict : Union[dict, None] = None
  r_data_long_df : Union[pd.DataFrame, None] = None

  # settings used dictionary, initialized as empty
  _settings_used : dict = field(default_factory= lambda: {}) # "msfeast_version": __version__["__version__"]

  def attach_spectra_from_file(self, filepath : str, identifier_key : str = "feature_id") -> None:
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
      spectra_matchms = add_feature_id_key(spectra_matchms, identifier_key)
    
    validate_spectra(spectra_matchms, identifier_key)
    extract_feature_ids_from_spectra(spectra_matchms) # loads feature_ids & checks uniqueness of every entry
    self.spectra_matchms = spectra_matchms
    self._spectral_data_loading_complete = True
    return None

  def attach_spectra_from_list(self, spectra : List[matchms.Spectrum], identifier_key : str = "feature_id") -> None:
    """
    Attaches spectral data from python list. Must have feature_id entry, or alternative identifier_key name to fetch. 

    Parameters
      spectra : Python object of type List[matchms.Spectrum] containing the spectral data. 
      identifier_key : str defaults to feature_id. Must be a valid key in spectral metadata pointing to unique 
      feature_id. Note that identifiers should always be supplied in all lowercase letters; they will not be recognized 
      if provided in uppercase even if the original entry is uppercase. This is because the matchms.Spectrum internal 
      representation makes use of lowercase only.
    Returns 
    """
    spectra_matchms = copy.deepcopy(spectra)
    if identifier_key != "feature_id":
      spectra_matchms = add_feature_id_key(spectra_matchms, identifier_key)
    validate_spectra(spectra_matchms)
    self.spectra_matchms = spectra_matchms
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
  
  def attach_spectral_similarity_array(self, similarity_array : np.ndarray, score_name : str = "unspecified") -> None:
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
    self.attach_settings_used(score_name = score_name)
    return None
  
  def run_and_attach_spectral_similarity_computations(
    self, 
    score_name : str = "ModifiedCosine",
    model_directory: Union[str, None] = None, 
    force : bool = False) -> None:
    """ 
    Runs and attaches spectral similarity measures using self.spectra. Requires model_directory_path as input.
    
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
    self.attach_settings_used(score_name = score_name)
    return None

  def return_spectral_similarity_array(self) -> Union[np.ndarray, None]:
    """
    Returns a deep copy of the spectral similarity array from pipeline.
    """
    if self.similarity_array is not None:
      return copy.deepcopy(self.similarity_array)
    else: 
      warn("Similarity array not available. Returning None instead.")
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
    check_k_values(k_values, len(self.spectra_matchms))
    distance_matrix = convert_similarity_to_distance(self.similarity_array)
    self.kmedoid_grid = run_kmedoid_grid(distance_matrix, k_values)
    print_kmedoid_grid(self.kmedoid_grid)
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
    feature_ids = extract_feature_ids_from_spectra(self.spectra_matchms)
    self.assignment_table = pd.DataFrame(data = {
        "feature_id" : feature_ids,  
        "set_id" : [f"group_{clust}" for clust in self.kmedoid_grid[iloc].cluster_assignments]
      }
    )
    selected_k = self.kmedoid_grid[iloc].k
    self.attach_settings_used(kmedoid_n_clusters = selected_k)
    return None
  
  def run_and_attach_statistical_comparisons(
      self, 
      directory: str, 
      r_filename: str = "msfeast_r_output.json", 
    ) -> None:
    """ 
    Function runs statistical contrast comparisons at group level and at feature level. 
    
    Function refers statistical computations to msfeast R bash script requiring file system exchange.
    """
    r_json_results = run_statistics_routine(
      directory, 
      self.quantification_table, 
      self.treatment_table, 
      self.assignment_table,
      r_filename, 
      True
    )
    self.r_json_results = r_json_results
    return None

  def integrate_and_attach_dashboard_data(self, top_k_max : int = 50, alpha : float = 0.01) -> None: 
    """ Integrate pipeline intermediate results into dashboard compatible dictionary for json exporting. """
    feature_ids = extract_feature_ids_from_spectra(self.spectra_matchms)
    
    # Create dashboard json dict components
    node_list = construct_node_list(self.r_json_results, self.assignment_table, self.embedding_coordinates_table)
    edge_list = construct_edge_list(self.similarity_array, feature_ids, top_k_max)
    group_stats_list = apply_bonferroni_correction_to_group_stats(self.r_json_results["set_specific"], alpha)
    group_keys = self.r_json_results["set_id_keys"]
    univariate_measure_keys = ["log2FoldChange", "globalTestFeaturePValue"]
    group_measure_keys = ["globalTestPValue"]
    contrast_keys = self.r_json_results["contrast_keys"]

    # Create dashboard json dictionary from components.
    dashboard_json_dict = {
      "groupKeys": group_keys,
      "univMeasureKeys": univariate_measure_keys,
      "groupMeasureKeys": group_measure_keys,
      "contrastKeys": contrast_keys,
      "groupStats": group_stats_list,
      "nodes": node_list,
      "edges": edge_list,
    }
    self.dashboard_json_dict = dashboard_json_dict
    return None

  def export_dashboard_json(self, filepath : str):
    """ Function exports dashboard json to file at filepath. """
    from integrate import write_dict_to_json_file
    write_dict_to_json_file(self.dashboard_json_dict, filepath)
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
    check_perplexities(perplexity_values, len(self.spectra_matchms))
    distance_matrix = convert_similarity_to_distance(self.similarity_array)
    self.tsne_grid = run_tsne_grid(distance_matrix, perplexity_values)
    print_tsne_grid(self.tsne_grid)
    return None

  def select_tsne_settings(self, iloc : int) -> None:
    """ Select particular t-SNE coordinate setting using entry iloc. 
    
    Parameters:
      iloc : int pointing towards entry in tsne grid to select for coordinates extraction.
    Returns:
      Attaches t-sne coordinates table to self. Returns None.
    """
    # Check grid exists
    assert self.tsne_grid is not None, (
        "Error: tsne_grid is None. Please run 'run_and_attach_tsne_grid' before selecting a value."
    )
    assert_iloc_valid(iloc, len(self.tsne_grid))
    embedding_coordinates_table = pd.DataFrame({"feature_id" : extract_feature_ids_from_spectra(self.spectra_matchms)})
    embedding_coordinates_table["x"] = self.tsne_grid[iloc].x_coordinates
    embedding_coordinates_table["y"] = self.tsne_grid[iloc].y_coordinates
    self.embedding_coordinates_table = embedding_coordinates_table
    self.attach_settings_used(tsne_perplexity = self.tsne_grid[iloc].perplexity)
    return None
  
  def attach_settings_used(self, **kwargs) -> None:
    """Helper function attaches used settings to settings dictionary via key value pairs passed as kwargs. """
    for key, value in kwargs.items():
        if key is not None and value is not None:
            self._settings_used[key] = value
    return None
  
