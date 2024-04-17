from math import isnan
import numpy as np
import json
import pandas as pd
import copy
from warnings import warn
from msfeast.process_spectra import extract_feature_ids_from_spectra

def construct_node_list(
    r_json_data : dict, 
    assignment_table : pd.DataFrame, 
    embedding_coordinates_table: pd.DataFrame,
    spectral_ms1_information: dict
  ) -> list:
  """
  
  spectral_ms1_information : dict with feature_id keys, each with precursor_mz key value and retention_time key value 
    pairs.
  """
  # Constructs nodes using all relevant information for nodes
  # get group id from assignment_table
  # get feature_stats from R output <--> add conversions to node size (may require global bounds information)
  # For log10 pvalues it may make sense to transform using a linear scale within range, collapsing anything above
  # a certain level. For instance, max node size could be reached at p-value of 0.0001, and min size at 0.5 already
  # this would allow tho focus the scale on the part of the measure that requires granularity:
  # 0.5 -> 0.1 -> 0.01 -> 0.001 -> 0.0001
  # get x and y coordinates from embedding_coordinates_table
  ...
  node_entries = list()
  measure_keys = ["globalTestFeaturePValue", "log2FoldChange"]

  for feature_key in r_json_data["feature_specific"].keys():
    feature_group = assignment_table.loc[assignment_table['feature_id'] == feature_key, "set_id"].values[0]
    coordinates = embedding_coordinates_table.loc[embedding_coordinates_table['feature_id'] == feature_key,]
    node_data = r_json_data["feature_specific"][feature_key]
    for contrast_key in node_data.keys(): 
      for measure_key in measure_keys:
        # Only add nodeSize conversion if among the supported keys for conversion
        value = node_data[contrast_key][measure_key]   
        size = transform_measure_to_node_size(value, measure = measure_key)
        node_data[contrast_key][measure_key] = {
          "measure": str(value),
          "nodeSize": size,
        }
    node_data["spectrum_ms_information"] = spectral_ms1_information[feature_key]
    node = {
      "id" : feature_key,
      "size": 10, # --> measure derived variable, set to 10 for now.
      "group": feature_group, # feature derived variable
      "x": coordinates["x"].values[0] * 100, # default scaling for better visual representation
      "y": coordinates["y"].values[0] * 100, # default scaling for better visual representation
      "data" : node_data
    }
    # Attach the processed node to the node_entries list
    node_entries.append(node)
  return(node_entries)

def apply_bonferroni_correction_to_group_stats(groupStats, alpha = 0.01):
  """ 
  Applies Bonferroni adjustment to p-values in groupStats
  
  The number of groups times the number of contrasts gives the number of tests performed in total. Individual
  feature pvalues are not considered here and treated as descriptives instead. 
  """
  groups =  list(groupStats.keys())
  contrasts = list(groupStats[groups[0]].keys())
  assert isinstance(groups, list) and isinstance(contrasts, list), "Error: expected list types."
  n_tests = len(groups) * len(contrasts)
  adjusted_group_stats = copy.deepcopy(groupStats)
  for group in adjusted_group_stats:
    for contrast in contrasts:
      adjusted_pvalue = min(1, adjusted_group_stats[group][contrast]["globalTestPValue"] * n_tests)
      adjusted_group_stats[group][contrast]["globalTestPValue"] = adjusted_pvalue
  return adjusted_group_stats

def force_to_numeric(value, replacement_value):
  """ 
  Helper function to force special R output to valid numeric forms for json export. 
  
  Checks whether input is numeric or can be coerced to numeric, replaces it with provided default if not.

  Covered are: 
    string input, 
    empty string input, 
    None input, 
    "-inf", "-INF" and positive equivalents that are translated into infinite but valid floats.
  """
  if value is None: # catch None, since None breaks the try except in float(value)
    return replacement_value
  try:
    # Try to convert the value to a float
    num = float(value)
    # Check if the number is infinite or NaN
    if isnan(num):  # num != num is a check for NaN
      return replacement_value
    else:
      return num
  except ValueError:
    # return replacement_value if conversion did not work
    return replacement_value
  
def linear_range_transform(
    input_scalar : float, 
    original_lower_bound : float, 
    original_upper_bound : float, 
    new_lower_bound : float, 
    new_upper_bound : float
  ) -> float:
  """ Returns a linear transformation of a value in one range to another. 
  
  Use to scale statistical values into appropriate size ranges for visualization.
  """
  assert original_lower_bound < original_upper_bound, "Error: lower bound must be strictly smaller than upper bound."
  assert new_lower_bound < new_upper_bound, "Error: lower bound must be strictly smaller than upper bound."
  assert original_lower_bound <= input_scalar <= original_upper_bound, (
    f"Error: input must be within specified bounds but received {input_scalar}"
  )
  # Normalize x to [0, 1]
  normalized_scalar = (input_scalar - original_lower_bound) / (original_upper_bound - original_lower_bound)
  # Map the normalized value to the output range
  output_scalar = new_lower_bound + normalized_scalar * (new_upper_bound - new_lower_bound)
  return output_scalar

def construct_edge_list(similarity_array : np.ndarray, feature_ids : list[str], top_k : int = 30) -> list:
  # Construct edges using all relevant information for edges
  # use similarity array and corresponding feature_ids to determine top-K neighbours
  # use standard linear scale projection for edge weights (assume between 0 and 1)
  """ Constructs edge list for network visualization. """

  assert top_k + 1 <= similarity_array.shape[0], "Error: topK exceeds number of possible neighbors!"
  top_k = top_k + 1 # to accommodate self being among top-k; removed downstream.
  edge_list = []

  # Get top-k neighbor index array; for each row, the top K neighbors are are extracted
  top_k_indices_sorted = np.argsort(similarity_array, axis=1)[:, ::-1][:, :top_k]

  # Using the top-k neighbours, construct the edge list (prevent duplicate edge entries using set comparison)
  node_pairs_covered = set()
  
  # Create edge list
  for row_index, column_indices in enumerate(top_k_indices_sorted):
    # iloc reperesents the row, and hence the current feature
    # column_index
    feature_id = feature_ids[row_index]
    for column_index in column_indices:
      neighbor_id = feature_ids[column_index]
      if frozenset([feature_id, neighbor_id]) not in node_pairs_covered and feature_id is not neighbor_id:
        node_pairs_covered.add(frozenset([feature_id, neighbor_id]))
        # Add the node
        score = similarity_array[row_index, column_index]
        edge = {
          "id": f"{feature_id}_to_{neighbor_id}",
          "from": feature_id,
          "to": neighbor_id,
          "width": transform_similarity_score_to_width(score), # 1 and 30 are the px widths for edges
          "data": {
            "score": score
          }
        }
        edge_list.append(edge)
  return edge_list


def generate_and_attach_long_format_r_data(self, r_json_data : dict) -> None:
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

def write_dict_to_json_file(data : dict, filepath : str, force = False):
  """ 
  Function exports pipeline outout dictionary to json file.
  """
  # validate the that all self object data available
  # self.validate_complete()
  # validate the filepath does not exist or force setting to ensure everything works 
  # assert True
  # construct json string for entire dataset
  json_string = json.dumps(data, indent=2)
  with open(filepath, 'w') as f:
      f.write(json_string)
  return None

def transform_p_value_to_node_size(value : float):
  # transform p value to log10 scale to get order of magnitude scaling
  # take absolute value to make increasing scale; the smaller p, the larger the value
  # cutoff pvalue at size 10 to avoid masking smaller relevant effects, start sizing at 1, equivalent of
  # pvalue = 0.1 (all above are simply min node size of 10)
  # A abs(log10(p_value = 0.1)) = 1 ; this is size 10 for the nodes. Going below means getting size.
  # max visual is log10(0.0000001) -> -6
  lb_node_size = 10
  ub_node_size = 50
  lb_original = 0
  ub_original = 6 # also max considered for visualization, equivalent to 1 in a million probability
  round_decimals = 4
  # make sure the input is valid, and if not, replace with default lb (no size emphasis)
  value = force_to_numeric(value, lb_original) 
  # check for exact zero input before log transformation
  if value != 0:
    size = round(
      linear_range_transform(
        np.clip(np.abs(np.log10(value)), lb_original, ub_original), 
        lb_original, ub_original, lb_node_size, ub_node_size), 
      round_decimals
    )
  else:
    size = ub_node_size # maximum size for p value of zero
  return size

def transform_log2_fold_change_to_node_size(value : float) -> float:
  # transform to abs scale for positive and negative fold to be treated equally 
  # limit to range 0 to 10 (upper bounding to limit avoid a huge upper bound masking smaller effects), 
  # recast to size 10 to 50
  lb_node_size = 10
  ub_node_size = 50
  lb_original = 0
  ub_original = 13 # also max considered for visualization, equivalent of a 8192 fold increase or decrease
  round_decimals = 4
  # make sure the input is valid, and if not, replace with default lb (no size emphasis)
  value = force_to_numeric(value, lb_original)        
  size = round(
    linear_range_transform(
      np.clip(np.abs(value), lb_original, ub_original),
      lb_original, ub_original, lb_node_size, ub_node_size), 
    round_decimals
  )
  return size

def transform_measure_to_node_size(value : float, measure : str):
  """ 
  Function delegates transformation to respective implemenation for measure. 
  
  Supported measures are: "log2foldChange" and "globalTestFeaturePValue"
  """
  size = None
  if measure == "log2FoldChange":
    size = transform_log2_fold_change_to_node_size(value)
  elif measure == "globalTestFeaturePValue":
    size = transform_p_value_to_node_size(value)
  else:
    warn("Measure provided measure has no defined transformation function!")
  assert size is not None, "Error: size computation failed."
  return size

def transform_similarity_score_to_width(score: float):
  """ 
  Function transforms edge similarity score in range 0 to 1 to edge widths using a discrete mapping.

  The range of scores between [0, 1] is divided into 
  0 to <0.2, 0.2 to <0.4, 0.4 to <0.6, 0.6 to <0.8, and 0.8 to < 1,
  Values below 0 are assigned width of 1. Values equal to or above 1 are assigned 26.

  """
  # multiply the score by one hundred and force to integer, take range from [0,1] to [0,100]
  # This avoids numpy floating point problems making discrete cut locations unexpected, e.g. 0.60000000000001 rather
  # than 0.6, leading to values of 0.6... being mapped to the wrong bin!
  score_int = np.int64(score * 100)
  digitized_score = np.digitize(score_int, np.linspace(20, 100, num=5, dtype= np.int64))
  mapping = dict(zip(np.arange(0, 6), [1, 6, 11, 16, 21, 26]))
  width = mapping[digitized_score]
  #if score > 1 or score < 0: 
  #  warn(f"Expected score in range [0,1] but received {score}, determined width to be {width}")
  return width