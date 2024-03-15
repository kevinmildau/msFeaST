from math import isnan
import numpy as np
import json
import pandas as pd
import copy

def load_and_validate_r_output(filepath : str) -> dict:
  """ Function loads and validates r output file.
  Returns the r output json data as a python dictionary. First level entries are:
  
  feature_specific
  --> feature id specific data, subdivided into contrast specific, measure specific, and finally value. I.e. for each
  feature id, for each contrast key, for each measure key, there will be a corresponding value in a nested dict
  of hierarchy [feature_identifier][contrast_key][measure_key] --> value. Feature_identifier, contrast_key, and
  measure keys are data dependent strings. The hierarchy gives the type of entry.
  
  set_specific
  --> set id specific data, subdivided into contrast specific, measure specific, and finally value
  feature_id_keys. Similar to feature_id.
  
  set_id_keys
  --> list of set identifiers
  
  contrast_keys
  --> list of contrast keys
  
  feature_specific_measure_keys
  --> list of measure keys for the feature specific entry
  
  set_specific_measure_keys
  --> list of measure keys for the set specific entries
  """
  json_data = json.load(open(filepath, mode = "r"))
  # Assert that the top level keys are all populated (partial input assertion testing only!)
  assert json_data["feature_specific"]is not None, "ERROR: Expected feature_specific  entry to not be empty."
  assert json_data["feature_specific"].keys() is not None, "ERROR: Expected feature specific keys."
  assert json_data["set_specific"] is not None, "ERROR: Expected set_specific  entry to not be empty."
  assert json_data["set_specific"].keys() is not None, "ERROR: Expected set specific keys."
  assert json_data["feature_id_keys"] is not None, "ERROR: Expected feature id keys entry to not be empty."
  assert json_data["set_id_keys"] is not None, "ERROR: Expected feature id keys entry to not be empty."
  assert json_data["contrast_keys"] is not None, "ERROR: Expected contrast_keys entry to not be empty."
  assert json_data["feature_specific_measure_keys"] is not None, "ERROR: Expected feature_specific_measure_keys to not be empty."
  assert json_data["set_specific_measure_keys"] is not None, "ERROR: Expected set_specific_measure_keys entry to not be empty."
  # TODO: for robustness, Cross compare R entries against python data from pipeline (contrasts, setids, fids)
  # TODO: for robustness, Validate each feature_id and set_id entry
  # return the validated data
  return json_data


def _construct_nodes(
    r_json_data : dict, 
    assignment_table : pd.DataFrame, 
    embedding_coordinates_table: pd.DataFrame
  ) -> list:
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
    node = {
      "id" : feature_key,
      "size": 10, # --> measure derived variable, set to 10 for now.
      "group": feature_group, # feature derived variable
      "x": coordinates["x"].values[0] * 100, # default scaling for better visual representation
      "y": coordinates["y"].values[0] * 100, # default scaling for better visual representation
      "data" : r_json_data["feature_specific"][feature_key]
    }
    # For specific expected measures, translate the measure into node size: supported: p-value & log2foldchange
    # Currently no scale available.
    lb_node_size = 10
    ub_node_size = 50
    for contrast_key in node["data"].keys(): 
      for measure_key in node["data"][contrast_key].keys():
        # Only add nodeSize conversion if among the supported keys for conversion
        if measure_key in measure_keys:
          value = node["data"][contrast_key][measure_key] 
          size = None
          if measure_key == "log2FoldChange":
            # transform to abs scale for positive and negative fold to be treated equally 
            # limit to range 0 to 10 (upper bounding to limit avoid a huge upper bound masking smaller effects), 
            # recast to size 10 to 50
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
          if measure_key == "globalTestFeaturePValue":
            # transform p value to log10 scale to get order of magnitude scaling
            # take absolute value to make increasing scale; the smaller p, the larger the value
            # cutoff pvalue at size 10 to avoid masking smaller relevant effects, start sizing at 1, equivalent of
            # pvalue = 0.1 (all above are simply min node size of 10)
            # A abs(log10(p_value = 0.1)) = 1 ; this is size 10 for the nodes. Going below means getting size.
            # max visual is log10(0.0000001) -> -6
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
          #assert size is not None, "Error: size computation failed."
          node["data"][contrast_key][measure_key] = {
            "measure": str(value),
            "nodeSize": size,
          }
    # Attach the processed node to the node_entries list
    node_entries.append(node)
  return(node_entries)

def _apply_bonferroni_correction_to_group_stats(groupStats, alpha = 0.01):
  """ Applies Bonferroni adjustment to p-values in groupStats
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

def construct_edge_list(similarity_array : np.ndarray, feature_ids : list[str], top_k : int = 30) -> List:
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
          "width": round(linear_range_transform(score, 0, 1, 1, 30), 2), # 1 and 30 are the px widths for edges
          "data": {
            "score": score
          }
        }
        edge_list.append(edge)
  return edge_list