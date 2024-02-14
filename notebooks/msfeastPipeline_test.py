# run using thew following line of code from within notebook directory:
# pytest
# or for showing prints:
# pytest -s

import msfeastPipeline as msfeast
import pytest
import os

filepath_test_spectra = os.path.join("test_data", "test_spectra.mgf")
filepath_test_quant_table = os.path.join("test_data", "test_quant_table.csv")
filepath_test_treat_table = os.path.join("test_data", "test_treat_table.csv")

def test_load_data():
  assert 1 == 1, ""


def test_load_data2():
  with pytest.raises(Exception) as e_info:
    x = 1 / 0

def test_feature_id_selector():
  assert msfeast._assert_feature_ids_valid(["1", "2", "3"]) is None
  with pytest.raises(Exception) as e_info:
    msfeast._assert_feature_ids_valid([])
  with pytest.raises(Exception) as e_info:
    msfeast._assert_feature_ids_valid("not correct")
  with pytest.raises(Exception) as e_info:
    msfeast._assert_feature_ids_valid([1,2,3])


def test_run_pipeline_integration():
  pipeline = msfeast.Msfeast()
  assert isinstance(pipeline, msfeast.Msfeast)
  msfeast._assert_filepath_exists(filepath_test_spectra)
  pipeline.attach_spectral_data_from_file(filepath_test_spectra, identifier_key="scans")
  assert pipeline.spectra_matchms is not None

def test_data_attach():
  import pandas as pd
  treat_table = pd.read_csv(filepath_test_treat_table)
  quant_table = pd.read_csv(filepath_test_quant_table)
  pipeline = msfeast.Msfeast()
  pipeline.attach_spectral_data_from_file(filepath_test_spectra, identifier_key="scans")
  pipeline.attach_quantification_table(quant_table)
  pipeline.attach_treatment_table(treat_table)
  assert all(pipeline.treatment_table == treat_table)
  assert all(pipeline.quantification_table == quant_table)

def test_cosine_sim():
  import pandas as pd
  import numpy as np
  treat_table = pd.read_csv(filepath_test_treat_table)
  quant_table = pd.read_csv(filepath_test_quant_table)
  pipeline = msfeast.Msfeast()
  pipeline.attach_spectral_data_from_file(filepath_test_spectra, identifier_key="scans")
  pipeline.attach_quantification_table(quant_table)
  pipeline.attach_treatment_table(treat_table)
  assert all(pipeline.treatment_table == treat_table)
  assert all(pipeline.quantification_table == quant_table)
  pipeline.run_spectral_similarity_computations("ModifiedCosine")
  assert isinstance(pipeline.similarity_array, np.ndarray)
  n_spec = len(pipeline.spectra_matchms)
  assert pipeline.similarity_array.shape == (n_spec, n_spec)

def test_kmedoids():
  import pandas as pd
  import numpy as np
  treat_table = pd.read_csv(filepath_test_treat_table)
  quant_table = pd.read_csv(filepath_test_quant_table)
  pipeline = msfeast.Msfeast()
  pipeline.attach_spectral_data_from_file(filepath_test_spectra, identifier_key="scans")
  pipeline.attach_quantification_table(quant_table)
  pipeline.attach_treatment_table(treat_table)
  assert all(pipeline.treatment_table == treat_table)
  assert all(pipeline.quantification_table == quant_table)
  pipeline.run_spectral_similarity_computations("ModifiedCosine")
  assert isinstance(pipeline.similarity_array, np.ndarray)
  n_spec = len(pipeline.spectra_matchms)
  assert pipeline.similarity_array.shape == (n_spec, n_spec)
  pipeline.run_and_attach_kmedoid_grid()
  pipeline.select_kmedoid_settings(iloc = 0)
  assert isinstance(pipeline.assignment_table, pd.DataFrame)
  print(pipeline.assignment_table.head())

def test_tsne():
  import pandas as pd
  import numpy as np
  treat_table = pd.read_csv(filepath_test_treat_table)
  quant_table = pd.read_csv(filepath_test_quant_table)
  pipeline = msfeast.Msfeast()
  pipeline.attach_spectral_data_from_file(filepath_test_spectra, identifier_key="scans")
  pipeline.attach_quantification_table(quant_table)
  pipeline.attach_treatment_table(treat_table)
  assert all(pipeline.treatment_table == treat_table)
  assert all(pipeline.quantification_table == quant_table)
  pipeline.run_spectral_similarity_computations("ModifiedCosine")
  assert isinstance(pipeline.similarity_array, np.ndarray)
  n_spec = len(pipeline.spectra_matchms)
  assert pipeline.similarity_array.shape == (n_spec, n_spec)
  pipeline.run_and_attach_tsne_grid()
  pipeline.select_tsne_settings(iloc = 0)
  assert isinstance(pipeline.embedding_coordinates_table, pd.DataFrame)
  print(pipeline.embedding_coordinates_table.head())



if __name__ == "__main__":
  print("Starting Main")
  import pandas as pd
  import numpy as np
  treat_table = pd.read_csv(filepath_test_treat_table)
  quant_table = pd.read_csv(filepath_test_quant_table)
  pipeline = msfeast.Msfeast()
  pipeline.attach_spectral_data_from_file(filepath_test_spectra, identifier_key="scans")
  pipeline.attach_quantification_table(quant_table)
  pipeline.attach_treatment_table(treat_table)
  assert all(pipeline.treatment_table == treat_table)
  assert all(pipeline.quantification_table == quant_table)
  pipeline.run_spectral_similarity_computations("ModifiedCosine")
  assert isinstance(pipeline.similarity_array, np.ndarray)
  n_spec = len(pipeline.spectra_matchms)
  assert pipeline.similarity_array.shape == (n_spec, n_spec)
  pipeline.run_and_attach_kmedoid_grid([10])
  #msfeast._plot_kmedoid_grid(pipeline.kmedoid_grid)
  pipeline.select_kmedoid_settings(iloc = 0)
  assert isinstance(pipeline.assignment_table, pd.DataFrame)
  pipeline.run_and_attach_tsne_grid()
  #msfeast._plot_tsne_grid(pipeline.tsne_grid)
  pipeline.select_tsne_settings(iloc = 0)
  #pipeline.plot_selected_embedding()
  assert isinstance(pipeline.embedding_coordinates_table, pd.DataFrame)
  print(pipeline.embedding_coordinates_table.head())
  #pipeline.run_r_testing_routine("tmp_output")
  feature_ids = msfeast._extract_feature_ids_from_spectra(pipeline.spectra_matchms)
  edges = msfeast._construct_edge_list(pipeline.similarity_array, feature_ids, top_k = 5)
  print(edges)


if True:
  import json
  import pandas as pd

  
  


  
  
  json_data = msfeast._load_and_validate_r_output("tmp_output/test_r_output.json")
  pandas_data = msfeast._convert_r_output_to_long_format(json_data)
  print(pipeline.assignment_table.head())
  node_entries = construct_nodes(json_data, pipeline.assignment_table, pipeline.embedding_coordinates_table)
  
  print(json_data["set_specific"])
  
  def construct_sets():
    # Construct set specific statisical data, basic fetch from R output
    # for now includes only: group key --> contrast key --> measure key --> value
    # Also detect the number of sets and apply Bonferroni adjustment to p-values based on number of sets.
    # Using set identifier and assignment table also add the set corresponding feature_ids. This might be
    # useful down the line when adding additonal visual components in js.
    ...

  def construct_set_keys():
    # fetch set keys from assignment table or R output
    ...
  
  def construct_contrast_keys():
    # basic fetch of contrast keys from R output. Maybe include as a third entry: 
    # feature_specific, set_specifc, contrast_keys 
    # for easy fetch
    ...
  
  def construct_tabular_output():
    # construct long format data frame of entire output for excel export using columns:
    # type (feature or set)
    # id (feature_id or set_id)
    # contrast (contrast key)
    # measure (measure key, from feature specific and set specific measures)
    # value (number or string with the appropriate data)
    # size (the recast size used in visualization?)
    ...
  
  def construct_feature_keys():
    # probably not needed
    ...
  print ("Running code complete. ")
  