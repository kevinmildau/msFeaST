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
  pipeline.run_and_attach_kmedoid_grid([8,9,10,10,12,13,14,15])
  #msfeast._plot_kmedoid_grid(pipeline.kmedoid_grid)
  pipeline.select_kmedoid_settings(iloc = 0)
  assert isinstance(pipeline.assignment_table, pd.DataFrame)
  pipeline.run_r_testing_routine("tmp_output")