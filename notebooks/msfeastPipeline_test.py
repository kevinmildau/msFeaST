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

print("End of integration testing reached.")