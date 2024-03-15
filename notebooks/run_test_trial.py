if __name__ == "__main__":
  print("Starting Main...")
  import msfeast.pipeline
  import os
  import pandas as pd


  print("Define Filepaths...")
  test_data_directory = os.path.join("notebooks", "test_data_input")
  filepath_test_spectra = os.path.join(test_data_directory, "test_spectra.mgf")
  filepath_test_quant_table = os.path.join(test_data_directory, "test_quant_table.csv")
  filepath_test_treat_table = os.path.join(test_data_directory, "test_treat_table.csv")
  output_directory = os.path.join("notebooks", "test_output")
  r_output_filename = os.path.join("r_output.json")
  r_filepath = os.path.join(output_directory, r_output_filename)
  dashboard_output_filepath = os.path.join(output_directory, "dashboard_data.json")
  
  print("Loading Input Data...")
  treat_table = pd.read_csv(filepath_test_treat_table)
  quant_table = pd.read_csv(filepath_test_quant_table)

  print("Initializing pipeline...")
  pipeline = msfeast.pipeline.Msfeast()

  print("Attaching data...")
  pipeline.attach_spectral_data_from_file(filepath_test_spectra, identifier_key="scans")
  pipeline.attach_quantification_table(quant_table)
  pipeline.attach_treatment_table(treat_table)

  print("Running spectral similarity computations...")
  pipeline.run_spectral_similarity_computations("ModifiedCosine")
  
  print("Run kmedoid grid...")
  pipeline.run_and_attach_kmedoid_grid([5])
  pipeline.select_kmedoid_settings(iloc = 0)

  print("Run t-sne grid...")
  pipeline.run_and_attach_tsne_grid()
  pipeline.select_tsne_settings(iloc = 0)

  print("Initializing R runtime...")
  if os.path.isfile(r_filepath):
        os.remove(r_filepath)
  pipeline.run_r_testing_routine(output_directory, r_output_filename, top_k = 10)
  
  print("exporting json data...")
  pipeline.export_to_json_file(dashboard_output_filepath)

  print("Reached end of trial run.")
