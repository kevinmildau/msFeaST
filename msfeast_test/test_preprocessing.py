from msfeast.preprocessing import align_treatment_and_quantification_table
import pandas as pd

def test_align_treatment_and_quantification_table():
  # Generate test case
  df1 = pd.DataFrame({"sample_id": ["1", "2", "3", "4"], "data":[1,2,3,4]})
  df1.set_index("sample_id", drop=False, inplace=True)

  df2 = pd.DataFrame({"sample_id": ["4", "2", "3", "1"], "data":[4,2,3,1]})
  df2.set_index("sample_id", drop=False, inplace=True) 
  
  # Align
  df1_aligned, df2_aligned = align_treatment_and_quantification_table(df1, df2)
  # Expect equality if aligned correctly and index dropped
  df1.reset_index(inplace=True, drop = True)
  assert df1_aligned.equals(df1)
  assert df2_aligned.equals(df1)
  return None