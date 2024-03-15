from math import isnan

def _force_to_numeric(value, replacement_value):
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