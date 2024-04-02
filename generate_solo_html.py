import os
from typing import List

def read_to_string(filepath : str) -> str:
  assert os.path.exists(filepath), f"Error: expected file but filepath {filepath} does not exist."
  with open(filepath) as infile:
      filestring = infile.read()
  return filestring

def concatenate_files(filepaths : List[str]):
  output_string = "// Combined javascript \n"
  for filepath in filepaths:
    filestring = read_to_string(filepath)
    output_string = output_string + filestring + "\n"
  return output_string

print("Starting process...")
js_files = [
  os.path.join("src", "javascript", "constants.js"),
  os.path.join("src", "javascript", "heatmap.js"),
  os.path.join("src", "javascript", "load-data-controller.js"),
  os.path.join("src", "javascript", "navtab-control.js"),
  os.path.join("src", "javascript", "network-stabilization.js"),
  os.path.join("src", "javascript", "node-scaling.js"),
  os.path.join("src", "javascript", "topk-selection.js"),
  os.path.join("src", "javascript", "utility-functions.js"),
  os.path.join("src", "javascript", "vis-network-styling-functions.js"),
  os.path.join("src", "javascript", "vis-network-utils.js"),
  os.path.join("src", "javascript", "main.js"),
]
css_file = os.path.join("src", "css", "style.css")
logo_file = os.path.join("assets", "logo-embedding-code.txt")
html_scaffold_file = os.path.join("html-scaffold.html")

js_code = concatenate_files(js_files)
css_code = read_to_string(css_file)
logo_code = read_to_string(logo_file)

scaffold = read_to_string(html_scaffold_file)

scaffold = scaffold.replace("/*#javascriptCodeInsertLocation#*/", js_code)
scaffold = scaffold.replace("#logoInsertLocation#", logo_code)
scaffold = scaffold.replace("#cssStyleSheetInsertLocation#", css_code)
with open("msFeaST_Dashboard_Bundle.html", "w") as html:
  html.write(scaffold)

print("Process Complete...")


print("Generating tmp file for local running...")

import tempfile
import webbrowser
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as tmpfile:
  url = 'file://' + tmpfile.name
  tmpfile.write(scaffold)
  webbrowser.open(url)

print("Complete.")

if False:
  def concatenate_files(file_paths, output_file):
    with open(output_file, 'w') as outfile:
      for fname in file_paths:
        with open(fname) as infile:
          outfile.write(infile.read())

  # List of file paths to concatenate
  js_files = ['file1.js', 'file2.js']
  css_files = ['file1.css', 'file2.css']
  svg_files = ['file1.svg', 'file2.svg']

  # Output files
  js_output = 'combined.js'
  css_output = 'combined.css'
  svg_output = 'combined.svg'

  # Concatenate files
  concatenate_files(js_files, js_output)
  concatenate_files(css_files, css_output)
  concatenate_files(svg_files, svg_output)

  # Create HTML file
  with open('combined.html', 'w') as html_file:
    html_file.write('<html>\n')
    html_file.write('<head>\n')
    html_file.write('<link rel="stylesheet" type="text/css" href="{}">\n'.format(css_output))
    html_file.write('<script src="{}"></script>\n'.format(js_output))
    html_file.write('</head>\n')
    html_file.write('<body>\n')
    html_file.write('<img src="{}" />\n'.format(svg_output))
    html_file.write('</body>\n')
    html_file.write('</html>\n')