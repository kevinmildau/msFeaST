# msFeaST

The current msFeaST pre-processing and pipeline workflow has been tested on macos and should work identically on linux operating systems. Windows support is currently being worked on. The interactive visualization dashboard works regardless of os on desktop browsers (e.g., firefox, chrome, edge, safari)

# msFeaST Quickstart

To inspect the interactive dashboard for the illustrative examples, please download the *msFeaST_Dashboard_bundle.html* and the ready made data from notebooks\data\omsw_pleurotus_ms2deepscore\dashboard_data.json. Open the html bundle in your browser and load the select and load the data. Changing to the dataview tab shows the now loaded data. 

To run msFeaST using your own data, follow the installation instructions (msFeaST setup). Navigate to the notebook folder and open the preprocessing_mushroom_type_comparison.ipynb and msfeast_pipeline_mushroom_type_comparison.ipynb notebooks on your local machine. These notebooks contain a complete example of quantification table, metadata table, and spectral data processing required for msFeaST, as well as a complete use-case example. To make use of your own data, change the data filepath arguments to your own data file location and run the pipeline. Text in magenta italics font highlights required user input for the pipeline. The jupyter-notebook pipeline produces the a text file in json format that can be interactively explored in the interactive dashboard.

# msFeaST setup

msFeaST is a data analysis workflow that works with Python, R, and web-browser based visualizations (javascript, html, css). To work with msFeaST on your local machine, you need to install the msFeaST python module, install the R dependencies, and download the bundled visualization dashboard (*msFeaST_Dashboard_bundle.html*). If you only want to inspect pre-processed example files using the visual dashboard, the *msFeaST_Dashboard_bundle.html* is the only file needed alongside the .json file. No dependencies need to be installed to do so.

The python module dependencies are managed using conda ([conda installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)), R is installed within this conda environment at version 4.3.3, and any R dependencies are installed. Please note that R installation related compilation steps may require a couple of minutes on some systems.

To set-up msFeaST, open a terminal from within a suitable working directory and run the following commands one after another. Some commands may request user input regarding package updating, we recommend entering y (for yes) and pressing enter for these requests.

```{bash}
conda create --name msfeast_environment python=3.10
conda activate msfeast_environment
conda install conda-forge::r-base=4.3
```

*What are these commands are doing?*
1. Command creates a conda environment containing an isolated python envioronment for msFeaST to be placed into.
2. Command activates this environment. Subsequent command line calls take effect within this environment.
3. Command installs R at the required version.

To avoid problems with the R path (see known problems), close (kill) the open terminal, and reopen a new terminal window to run the following commands to complete the installation:

```
conda activate msfeast_environment
pip install "git+https://github.com/kevinmildau/msfeast.git"
rscript -e "install.packages(c('remotes', 'BiocManager'), repos='https://cloud.r-project.org')"
rscript -e "remotes::install_version('Matrix', version = '1.6-5', repos='https://cloud.r-project.org');"
rscript -e "remotes::install_version('survival', version = '3.5-8', repos='https://cloud.r-project.org');"
rscript -e "remotes::install_version('listenv', version = '0.9.1', repos='https://cloud.r-project.org');"
rscript -e "remotes::install_version('readr', version = '2.1.5', repos='https://cloud.r-project.org');"
rscript -e "remotes::install_version('tibble', version = '3.2.1', repos='https://cloud.r-project.org');"
rscript -e "remotes::install_version('dplyr', version = '1.1.4', repos='https://cloud.r-project.org');"
rscript -e "BiocManager::install('globaltest', version='3.18')"
```

*What are these commands are doing?*
1. Command activates the created environment in the newly opened terminal.
2. Command installs the msFeaST python module and any required Python dependencies. If using a repository clone, move to the root directory of the package and run "pip install ." instead. *<span style="color:magenta">To avoid rscript command caching problems, we recommend closing the terminal after this step and reopening it, and re-entering ````conda activate msfeast_environment``` to make sure that the RSCript calls install the packages conda R version.</span>* 
3. Command installs R package management dependencies. *1
4. Command Installs the R package Matrix at required version (indirect requirement for globaltest) *1
5. Command installs survival package at development version. (indirect requirement for globaltest) *1
6. Command installs listenv package at development version. *1
7.  Command installs readr package at development version. *1
8.  Command installs tibble package at development version. *1
9.  Command installs dplyr package at development version. *1
10. Command install globaltest dependency at development version using Bioconductor release version. Note that the Bioconductor version of 3.18 implies globaltest 5.56.0 & R version 4.3.3  *1

*1 *R packages are currently not installed using conda since the conda R package environment is not working reliably for the required packages yet. Instead, remotes and biocmanager are used to control R package versions.*

After initial set-up is done, the jupyter-notebook can be accessed in the right environment using these two commands:
```{bash}
conda activate msfeast_environment
jupyter-notebook
```
where it is important that the terminal used is newly opened and does not have R or rscript paths buffered (see known problems).

To inspect the R configuration installed run the following command (only works if all packages installed successfully):

```{bash}
rscript -e "catch <- lapply(c('dplyr', 'tibble', 'readr', 'listenv', 'BiocManager', 'globaltest'), library, character.only = TRUE); sessionInfo()"
```

This should return, among other things, the following session information (platform and OS may differ):

```{text}
R version 4.3.3 (2024-02-29)
Platform: aarch64-apple-darwin20.0.0 (64-bit)
Running under: macOS Ventura 13.5.1

other attached packages:
[1] globaltest_5.56.0   survival_3.5-8      BiocManager_1.30.22
[4] listenv_0.9.1       readr_2.1.5         tibble_3.2.1       
[7] dplyr_1.1.4      
```

If this command does not work, msFeaST will not be able to run successfully.

This set-up has been tested on a macos-arm64 machine. It should work identically in linux. 

*UNTESTED*: In windows, the commands should be run from within the the ANACONDA PROMPT required for conda use ([ANACONDA SET-UP](https://www.anaconda.com/download#downloads)). This ANACONDA PROMPT will also be required to start the tool within the right environment and run the jupyter-notebooks.

**Known Problems:**
1. In rare cases where the rscript command is run from the terminal prior to the conda installation of R as instructed above, the temporary cached path to R used within conda may be faulty. 
Here, the cached path to R will be used when installing using rscript rather than the new conda environment specific R path. 
To avoid this issue, close the terminal after step 4, reopen the terminal, repeat the conda activation (step2), and proceed with the installation. 
This will avoid rscript calls installing the packages in the wrong R path. 
See the following github issue for further information: [r path issue](https://github.com/conda/conda/issues/1258#issuecomment-91035641).
2. Some terminal interfaces such as the vscode terminal may default into specific conda environments. This default move into conda may cause the path miss-alignment issues of point 1. If problems persists, check the R and rscript path via the following commands:
```
type R
type rscript
```
Both should be situated within the conda environment created. The safest way to work with msfeast and avoiding path mismatch is to open a new terminal and activate the conda environment immediately. 
