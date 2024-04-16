# msFeaST

The current msFeaST pre-processing and pipeline workflow has been tested on macos and should work identically on linux operating systems. Windows support is currently being worked on. The interactive visualization dashboard works regardless of os on desktop browsers (e.g., firefox, chrome, edge, safari)

# msFeaST Quickstart

To inspect the interactive dashboard for the illustrative examples, please download the *msFeaST_Dashboard_bundle.html* and the ready made data from notebooks\data\omsw_pleurotus_ms2deepscore\dashboard_data.json. Open the html bundle in your browser and load the select and load the data. Changing to the dataview tab shows the now loaded data.

# msFeaST setup

msFeaST is a data analysis workflow that works with Python, R, and web-browser based visualizations (javascript, html, css). To work with msFeaST on your local machine, you need to install the msFeaST python module, install the R dependencies, and download the bundled visualization dashboard (*msFeaST_Dashboard_bundle.html*). If you only want to inspect pre-processed example files using the visual dashboard, the *msFeaST_Dashboard_bundle.html* is the only file needed alongside the .json file. No dependencies need to be installed to do so.

The python module dependencies are managed using conda ([conda installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)), R is installed within this conda environment at an appropriate version, and any dependencies are installed. Please note that R installation related compilation steps may require a couple of minutes on some systems.

To set-up msFeaST, open a terminal from within a suitable working directory and run the following commands one after another. Some commands may request user input regarding package updating, we recommend using entering y (for yes) and pressing enter for these requests. To avoid R path caching issues (see known problems), make sure to open a new command line prompt for installation and set-up of the conda environment prior to any calls of RScript. 

```
conda create --name msfeast_environment python=3.10
conda activate msfeast_environment
conda install conda-forge::r-base=4.3
pip install "git+https://github.com/kevinmildau/msfeast.git"
RScript -e "install.packages(c('remotes', 'BiocManager'), repos='https://cloud.r-project.org')"
RScript -e "remotes::install_version('survival', version = '3.5-8', repos='https://cloud.r-project.org');"
RScript -e "remotes::install_version('listenv', version = '0.9.1', repos='https://cloud.r-project.org');"
RScript -e "remotes::install_version('readr', version = '2.1.5', repos='https://cloud.r-project.org');"
RScript -e "remotes::install_version('tibble', version = '3.2.1', repos='https://cloud.r-project.org');"
RScript -e "remotes::install_version('dplyr', version = '1.1.4', repos='https://cloud.r-project.org');"
RScript -e "BiocManager::install('globaltest', version='3.18')"
jupyter-notebook
```

*What are these commands are doing?*
1. Command creates a conda environment containing an isolated python envioronment for msFeaST to be placed into.
2. Command activates this environment. Subsequent command line calls take effect within this environment.
3. Command installs R at the required version. *1
4. Command installs the msFeaST python module and any required Python dependencies. If using a repository clone, move to the root directory of the package and run "pip install ." instead. *<span style="color:magenta">To avoid RScript command caching problems, we recommend closing the terminal after this step and reopening it, and re-entering ````conda activate msfeast_environment``` to make sure that the RSCript calls install the packages conda R version.</span>* 
5. Command installs R package management dependencies. *1
6. Command installs survival package at development version. *1
7. Command installs listenv package at development version. *1
8. Command installs readr package at development version. *1
9. Command installs tibble package at development version. *1
10. Command installs dplyr package at development version. *1
11. Command install globaltest dependency at development version using Bioconductor release version. Note that the Bioconductor version of 3.18 implies globaltest 5.56.0. *1
12. Command opens jupyter notebook in browser. From here, the msFeaST data pre-processing and processing pipeline examples can be accessed and modified.

*1 *R packages are currently not installed using conda since the conda R package environment is not working reliably for the required packages yet.*

To inspect the R configuration installed run the following command (only works if all packages installed successfully):

```{bash}
RScript -e "catch <- lapply(c('dplyr', 'tibble', 'readr', 'listenv', 'BiocManager', 'globaltest'), library, character.only = TRUE); sessionInfo()"
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

This set-up has been tested on a macos-arm64 machine. It should work identically in linux. 

*UNTESTED*: In windows, the commands should be run from within the the ANACONDA PROMPT required for conda use ([ANACONDA SET-UP](https://www.anaconda.com/download#downloads)). This ANACONDA PROMPT will also be required to start the tool within the right environment and run the jupyter-notebooks.

**Known Problems:**
In rare cases where the RScript command is run from the terminal prior to the conda installation of R as instructed above, the temporary cached path to R used within conda may be faulty.
Here, the cached path to R will be used when installing using RScript rather than the new conda environment specific R path. To avoid this issue, close the terminal after step 4, reopen the terminal, repeat the conda activation (step2), and proceed with the installation. This will avoid RScript calls installing the packages in the wrong R path. See the following github issue for further information: [r path issue](https://github.com/conda/conda/issues/1258#issuecomment-91035641).
