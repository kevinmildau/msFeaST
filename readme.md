
# Feature Set Tester Pipeline and Dashboard Repo

This repository is currently in development and not yet ready for use. This readme will be updated once the the tool changes fro 0.0.dev to 0.0.1dev



# msFeaST setup

msFeaST is a data analysis workflow that works with Python, R, and web-browser based visualizations (javascript, html, css). To work with msFeaST on your local machine, you need to install the msFeaST python module, install the R dependencies, and download the bundled visualization dashboard.

The python module dependencies are managed using conda ([conda installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)), R is installed within this conda environment at an appropriate version, and any dependencies are installed. Please note that R installation related compilation steps may require a couple of minutes on some systems.

To set-up msFeaST, open a terminal from within a suitable working directory and run the following commands one after another. Some commands may request user input regarding package updating, we recommend using entering y and pressing enter for these requests. To avoid R path caching issues (see known problems), make sure to open a new command line prompt for installation and set-up the conda environment prior to any calls of RScript. 

```{bash}
conda create --name msfeast_env python=3.10
conda activate msfeast_env
conda install conda-forge::r-base=4.3
pip install "git+https://github.com/kevinmildau/msfeast.git"
RScript -e "install.packages(c('dplyr', 'tibble', 'readr', 'listenv', 'BiocManager'), repos='https://cloud.r-project.org')"
RScript -e "BiocManager::install('globaltest', force = TRUE)"
jupyter-notebook
```

*What are these commands are doing?*

1. Command creates a conda environment containing an isolated python envioronment for msFeaST to be placed into.
2. Command activates this environment. Subsequent command line calls take effect within this environment.
3. Command installs R at the required version. *1
4. Command installs the msFeaST python module and any required Python dependencies. If using a repository clone, move to the root directory of the package and run "pip install ." instead.
5. Command installs general R package dependencies (may take a while because of compilation steps). *1
6. Command install globaltest dependency. *1
7. Command opens jupyter notebook in browser. From here, the msFeaST data processing pipeline demo can be accessed and modified.

*1 *R packages are currently not installed using conda since the conda R package environment is not working reliably for the required packages yet.*

To inspect the R configuration installed run the following command (only works if all packages installed successfully):
```{bash}
RScript -e "catch <- lapply(c('dplyr', 'tibble', 'readr', 'listenv', 'BiocManager', 'globaltest'), library, character.only = TRUE); sessionInfo()"
```


This should return, among other things, the following or similar session information:

```{text}
R version 4.3.3 (2024-02-29)
Platform: aarch64-apple-darwin20.0.0 (64-bit)
Running under: macOS Ventura 13.5.1

other attached packages:
[1] globaltest_5.56.0   survival_3.5-8      BiocManager_1.30.22
[4] listenv_0.9.1       readr_2.1.5         tibble_3.2.1       
[7] dplyr_1.1.4      
```

This set-up has been tested on a macos-arm64 machine. It should work identically in linux. In windows, the commands must be run from within the the ANACONDA PROMPT ({ANACONDA SET-UP}(https://www.anaconda.com/download#downloads)).


**Known Problems:**
In rare cases where R is run prior to the conda installation of R, and R is called, the temporary cached path to the R used within conda may be faulty.
Here, the cached path to R will be used when installing using RScript rather than the new conda repo specific R path. Make sure to open a new terminal when following the installation guide to avoid RScript calls installing the packages in the wrong R path. See the following github issue for further information: {r path issue}(https://github.com/conda/conda/issues/1258#issuecomment-91035641).
