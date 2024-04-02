
# Feature Set Tester Pipeline and Dashboard Repo

This repository is currently in development and not yet ready for use. This readme will be updated once the the tool changes fro 0.0.dev to 0.0.1dev



# msFeaST setup

msFeaST is a data analysis workflow that works with Python, R, and web-browser based visualizations (javascript, html, css). To work with msFeaST on your local machine, you need to install the msFeaST python module, install the R dependencies, and download the bundled visualization dashboard.

The python module dependencies are managed using conda ( [conda installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)) ), R is installed within this conda environment at an appropriate version, and any dependencies are installed. Please not that R installation related compilation steps may require a couple of minutes on some systems.

```{bash}
conda create --name msfeast_env python=3.10
conda activate msfeast_env
conda install 
pip install "git+https://github.com/kevinmildau/msfeast.git"
```
