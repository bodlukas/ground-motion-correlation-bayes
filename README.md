# Parameter estimation for spatial ground-motion correlation models using Bayesian inference

[![DOI](https://zenodo.org/badge/542139247.svg)](https://zenodo.org/badge/latestdoi/542139247)

This repository contains the code to perform Bayesian parameter estimation for spatial correlation models, as well as implementations of novel correlation models that account for site and path effects in addition to spatial proximity. The methodology and the novel correlation models are presented in :

> Bodenmann L., Baker J. W., Stojadinovic B. (20xx): "Accounting for path and site effects in spatial ground-motion correlation models using Bayesian inference" (Include link and further info once available).

## Structure

The folder [data](data/) contains the data used to estimate the correlation model parameters, as well as some further data used in the introductory figure in the manuscript, and for the case study application.

The folder [results](results/) contains the post-processed results illustrated in the manuscript.

### Model implementation, parameter estimation and post-processing

**Quick start** Open the notebook [introductory_example_estimation.ipynb](introductory_example_estimation.ipynb) on a hosted Jupyter notebook service (e.g., Google Colab). It explains how the models are implemented, how the parameters are estimated using MCMC and how to compute the predictive accuracy. Because it does not require any local python setup you can easily customize the proposed models and perform the computations yourself.

The script [main_estimate_postprocess.py](main_estimate_postprocess.py) performs all computations - from model estimation to post-processing - to reproduce the results presented in the paper (and stored in the folder [results](results/)). These computations were performed on a high-performance computing cluster operating with Linux. The employed (mini-)conda environment is specified in the [environment_numpyro.yml](environment_numpyro.yml) file. To enable efficient MCMC, this script uses the `jax` implementations of the models [models_jax.py](models_jax.py). We do not recommend to run this computationally expensive script on a personal computer. 

### Model application and plotting

**Quick start** Take a look at the [introductory_example_application.ipynb](introductory_example_application.ipynb) notebook to see how the estimated models can be applied in regional risk simulations. It uses `numpy` implementation of the models [models_numpy.py](models_numpy.py) and the required packages are specified in [environment_basic.yml](environment_basic.yml).

The notebook [figures.ipynb](figures.ipynb) reproduces the figures shown in the manuscript using utility functions from [utils_plotting.py](utils_plotting.py).

## Installation

We performed model estimation on Linux and did not test other opearting systems. To perform model estimation on your local machine, you can set up a Windows Subsystem for Linux ([WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). Then install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and create an environment as `conda env create -f environment_numpyro.yml`.

Application of the estimated models and the plotting of the results should also work on other operating systems than Linux and you can create a miniconda environment as `conda env create -f environment_basic.yml`.