# Parameter estimation for spatial ground-motion correlation models using Bayesian inference

[![DOI](https://zenodo.org/badge/542139247.svg)](https://zenodo.org/badge/latestdoi/542139247)

This repository contains the code and data used to estimate parameters of spatial correlation models using Bayesian inference. It also contains implementations of novel correlation models that account for site and path effects in addition to spatial proximity. The methodology and the novel correlation models are discussed in :
> Bodenmann L., Baker J. W., Stojadinovic B. (2022): "Accounting for path and site effects in spatial ground-motion correlation models using Bayesian Inference" (Include link and further info once available).

## Structure

The folder [data](data/) contains the data used to estimate the correlation model parameters, as well as some further data used in the introductory figure in the manuscript, and for the case study application.

The folder [results](results/) contains the post-processed results illustrated in the manuscript.

### Model implementation, parameter estimation and post-processing

To get a quick start, we recommend to use the notebook [introductory_example_estimation.ipynb](introductory_example_estimation.ipynb). This can be opened on a remote server (e.g., Google Colaboratory) and does not require a local python installation. It explains how the proposed correlation models are implemented, how the parameters are estimated using MCMC and how to evaluate the predictive accuracy of the different models. It allows interested users to customize the proposed models and perform parameter estimation. 

The script [main_estimate_postprocess.py](main_estimate_postprocess.py) performs all computations - from model estimation to post-processing - to reproduce the results presented in the paper (and stored in the folder [results](results/)). We do not recommend to run this computationally expensive script on a personal computer. These computations were performed on a high-performance cluster computer that runs on Linux. We used (mini-)conda to set up a virtual environment which is specified in the [environment_numpyro.yml](environment_numpyro.yml) file. To enable efficient MCMC, this script uses the `jax` implementations of the models [models_jax.py](models_jax.py).

### Model application and plotting

To get a quick overview on how the estimated models can be applied in regional risk simulations, we recommend to take a look at the [introductory_example_application.ipynb](introductory_example_application.ipynb) notebook. This can be easily run locally by setting up a virtual (mini-)conda environment using [environment_basic.yml](environment_basic.yml). It uses `numpy` implementation of the models [models_numpy.py](models_numpy.py).

The notebook [figures.ipynb](figures.ipynb) reproduces the figures shown in the manuscript using utility functions from [utils_plotting.py](utils_plotting.py).






