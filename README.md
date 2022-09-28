# Parameter estimation for spatial ground-motion correlation models using Bayesian inference

This repository contains the code and data used to estimate parameters of spatial correlation models using Bayesian inference. It also contains implementations of novel correlation models that account for site and path effects in addition to spatial proximity. The methodology and the novel correlation models are discussed in :
> Bodenmann L., Baker J. W., Stojadinovic B. (2022): "Accounting for path and site effects in spatial ground-motion correlation models using Bayesian Inference" (Include link and further info once available).

## Getting started

To get a quick start, we recommend to use the notebook `introductory_example_estimation.ipynb`. This can be opened in Google Colab (c) and does not require any local dependencies. It shows how the proposed correlation models are implemented, how the parameters are estimated using MCMC and how to evaluate the predictive accuracy of the different models. It allows interested users to customize their own models and perform parameter estimation. 

To get a quick overview on how the estimated models can be applied in regional risk simulations, we recommend to take a look at the `Application_CaseStudy_SFBay.ipynb` notebook. This requires minimal local dependencies. 

## Structure

The folder [results](results/) `data` contains the data used to estimate the correlation model parameters, as well as some further data used in the introductory figure in the manuscript, and for the case study application.

The folder `results` contains the post-processed results illustrated in the manuscript.

The script `main_estimation_preprocessing.py` performs all computations - from model estimation to post-processing - to reproduce the results presented in the paper (and stored in the above mentioned folder). We do not recommend to run this computationally expensive script on a personal computer. These computations were performed on a high-performance cluster computer that runs on Linux. We used (mini-) conda to set up a virtual environment which is specified in the `environment_numpyro.yml` file. 

The notebook `plot_figures.ipynb` shows how the results are used to create the figures shown in the manuscript.

