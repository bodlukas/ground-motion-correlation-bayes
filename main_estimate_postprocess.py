# -------------------
# Main script to reproduce results
# -------------------
'''
This script shows all required steps to compute 
the results used in the manuscript's figures and tables. 

Note that the model estimations are computationally
expensive. We ran the script on a high-performance 
cluster computer and parallelized various for-loops. 

For a thorough explanation of the estimation and 
post-processing computations check out the following notebook:
...
This can be easily imported to colab and you don't have
to set up a special environment locally.

'''


import pandas as pd
import numpy as np
import arviz as az
import os
import numpyro as numpyro
import jax as jax

# Change JAX default from float32 to float64
jax.config.update("jax_enable_x64", True)

from numpyro.infer import MCMC, NUTS, log_likelihood
from scipy.special import logsumexp

# -------------------
# Settings
# -------------------
'''
if load = True: 
    The script loads all pre-calculated results.
else: 
    New models are estimated (not recommended on a personal 
    computer) and the same results are calculated again 
    (note that we used a different seed, so the results 
    won't match to the last digit).

if test_data = True:
    Load test data (2019 Ridgecrest earthquake sequence) 
    and compute ouf-of-sample performance.
    Test data sets are NOT in repository because the employed
    ergodic GMM is neither ours nor is it publicly available
    (see paper for reference). 
    The computed out-of-sample performance metrics are
    available in the repository and loaded in this script.
'''
load = True
test_data = False
if load == False:
    # Set some specific settings for inference
    numpyro.set_platform("cpu")
    # Number of CPUs -> To sample several MCMC chains in parallell
    numpyro.set_host_device_count(4)
    num_chains = 4
    num_warmup = 1000
    num_samples = 1000


# -------------------
# Specify paths
# -------------------

# training data (PEER NGA-West2)
path_data = os.path.join(os.getcwd() + os.sep + 'data' 
    + os.sep + 'ngawest2' + os.sep)

# Store or load results for sampled parameter sets from the posterior
path_res_param = os.path.join(os.getcwd() + os.sep + 'results' 
    + os.sep + 'PosteriorParameters' + os.sep)

# Store or load results for in-sample model performance
path_res_is = os.path.join(os.getcwd() + os.sep + 'results' 
    + os.sep + 'PostProcessed' + os.sep + 'in_sample' + os.sep)

if test_data:
    # testing data (Ridgecrest)
    # Datasets are not in repository (see above)
    path_test_data = os.path.join(os.getcwd() + os.sep + 'data' 
        + os.sep + 'ridgecrest' + os.sep)

# Store or load results for out-of-sample model performance
path_res_oos = os.path.join(os.getcwd() + os.sep + 'results' 
    + os.sep + 'PostProcessed' + os.sep + 'out_of_sample' + os.sep)

# -------------------
# Event-Specific models
# -------------------

# This was only performed for Sa(T=1s)
period = 1 # Sa(T) where T is the period
str_per = str(int(period*100))

# Choose events which we'll plot in more detail
# San Simeon, Yorba Linda, Chuetsu-oki, Hector mine
eqs_detail = [177, 167, 278, 158]

if load:
    # In-Sample LPPD for each event in the training set
    # Used in Figure 6
    df_es = pd.read_csv(path_res_is + 
        'LPPD_EventSpecific_EventSpecific_T' + str_per + '.csv')
    # For the events specified above:
    # Sampled parameter sets (used in Figure 3) and conditional
    # log-likelihood (used in Figure 5)
    dfs_es_detail = []
    for eqid in eqs_detail:
        dfs_es_detail.append(pd.read_csv(path_res_is + 'LogLik_EventSpecific_eqid' + 
                str(eqid) + '_EventSpecific_T' + str_per + '.csv', index=False))
else:
    # Load model
    from models_jax import modelE

    # import data
    df = pd.read_csv(path_data + 'data_ngawest2_T' + str_per + '.csv')

    # Assemble dataframe to store results
    grouped = df.groupby('eqid')
    df_es = pd.DataFrame()
    df_es['eqid'] = grouped.size().keys()
    df_es['num_recs'] = grouped.size().values
    df_es['magnitude'] = [group.magnitude.values[0] for _,group in grouped]
    # Initialize empty list where we will store the LPPD of each event
    lppds = [] 
    # Generate separate seeds for each event
    rng = np.random.default_rng(91)
    seeds = rng.integers(0, 100000, len(df_es))

    for i, eqid in enumerate(df_es.eqid.values):
        group = df[df.eqid==eqid].copy()
        seed = seeds[i]
        # Assemble inputs and output
        X = [group[['epi_dist', 'epi_azimuth', 'vs30']].values]
        z = [group['scaled_deltaW'].values]
        dat_list = dict(X=X, eqids=[eqid], z=z)
        # Specify model and MCMC parameters
        model = MCMC(NUTS(modelE), num_warmup=num_warmup, 
            num_samples=num_samples, num_chains=num_chains)
        # Run MCMC
        model.run(jax.random.PRNGKey(seed), **dat_list)
        # Get sampled parameter sets
        samples = model.get_samples()
        # Compute log-likelihood for each sample
        cond_loglik = log_likelihood(modelE, samples, **dat_list)['z_' + str(eqid)]
        # Store detailed results for specific events (Figures 3 and 5)
        if np.isin(eqid, eqs_detail):
            df_es_detail = pd.DataFrame(samples)[['LE', 'gammaE']]
            df_es_detail['loglik'] = cond_loglik
            df_es_detail.to_csv(path_res_is + 'LogLik_EventSpecific_eqid' + 
                str(eqid) + '_EventSpecific_T' + str_per + '.csv', index=False)

        # Compute Log-posterior predictive (LPPD)
        lppds.append(-np.log(num_samples) + logsumexp(jax.device_get(cond_loglik)))

    df_es['E'] = lppds
    df_es.to_csv(path_res_is + 'LPPD_EventSpecific_EventSpecific_T' + str_per + '.csv', index=False)

# -------------------
# Pooled models
# -------------------

# Choose events which are analyzed in more detail
# Hector mine
eqs_detail = [158]
m_strings = ['E', 'EA', 'EAS']
if load:
    # In-Sample LPPD on the pooled training set for all periods
    # Used in Figure 7
    df_lppd_pool_train_allperiods = pd.read_csv(path_res_is + 
        'LPPD_Pooled_Pooled_Ttot.csv')
    period = 1 # Sa(T) where T is the period
    str_per = str(int(period*100))
    # Posterior parameter samples for Sa(T=1s)
    # Used in Table 2 and Figure 4
    dfs_param = [pd.read_csv(path_res_param + 'PostParam_Pooled_T' + 
        str_per + '_' + m_str + '.csv') for m_str in m_strings]
    # In-Sample LPPD for each event in the training set for Sa(T=1s)
    # Used in Figure 6
    df_es_pooled = pd.read_csv(path_res_is + 
        'LPPD_EventSpecific_Pooled_T' + str_per + '.csv')
    # In-Sample Conditional LogLik of pooled models for Hector Mine
    # Used in Figure 5
    df_loglik_detail = pd.read_csv(path_res_is + 
        'LogLik_EventSpecific_eqid' + str(eqs_detail[0]) + 
        '_Pooled_T' + str_per + '.csv')
    # Out-of-Sample LPPD for the pooled test set 1
    # Used in Table 3
    df_lppd_pool_test_set1 = pd.read_csv(path_res_oos + 
        'LPPD_Pooled_set1_Pooled.csv')
    # Out-of-Sample Conditional LogLik for each event in test set 2
    # Used in Figure 8
    mags = [5.4, 6.4, 7.1] # Magnitudes
    dfs_loglik_es_test_set2 = [pd.read_csv(path_res_oos + 
        'loglik_EventSpecific_set2_M' + str(int(mag*10)) + 
        '_Pooled.csv') for mag in mags]

else:

    from models_jax import modelE, modelEA, modelEAS
    models = [modelE, modelEA, modelEAS]
    vars = {
        'E': ["LE", "gammaE"],
        'EA': ["LE", "gammaE", "LA"],
        'EAS': ["LE", "gammaE", "LA", "LS", "w"]
    }

    # This list will store the lppd on the pooled training set
    # for all periods and all models (used for Figure 7!)
    list_lppds_pool_train = []

    # Set the seed to generate separate seeds in each run
    rng = np.random.default_rng(31)

    # Loop over periods
    for period in [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 6.0]:
        str_per = str(int(period*100))

        # import training data for that period
        df = pd.read_csv(path_data + 'data_ngawest2_T' + str_per + '.csv')
        # Assemble data per event
        grouped = df.groupby('eqid')
        eqids = grouped.size().keys()
        Xtot = []; ztot = [] 
        for eqid in eqids:
            group = df[df.eqid==eqid].copy()
            Xtot.append(group[['epi_dist', 'epi_azimuth', 'vs30']].values)
            ztot.append(group['scaled_deltaW'].values)

        dat_list_train = dict(X=Xtot, eqids=eqids, z=ztot)

        # import test data sets (Ridgecrest)
        # Only T=1s
        if (str_per=='100') & (test_data is not None): 
            df_test_set_1 = pd.read_csv(path_test_data + 'data_ridgecrest_set1.csv')
            df_test_set_2 = pd.read_csv(path_test_data + 'data_ridgecrest_set2.csv')
            Xtest = []; ztest = []; eqidstest = []
            for dft in [df_test_set_1, df_test_set_2]:
                eqids_num = dft.groupby('eqid').size().keys()
                Xt = []; zt = []
                for eqid in eqids_num:
                    group = dft[dft.eqid==eqid].copy()
                    Xt.append(group[['epi_dist', 'epi_azimuth', 'vs30']].values)
                    zt.append(group['scaled_deltaW'].values)
                Xtest.append(Xt); ztest.append(zt)
                eqidstest.append(eqids_num)
            lppds_pool_test_set1 = dict()
            dfs_loglik_set2 = []
            dat_list_test_set1 = dict(X=Xtest[0], eqids=eqidstest[0], z=ztest[0])
            dat_list_test_set2 = dict(X=Xtest[1], eqids=eqidstest[1], z=ztest[1])
                    


        

        if str_per == '100':
            # Initialize DataFrame for LPPDs per event (Figure 6)
            df_es_pool = pd.DataFrame()
            df_es_pool['eqid'] = grouped.size().keys()
            df_es_pool['num_recs'] = grouped.size().values
            df_es_pool['magnitude'] = [group.magnitude.values[0] 
                for _,group in grouped]
            # Initialize DataFrame for event with more detail (Figure 5)
            if len(eqs_detail) > 0:
                dfs_loglik_detail = [pd.DataFrame() for i in range(len(eqs_detail))]

        # Initialize dict for the LPPD on the pooled training set (in-sample)
        lppds_pool_train = {'period': period}
        # Sample one seed per model
        seeds = rng.integers(0, 100000, 3)

        # Loop over models
        for m, seed, m_str in zip(models, seeds, m_strings):

            # Specify model and MCMC parameters
            model = MCMC(NUTS(m), num_warmup=num_warmup, 
                num_samples=num_samples, num_chains=num_chains)
            # Run MCMC
            model.run(jax.random.PRNGKey(seed), **dat_list_train)
            # Get sampled parameter sets
            samples = model.get_samples()

            # Store sampled parameter sets
            df_param = pd.DataFrame(samples)
            df_param = df_param[vars[m_str]].copy()
            df_param.to_csv(path_res_param + 'PostParam_Pooled_T' + 
                str_per + '_' + m_str + '.csv', index=False)

            # Compute in-sample predictive performance
            df_loglik = pd.DataFrame(log_likelihood(m, samples, **dat_list_train))
            # Compute the pooled log-liked for each sample
            loglik_pool = np.sum(df_loglik.values, axis=1)
            # Compute the LPPD for the pooled training data
            lppds_pool_train[m_str] = (-np.log(num_samples) + logsumexp(loglik_pool))
            # For Sa(T=1s) compute the LPPDs for each event in the training data    
            if str_per == '100':
                df_es_pool[m_str] = -np.log(num_samples) + logsumexp(df_loglik.values,
                     axis=0)
                for i, eq_det in enumerate(eqs_detail):
                    dfs_loglik_detail[i][m_str] = df_loglik['z_'+str(eq_det)].values
            
                # Compute ouf-of-sample predictive performance
                if (test_data is not None):
                    # Test set 1: LPPD on the pooled test set
                    df_loglik = pd.DataFrame(log_likelihood(m, samples, 
                        **dat_list_test_set1))
                    loglik_pool = np.sum(df_loglik.values, axis=1)
                    lppds_pool_test_set1[m_str] = (-np.log(num_samples) 
                        + logsumexp(loglik_pool))
                    # Test set 2: Conditional loglik for each event
                    dfs_loglik_set2.append(pd.DataFrame(log_likelihood(m, 
                        samples, **dat_list_test_set2)))

        # Evaluate LPPD of independent model
        # Pooled training data
        zvals = np.hstack(dat_list_train['z'])
        lppds_pool_train['ind'] = -0.5 * (len(zvals)*np.log(2*np.pi) 
            + np.sum(zvals**2))
        list_lppds_pool_train.append(pd.Series(lppds_pool_train))
        # Pooled test set 1
        zvals = np.hstack(dat_list_test_set1['z'])
        lppds_pool_test_set1['ind'] = -0.5 * (len(zvals)*np.log(2*np.pi) 
            + np.sum(zvals**2))
        # Event-specific test set 2
        lppd_ind_test_set2 = []
        for zvals in dat_list_test_set2['z']:
            lppd_ind_test_set2.append(-0.5 * (len(zvals)*np.log(2*np.pi) 
                + np.sum(zvals**2)))

        # Store results
        if str_per == '100':
            df_es_pool.to_csv(path_res_is + 'LPPD_EventSpecific_Pooled_T' + 
                str_per + '.csv', index=False)
            for df_loglik_detail, eqid in zip(dfs_loglik_detail, eqs_detail):
                df_loglik_detail.to_csv(path_res_is + 'LogLik_EventSpecific_eqid' + 
                    str(eqid) + '_Pooled_T' + str_per + '.csv', index=False)
            if test_data is not None:
                # Set 1: 
                pd.DataFrame(lppds_pool_test_set1, index=[0]).to_csv(
                    path_res_oos + 'LPPD_Pooled_set1_Pooled.csv', index=False)
                # Set 2: Rearrange data frames per event
                for k, eqid in enumerate(dat_list_test_set2['eqids']):
                    mag = df_test_set_2[df_test_set_2.eqid==eqid].magnitude.values[0]
                    dft = pd.DataFrame()
                    for i, m_str in enumerate(m_strings):
                        dft[m_str] = dfs_loglik_set2[i]['z_' + str(eqid)]
                    dft['ind'] = lppd_ind_test_set2[k]
                    dft.to_csv(path_res_oos + 'loglik_EventSpecific_set2_M' 
                        + str(int(mag*10)) + '_Pooled.csv', index=False)


    df_lppd_pool_train_allperiods = pd.concat(list_lppds_pool_train, axis=1).T
    df_lppd_pool_train_allperiods.to_csv(path_res_is + 
        'LPPD_Pooled_Pooled_Ttot.csv', index=False)

