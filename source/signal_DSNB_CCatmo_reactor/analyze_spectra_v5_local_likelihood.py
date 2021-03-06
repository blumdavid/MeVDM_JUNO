""" Script to statistically analyze a large number of dataset-spectra with the simulation-spectrum on the local PC!!

    Version 5:  MCMC sampling is implemented differently to version4:
                - burnin-phase is implemented differently -> first burnin-phase, than reset of the sampler, then new
                sampling with the positions of the walkers from the end of burnin-phase

                Advantage: burnin-phase and actual sampling can be treated separately

                New Implementation of the estimation of the autocorrelation time:
                - calculated with sampler.get_autocorr_time
                - implementation of try / except to avoid a crash when there is an AutocorrError

                !!NO Calculation of the conditional probability for the hypothesis H to be true or not!!

                The fitting of the datasets to the model (simulated spectrum) is based on the python package emcee:
                -  emcee is an MIT licensed pure-Python implementation of Goodman & Weare’s Affine Invariant
                Markov chain Monte Carlo (MCMC) Ensemble sampler
                - it's designed for Bayesian parameter estimation (marginalization of the posterior probability)
                - based on Markov Chain Monte Carlo (MCMC) sampling
                - more information about the algorithm in the paper emcee_1202.3665.pdf
                - the code here is based on the example described in the link http://dfm.io/emcee/current/user/line/
                - on the homepage http://dfm.io/emcee/current/ there are lots of information and also the documentation

    !!!!!
    DIFFERENCE TO analyze_spectra_v5_local.py:
        -   in analyze_spectra_v5_local.py the logarithm of the likelihood (ln_likelihood()) and the logarithm of the
            posterior probability (ln_posteriorprob()) is used

        -   HERE: NOT the logarithm is used, BUT the standard likelihood function

    MCMC sampling does NOT work -> likelihood becomes to small (-> 0), even if it is not 0. (float64 has to few digits)
    !!!!!

    Dataset (virtual experiments) are generated with gen_dataset_v1_local.py
    Simulated spectra are generated with gen_spectrum_v2.py

"""

# import of the necessary packages:
import datetime
import time
import numpy as np
from scipy.special import factorial
from scipy.special import erf
import emcee
import corner
import scipy.optimize as op
from matplotlib import pyplot as plt


# TODO: i have to cite the package 'emcee', when I use it to analyze

# TODO-me: Check the sensitivity of the results depending on the prior probabilities

# TODO-me: check the MCMC sampling again, corresponding to the information in emcee_1202.3665.pdf (acceptance fraction,
# TODO-me: auto-correlation time)


""" Set boolean value to define, if the result of the analysis are saved: """
# save the results and the plot of likelihood-function with fit parameters
# (if data should be saved, SAVE_DATA must be True):
SAVE_DATA = True

""" set the path to the correct folder: """
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor"

""" set the path of the output folder: """
path_output = path_folder + "/dataset_output_25"

""" set the path of the folder, where the datasets were saved: """
path_dataset = path_output + "/datasets"

""" set the path of the folder, where the results of the analysis should be saved: """
path_analysis = path_output + "/likelihood_test"

""" Go through every dataset and perform the analysis: """
# define the first dataset, which will be analyzed (file: Dataset_dataset_start) (integer):
dataset_start = 1
# define the last dataset, which will be analyzed (file: Dataset_dataset_stop) (integer):
dataset_stop = 1

""" Load information about the generation of the datasets from file (np.array of float): """
# TODO: Check, if info-files have the same parameter:
info_dataset = np.loadtxt(path_dataset + "/info_dataset_1_to_10000.txt")
# get the bin-width of the visible energy in MeV from the info-file (float):
interval_E_visible = info_dataset[0]
# get minimum of the visible energy in MeV from info-file (float):
min_E_visible = info_dataset[1]
# get maximum of the visible energy in MeV from info-file (float):
max_E_visible = info_dataset[2]

""" Load simulated spectra in events/MeV from file (np.array of float): """
path_simu = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2"
file_signal = path_simu + "/signal_DMmass25_bin100keV.txt"
Spectrum_signal = np.loadtxt(file_signal)
file_info_signal = path_simu + "/signal_info_DMmass25_bin100keV.txt"
info_signal = np.loadtxt(file_info_signal)
file_DSNB = path_simu + "/DSNB_EmeanNuXbar22_bin100keV.txt"
Spectrum_DSNB = np.loadtxt(file_DSNB)
file_reactor = path_simu + "/Reactor_NH_power36_bin100keV.txt"
Spectrum_reactor = np.loadtxt(file_reactor)
file_CCatmo = path_simu + "/CCatmo_Osc1_bin100keV.txt"
Spectrum_CCatmo = np.loadtxt(file_CCatmo)

""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

""" Get Dark Matter mass from the info_signal file: """
DM_mass = info_signal[9]

""" Define the energy window, where spectrum of virtual experiment and simulated spectrum is analyzed
    (from min_E_cut in MeV to max_E_cut in MeV): """
# INFO-me: when you use the whole energy window (from 10 to 100 MeV), the MCMC sampling works better, because then the
# INFO-me: maxlikeli values are NOT around 0
# TODO-me: also check the analysis for a smaller energy window
# min_E_cut = DM_mass - 5
# max_E_cut = DM_mass + 5
min_E_cut = min_E_visible
max_E_cut = max_E_visible
# calculate the entry number of the array to define the energy window:
entry_min_E_cut = int((min_E_cut - min_E_visible) / interval_E_visible)
entry_max_E_cut = int((max_E_cut - min_E_visible) / interval_E_visible)

""" Simulated spectra in events/bin (multiply with interval_E_visible): """
# spectrum per bin in the 'interesting' energy range from min_E_cut to max_E_cut
# (you have to take (entry_max+1) to get the array, that includes max_E_cut):
spectrum_Signal_per_bin = Spectrum_signal[entry_min_E_cut: (entry_max_E_cut + 1)] * interval_E_visible
spectrum_DSNB_per_bin = Spectrum_DSNB[entry_min_E_cut: (entry_max_E_cut + 1)] * interval_E_visible
spectrum_CCatmo_per_bin = Spectrum_CCatmo[entry_min_E_cut: (entry_max_E_cut + 1)] * interval_E_visible
spectrum_Reactor_per_bin = Spectrum_reactor[entry_min_E_cut: (entry_max_E_cut + 1)] * interval_E_visible

""" 'true' values of the parameters: """
# expected number of signal events in the energy window (float):
S_true = np.sum(spectrum_Signal_per_bin)
# maximum value of signal events consistent with existing limits (assuming the 90 % upper limit for the annihilation
# cross-section of Super-K from paper 0710.5420, for the description and calculation see limit_from_SuperK.py)
# INFO-me: S_max is assumed from the limit on the annihilation cross-section of Super-K (see limit_from_SuperK.py)
S_max = 24
# expected number of DSNB background events in the energy window (float):
B_DSNB_true = np.sum(spectrum_DSNB_per_bin)
# expected number of CCatmo background events in the energy window (float):
B_CCatmo_true = np.sum(spectrum_CCatmo_per_bin)
# expected number of Reactor background events in the energy window (float):
B_Reactor_true = np.sum(spectrum_Reactor_per_bin)

""" fractions (normalized shapes) of signal and background spectra: """
# Fraction of DM signal (np.array of float):
fraction_Signal = spectrum_Signal_per_bin / S_true
# Fraction of DSNB background (np.array of float):
fraction_DSNB = spectrum_DSNB_per_bin / B_DSNB_true
# Fraction of CCatmo background (np.array of float):
fraction_CCatmo = spectrum_CCatmo_per_bin / B_CCatmo_true
# Fraction of reactor background (np.array of float):
fraction_Reactor = spectrum_Reactor_per_bin / B_Reactor_true

""" Preallocate the array, where the acceptance fraction of the actual sampling of each analysis is appended to 
(empty np.array): """
af_sample_mean_array = np.array([])
""" Preallocate the array, where the acceptance fraction during burnin-phase is appended to (empty np.array): """
af_burnin_mean_array = np.array([])
""" Preallocate the array, where the mean of autocorrelation time is appended to (empty np.array): """
mean_acor_array = np.array([])


""" Define functions: """
# TODO-me: check likelihood(), priorprob() and posteriorprob(), if the values can be calculated
# (if they are large enough)!!


def neg_ln_likelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    """
    Function, which represents the negative log-likelihood function.
    The function is defined by the negative of natural logarithm of the function p_spectrum_sb(), that is defined
    on page 3, equation 11 in the GERDA paper.

    The likelihood function is given in the GERDA paper (equation 11: p_spectrum_sb) and in the paper arXiv:1208.0834
    'A novel way of constraining WIMPs annihilation in the Sun: MeV neutrinos', page 15, equation (24)

    :param param: np.array of the 'unknown' parameters of the log-likelihood function (param represents the number
    of signal events, the number of DSNB background events, the number of atmospheric CC background events,
    and the number of reactor background events (np.array of 4 float)

    :param data: represents the data, that the model will be fitted to ('observed' number of events for each bin
    from the dataset ('observed' spectrum)) (np.array of float)

    :param fraction_signal: normalized shapes of the signal spectra (represents f_S in equ. 9 of the GERDA paper),
    equivalent to the number of signal events per bin from the theoretical spectrum (np.array of float)

    :param fraction_dsnb: normalized shapes of the DSNB background spectra (represents f_B_DSNB in equ. 9 of the
    GERDA paper), equivalent to the number of DSNB background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ccatmo: normalized shapes of the CC atmo. background spectra (represents f_B_CCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of CC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_reactor: normalized shapes of the reactor background spectra (represents f_B_reactor in equ. 9 of
    the GERDA paper), equivalent to the number of reactor background events per bin from the theoretical spectrum
    (np.array of float)

    :return: the value of the log-likelihood function (float) (alternatively: returns the log-likelihood function
    as function of the unknown parameters param)
    """
    # get the single parameters from param (float):
    s, b_dsnb, b_ccatmo, b_reactor = param
    # calculate the variable lambda_i(s, b_dsnb, b_ccatmo, b_reactor), which is defined on page 3, equ. 9 of the
    # GERDA paper 'A Bayesian approach to the analysis of sparsely populated spectra, Calculating the sensitivity of
    # the GERDA experiment' (np.array of float):
    lamb = fraction_signal*s + fraction_dsnb*b_dsnb + fraction_ccatmo*b_ccatmo + fraction_reactor*b_reactor

    # calculate the addends (Summanden) of the log-likelihood-function defined on page 3, equ. 11 of the GERDA paper
    # (np.array of float):
    sum_1 = data*np.log(lamb)

    """ From documentation scipy 0.14: scipy.special.factorial: (on the server scipy 0.19 is installed, 
        on pipc79 scipy 1.0.0 is installed)
        scipy.special.factorial is the factorial function, n! = special.gamma(n+1).
        If exact is 0, then floating point precision is used, otherwise exact long integer is computed.
        Array argument accepted only for exact=False case. If n<0, the return value is 0.
        """
    # INFO-me: until factorial(14) there is no difference between the result for exact=True and exact=False!!
    # INFO-me: from fac(15) to fac(19) the deviation is less than 3.4e-7% !!
    # INFO-me: exact=True is only valid up to factorial(20), for larger values the int64 is to 'small'
    sum_2 = np.log(factorial(data, exact=False))
    sum_3 = lamb

    return -np.sum(sum_1 - sum_2 - sum_3)


def likelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    """
    Function, which represents the likelihood function.
    The function is defined by the function p_spectrum_sb(), that is defined on page 3, equ. 11 in the GERDA paper.

    The likelihood function is given in the GERDA paper (equation 11: p_spectrum_sb) and in the paper arXiv:1208.0834
    'A novel way of constraining WIMPs annihilation in the Sun: MeV neutrinos', page 15, equation (24)

    :param param: np.array of the 'unknown' parameters of the likelihood function (param represents the number
    of signal events, the number of DSNB background events, the number of atmospheric CC background events,
    and the number of reactor background events (np.array of 4 float)

    :param data: represents the data, that the model will be fitted to ('observed' number of events for each bin
    from the dataset ('observed' spectrum)) (np.array of float)

    :param fraction_signal: normalized shapes of the signal spectra (represents f_S in equ. 9 of the GERDA paper),
    equivalent to the number of signal events per bin from the theoretical spectrum (np.array of float)

    :param fraction_dsnb: normalized shapes of the DSNB background spectra (represents f_B_DSNB in equ. 9 of the
    GERDA paper), equivalent to the number of DSNB background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ccatmo: normalized shapes of the CC atmo. background spectra (represents f_B_CCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of CC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_reactor: normalized shapes of the reactor background spectra (represents f_B_reactor in equ. 9 of
    the GERDA paper), equivalent to the number of reactor background events per bin from the theoretical spectrum
    (np.array of float)

    :return: the value of the likelihood function (float) (alternatively: returns the likelihood function
    as function of the unknown parameters param)
    """
    # get the single parameters from param (float):
    s, b_dsnb, b_ccatmo, b_reactor = param
    # calculate the variable lambda_i(s, b_dsnb, b_ccatmo, b_reactor), which is defined on page 3, equ. 9 of the
    # GERDA paper 'A Bayesian approach to the analysis of sparsely populated spectra, Calculating the sensitivity of
    # the GERDA experiment' (np.array of float):
    lamb = fraction_signal*s + fraction_dsnb*b_dsnb + fraction_ccatmo*b_ccatmo + fraction_reactor*b_reactor

    # calculate the factors of the likelihood function defined on page 3, equ. 11 of the GERDA paper
    # (np.array of float):
    factor_1 = lamb**data

    """ From documentation scipy 0.14: scipy.special.factorial: (on the server scipy 0.19 is installed, 
        on pipc79 scipy 1.0.0 is installed)
        scipy.special.factorial is the factorial function, n! = special.gamma(n+1).
        If exact is 0, then floating point precision is used, otherwise exact long integer is computed.
        Array argument accepted only for exact=False case. If n<0, the return value is 0.
        """
    # INFO-me: until factorial(14) there is no difference between the result for exact=True and exact=False!!
    # INFO-me: from fac(15) to fac(19) the deviation is less than 3.4e-7% !!
    # INFO-me: exact=True is only valid up to factorial(20), for larger values the int64 is to 'small'
    factor_2 = factorial(data, exact=False)

    factor_3 = np.exp(-lamb)

    # calculate the value of the likelihood with the factors (float):
    value_likelihood = np.prod(factor_1 / factor_2 * factor_3)

    return value_likelihood


def priorprob(param):
    """
    Function to calculate the prior probabilities of the parameter param.

    The prior probabilities are partly considered from the GERDA paper, page 6, equ. 20 and 21.

    :param param: np.array of the 'unknown' parameters of the prior probability (param represents the number
    of signal events, the number of DSNB background events, the number of atmospheric CC background events,
    and the number of reactor background events (np.array of 4 float)

    :return: the product of the prior probabilities of the different parameters param
    """
    # get the single parameters from param (float):
    s, b_dsnb, b_ccatmo, b_reactor = param

    # define the prior probability for the expected signal contribution (flat_probability until maximum value S_max):
    if 0 <= s <= S_max:
        # between 0 and S_max, flat probability defined by 1/S_max (float):
        prior_s = 1/S_max
    else:
        # if s < 0 or s > S_max, the prior probability p_0_s is set to 0 (float):
        prior_s = 0

    # define the prior probability for the expected DSNB background contribution (the background
    # contribution is assumed to be 'very poorly known' -> width = 2*B_DSNB_true (poorly known background corresponds
    # to a width = B_DSNB_true, fairly known background corresponds to a width = B_DSNB_true/2).
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_DSNB_true and
    # width = 2*B_DSNB_true):
    # INFO-me: DSNB background is assumed to be Gaussian with width = 2*B_DSNB_true -> "very poorly known background"
    if b_dsnb >= 0:
        # define the mean and the width of the Gaussian function (float):
        mu_b_dsnb = B_DSNB_true
        sigma_b_dsnb = B_DSNB_true * 2

        # calculate the denominator of equ. 21 (integral over B from 0 to infinity of exp(-(B-mu)**2 / (2*sigma**2)),
        # the integral is given by the error function) (float):
        denominator_dsnb = np.sqrt(np.pi/2) * (sigma_b_dsnb * erf(mu_b_dsnb / (np.sqrt(2)*sigma_b_dsnb)) + sigma_b_dsnb)

        # calculate the prior probability (float):
        prior_b_dsnb = 1/denominator_dsnb * np.exp(-(b_dsnb - mu_b_dsnb)**2 / (2 * sigma_b_dsnb**2))

    else:
        # if b_dsnb < 0, the prior probability is set to 0 (float):
        prior_b_dsnb = 0

    # define the prior probability for the expected atmospheric CC background contribution (the background
    # contribution is assumed to be known up to a factor of 2 -> 'fairly known background'.
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_CCatmo_true and
    # width = B_CCatmo_true/2)
    if b_ccatmo >= 0:
        mu_b_ccatmo = B_CCatmo_true
        sigma_b_ccatmo = B_CCatmo_true / 2

        # calculate the denominator of equ. 21 (integral over B from 0 to infinity of exp(-(B-mu)**2 / (2*sigma**2)),
        # the integral is given by the error function) (float):
        denominator_ccatmo = np.sqrt(np.pi/2) * (sigma_b_ccatmo * erf(mu_b_ccatmo / (np.sqrt(2)*sigma_b_ccatmo))
                                                 + sigma_b_ccatmo)

        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):

        # calculate the prior probability (float):
        prior_b_ccatmo = 1/denominator_ccatmo * np.exp(-(b_ccatmo - mu_b_ccatmo)**2 / (2 * sigma_b_ccatmo**2))

    else:
        # if b_ccatmo < 0, the prior probability is set to 0 (float):
        prior_b_ccatmo = 0

    # define the prior probability for the expected reactor background contribution (the background
    # contribution is assumed to be known up to a factor of 2 -> 'fairly known background'.
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_Reactor_true and
    # width = B_events_reactor/2)
    if b_reactor >= 0:
        mu_b_reactor = B_Reactor_true
        sigma_b_reactor = B_Reactor_true / 2

        # calculate the denominator of equ. 21 (integral over B from 0 to infinity of exp(-(B-mu)**2 / (2*sigma**2)),
        # the integral is given by the error function) (float):
        denominator_reactor = np.sqrt(np.pi/2) * (sigma_b_reactor * erf(mu_b_reactor / (np.sqrt(2)*sigma_b_reactor))
                                                  + sigma_b_reactor)

        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):

        # calculate the prior probability (float):
        prior_b_reactor = 1/denominator_reactor * np.exp(-(b_reactor - mu_b_reactor)**2 / (2 * sigma_b_reactor**2))

    else:
        # if b_ccatmo < 0, the prior probability is set to 0 (float):
        prior_b_reactor = 0

    test = prior_s * prior_b_dsnb * prior_b_ccatmo * prior_b_reactor

    return prior_s * prior_b_dsnb * prior_b_ccatmo * prior_b_reactor


def posteriorprob(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    """
    Function, which represents the full posterior probability of the Bayesian statistics.

    The function is defined by the nominator of the function p_SB_spectrum(), that is defined on page 4,
    equation 12 in the GERDA paper.

    So the function is proportional to the product of the likelihood and the prior probabilities.

    IMPORTANT:  the denominator of equ. 12 (integral over p_spectrum_SB * p_0_S * p_0_b) is not considered
                -> it is just a normalization constant

    :param param: np.array of the 'unknown' parameters of the log-likelihood function (param represents the number
    of signal events, the number of DSNB background events, the number of atmospheric CC background events,
    and the number of reactor background events (np.array of 4 float)

    :param data: represents the data, that the model will be fitted to ('observed' number of events for each bin
    from the dataset ('observed' spectrum)) (np.array of float)

    :param fraction_signal: normalized shapes of the signal spectra (represents f_S in equ. 9 of the GERDA paper),
    equivalent to the number of signal events per bin from the theoretical spectrum (np.array of float)

    :param fraction_dsnb: normalized shapes of the DSNB background spectra (represents f_B_DSNB in equ. 9 of the
    GERDA paper), equivalent to the number of DSNB background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ccatmo: normalized shapes of the CC atmo. background spectra (represents f_B_CCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of CC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_reactor: normalized shapes of the reactor background spectra (represents f_B_reactor in equ. 9 of
    the GERDA paper), equivalent to the number of reactor background events per bin from the theoretical spectrum
    (np.array of float)

    :return: the value of the posterior probability function (full Bayesian probability) (float)
    (alternatively: returns the posterior probability as function of the unknown parameters param)
    """
    # calculate the prior probabilities from the parameter (float):
    prior = priorprob(param)

    # check if prior greater than zero. If not, return 0 as full probability:
    if prior < 0:
        return 0

    return prior * likelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor)


# loop over the Datasets (from dataset_start to dataset_stop):
for number in np.arange(dataset_start, dataset_stop+1, 1):
    print("Analyze dataset_{0:d}".format(number))
    # load corresponding dataset (unit: events/bin) (np.array of float):
    Data = np.loadtxt(path_dataset + "/Dataset_{0:d}.txt".format(number))
    # dataset in the 'interesting' energy range from min_E_cut to max_E_cut
    # (you have to take (entry_max+1) to get the array, that includes max_E_cut):
    Data = Data[entry_min_E_cut: (entry_max_E_cut + 1)]

    time_minimize_0 = time.time()
    # guess of the parameters (as guess, the total number of events from the simulated spectrum are used)
    # (np.array of float):
    parameter_guess = np.array([S_true, B_DSNB_true, B_CCatmo_true, B_Reactor_true])

    # bounds of the parameters (parameters have to be positive or zero) (tuple):
    bnds = ((0, None), (0, None), (0, None), (0, None))

    # Minimize the negative log-likelihood function with the L-BFGS-B method for the defined bounds above:
    result = op.minimize(neg_ln_likelihood, parameter_guess,
                         args=(Data, fraction_Signal, fraction_DSNB, fraction_CCatmo, fraction_Reactor),
                         method='L-BFGS-B', bounds=bnds, options={'disp': None})

    # get the best-fit parameters from the minimization (float):
    S_maxlikeli, B_dsnb_maxlikeli, B_ccatmo_maxlikeli, B_reactor_maxlikeli = result["x"]
    time_minimize_1 = time.time()
    print("time in seconds for minimization = {0}".format(time_minimize_1-time_minimize_0))

    time_sample_0 = time.time()
    """ Sample this distribution using emcee. 
        Start by initializing the walkers in a tiny Gaussian ball around the maximum likelihood result (in the example 
        on the emcee homepage they found that this tends to be a pretty good initialization ni most cases): 
        Walkers are the members of the ensemble. They are almost like separate Metropolis-Hastings chains but, of 
        course, the proposal distribution for a given walker depends on the positions of all the other walkers in 
        the ensemble. See mcmc_GoodmanWeare_2010.pdf for more details.
        Run with large number of walkers -> more independent samples per autocorrelation time. BUT: disadvantage of 
        large number of walkers is that the burnin-phase can be slow. Therefore: use the smallest number of walkers 
        for which the acceptance fraction during burn-in is good (see emcee_1202.3665.pdf): """
    # INFO-me: nwalkers=200 might be ok
    ndim, nwalkers = 4, 50
    P0 = [result["x"] + 10**(-4)*np.random.randn(ndim) for i in range(nwalkers)]

    """ Then, we can set up the sampler 
        EnsembleSampler: a generalized Ensemble sampler that uses 2 ensembles for parallelization.
        (The "a" parameter controls the step size, the default is a=2): """
    value_of_a = 3.0
    sampler = emcee.EnsembleSampler(nwalkers, ndim, posteriorprob, a=value_of_a,
                                    args=(Data, fraction_Signal, fraction_DSNB, fraction_CCatmo, fraction_Reactor))

    """ Set up the burnin-phase:
        the burnin-phase is initial sampling phase from the initial conditions (tiny Gaussian
        ball around the maximum likelihood result) to reasonable sampling. """
    # INFO-me: step_burnin should be 'correct' value
    # set number of steps, which are used for "burning in" (the first 'step_burnin' steps are not considered in the
    # sample) (integer):
    step_burnin = 500
    # run the MCMC for 'step_burnin' steps starting from the tiny ball defined above (run_mcmc iterates sample() for
    # N iterations and returns the result of the final sample):
    pos, prob, state = sampler.run_mcmc(P0, step_burnin)

    """ Calculate the mean acceptance fraction during burnin phase: """
    # INFO-me: mean of acceptance fraction should be roughly between 0.2 and 0.5
    # get acceptance fraction (1D np.array of float, dimension = nwalkers):
    af_burnin = sampler.acceptance_fraction
    # calculate the mean of the acceptance fraction (float):
    af_burnin_mean = np.mean(af_burnin)
    # append af_mean to the array, which will be saved to txt-file (np.array of float):
    af_burnin_mean_array = np.append(af_burnin_mean_array, af_burnin_mean)
    print("mean acceptance fraction during burnin-phase = {0}".format(af_burnin_mean))

    """ Reset the sampler (to get rid of the previous chain and start where the sampler left off at variable 'pos'):"""
    sampler.reset()

    """ Now run the MCMC for 'number_of_steps' steps starting, where the sampler left off in the burnin-phase: 
        (run_mcmc iterates sample() for N iterations and returns the result of the final sample) """
    # TODO-me: the number of steps should be large (greater than around 10000) to get a reproducible result
    number_of_steps = 15000
    sampler.run_mcmc(pos, number_of_steps)

    time_sample_1 = time.time()
    print("time for sampling = {0}".format(time_sample_1 - time_sample_0))

    """ The best way to see this is to look at the time series of the parameters in the chain. 
    The sampler object now has an attribute called chain that is an array with the shape 
    (nwalkers, number_of_steps, ndim) giving the parameter values for each walker at each step in the chain. 
    The figure below shows the positions of each walker as a function of the number of steps in the chain 
    (sampler.chain[:, :, 0] has shape (nwalkers, number_of_steps), so the steps as function of the walker, 
    sampler.chain[:, :, 0].T is the transpose of the array with shape (number_of_steps, nwalkers), so the walker as 
    function of steps): """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ndim, 1, sharex='all')
    fig.set_size_inches(10, 10)
    ax1.plot(sampler.chain[:, :, 0].T, '-', color='k', alpha=0.3)
    ax1.axhline(y=S_true, color='b', linestyle='--')
    ax1.set_ylabel('$S$')
    ax2.plot(sampler.chain[:, :, 1].T, '-', color='k', alpha=0.3)
    ax2.axhline(y=B_DSNB_true, color='b', linestyle='--')
    ax2.set_ylabel('$B_{DSNB}$')
    ax3.plot(sampler.chain[:, :, 2].T, '-', color='k', alpha=0.3)
    ax3.axhline(y=B_CCatmo_true, color='b', linestyle='--')
    ax3.set_ylabel('$B_{CCatmo}$')
    ax4.plot(sampler.chain[:, :, 3].T, '-', color='k', alpha=0.3)
    ax4.axhline(y=B_Reactor_true, color='b', linestyle='--', label='expected number of events')
    ax4.set_ylabel('$B_{Reactor}$')
    ax4.set_xlabel('step number')
    plt.legend()
    if SAVE_DATA:
        fig.savefig(path_analysis + '/Dataset{0}_chain_traces.png'.format(number))
    plt.close(fig)

    """ Calculate the mean acceptance fraction. The acceptance fraction is the ratio between the accepted steps 
        over the total number of steps (Fraction of proposed steps that are accepted). In general, acceptance_fraction 
        has an entry for each walker (So it is a nwalkers-dimensional vector, therefore calculate the mean). 
        (See: https://gist.github.com/banados/2254240 and 
        http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay8_MCMC_with_emcee.html and emcee_1202.3665.pdf)
        -   (thumb rule: the acceptance fraction should be between 0.2 and 0.5! If af < 0.2 decrease the a parameter,
            if af > 0.5 increase the a parameter)
        -   if af -> 0, then nearly all steps are rejected. So the chain will have very few independent samples
            and the sampling will not be representative of the target density.
        -   if af -> 1, then nearly all steps are accepted and the chain is performing a random walk with no regard
            for the target density . So this will also not produce representative samples (NO effectively sampling of 
            the posterior PDF)
        """
    # INFO-me: mean of acceptance fraction should be roughly between 0.2 and 0.5
    # get acceptance fraction (1D np.array of float, dimension = nwalkers):
    af_sample = sampler.acceptance_fraction
    # calculate the mean of the acceptance fraction (float):
    af_sample_mean = np.mean(af_sample)
    # append af_mean to the array, which will be saved to txt-file (np.array of float):
    af_sample_mean_array = np.append(af_sample_mean_array, af_sample_mean)
    print("mean acceptance fraction during sampling = {0}".format(af_sample_mean))

    """ Calculate the auto-correlation time for the chain. 
        The auto-correlation time is a direct measure of the number of evaluations of the posterior PDF required to 
        produce independent samples of the target density. It is an estimate of the number of steps needed in the 
        chain in order to draw independent samples from the target density.
        """
    # TODO-me: include the auto-correlation time to estimate the performance and reliability of the MCMC
    # (The longer the auto-correlation time, the larger the number of the samples we must generate to obtain the
    # desired sampling of the posterior PDF) (You should run the sampler for a few (e.g. 10) auto-correlation times.
    # After that, you are almost completely sure to have independent samples from the posterior PDF).
    # Estimate the autocorrelation time for each dimension (np.array of float, dimension=ndim):
    try:
        # estimate integrated autocorrelation time, c = the minimum number of autocorrelation times needed to trust
        # the estimate (default: 10) (np.array of float):
        autocorr_time = sampler.get_autocorr_time(c=10)
        print("auto-correlation time = {0}".format(autocorr_time))
        # calculate the mean of autocorr_time (float):
        mean_autocorr_time = np.mean(autocorr_time)
        print("mean of autocorrelation time = {0}".format(mean_autocorr_time))
    except emcee.autocorr.AutocorrError:
        # if there is an emcee.autocorr.AutocorrError, set the autocorrelation time to 1001001 (-> this means that there
        # was an error):
        mean_autocorr_time = 1001001
        print("emcee.autocorr.AutocorrError")

    # append mean_autocorr_time to the array (np.array of float):
    mean_acor_array = np.append(mean_acor_array, mean_autocorr_time)

    """ flatten the chain so that we have a flat list of samples: """
    # the chain is a three-dimensional array of shape (number walkers nwalkers, number of steps, dimensions ndim),
    # e.g (200, 3000, 4)).
    # The 'flatchain' function flattens the chain along the zeroth (nwalkers) axis and you get a two dimensional
    # array of shape (nwalkers*steps, ndim), so e.g. (200*3000, 4) = (600000, 4)):
    samples = sampler.flatchain

    time_mode_0 = time.time()
    """ Calculate the mode and the 90% upper limit of the signal_sample distribution: """
    # get the sample-chain of the signal contribution (np.array of float):
    signal_sample = samples[:, 0]
    # put signal_sample in a histogram (2 np.arrays of float), hist are the values of the histogram,
    # bins_edges return the bin edges (length(hist)+1):
    hist_S, bin_edges_S = np.histogram(signal_sample, bins='auto', range=(0, signal_sample.max()))

    # use bin_edges_S to calculate the value of the middle of each bin (np.array of float):
    bin_middle_S = (bin_edges_S[0:-1] + bin_edges_S[1:]) / 2

    # calculate the bin-width, 'auto' generates bins with equal width(float):
    bin_width_S = bin_edges_S[1] - bin_edges_S[0]

    # integrate hist_S over bin_middle_S (float):
    integral_hist_S = np.sum(hist_S) * bin_width_S
    print("integral = {0}".format(integral_hist_S))

    plt.step(bin_middle_S, hist_S, 'x')
    plt.xlabel("S")
    plt.ylabel("counts")
    plt.title("p(S) from MCMC sampling for one dataset")
    plt.show()

    # get the index of the bin, where hist_S is maximal (integer):
    index_S = np.argmax(hist_S)
    # get the value of the left edge of the bin for index_S from above (float):
    value_left_edge_S = bin_edges_S[index_S]
    # get the value of the right edge of the bin for index_S from above (float):
    value_right_edge_S = bin_edges_S[index_S + 1]
    # calculate the mode of the signal_sample, therefore calculate the mean of value_left_edge and value_right_edge to
    # get the value in the middle of the bin (float):
    S_mode = (value_left_edge_S + value_right_edge_S) / 2

    # Calculate the 90 percent upper limit of the signal contribution (float)
    S_90 = np.percentile(signal_sample, 90)

    # calculate S_90 by integrating hist_S over bin_middle_S up to 0.9*integral_hist_S:
    for index_bin in np.arange(1, len(bin_middle_S), 1):
        # integral until index_bin:
        integral = np.sum(hist_S[0:index_bin]) * bin_width_S

        if integral < 0.9*integral_hist_S:
            continue
        elif integral == 0.9*integral_hist_S:
            S_90_test = bin_middle_S[index_bin]
            break
        else:
            # must be improved further!!!!!
            blabla = bin_middle_S[index_bin - 1]
            S_90_test = bin_middle_S[index_bin]
            break

    print("S_90 from old calculation = {0}".format(S_90))
    print("S_90 from new calculation (index - 1) = {0}".format(blabla))
    print("S_90 from new calculation = {0}".format(S_90_test))

    """ Calculate the mode of the DSNB_sample distribution: """
    # get the sample-chain of the DSNB background contribution (np.array of float):
    DSNB_sample = samples[:, 1]
    # put DSNB_sample in a histogram (2 np.arrays of float):
    hist_DSNB, bin_edges_DSNB = np.histogram(DSNB_sample, bins='auto', range=(0, DSNB_sample.max()))
    # get the index of the bin, where hist_DSNB is maximal (integer):
    index_DSNB = np.argmax(hist_DSNB)
    # get the value of the left edge of the bin for index_DSNB from above (float):
    value_left_edge_DSNB = bin_edges_DSNB[index_DSNB]
    # get the value of the right edge of the bin for index_DSNB from above (float):
    value_right_edge_DSNB = bin_edges_DSNB[index_DSNB+1]
    # calculate the mode of the DSNB_sample, therefore calculate the mean of value_left_edge and value_right_edge to
    # get the value in the middle of the bin (float):
    DSNB_mode = (value_left_edge_DSNB + value_right_edge_DSNB) / 2

    """ Calculate the mode of the CCatmo_sample distribution: """
    # get the sample-chain of the atmo. CC background contribution (np.array of float):
    CCatmo_sample = samples[:, 2]
    # put CCatmo_sample in a histogram (2 np.arrays of float):
    hist_CCatmo, bin_edges_CCatmo = np.histogram(CCatmo_sample, bins='auto', range=(0, CCatmo_sample.max()))
    # get the index of the bin, where hist_CCatmo is maximal (integer):
    index_CCatmo = np.argmax(hist_CCatmo)
    # get the value of the left edge of the bin for index_CCatmo from above (float):
    value_left_edge_CCatmo = bin_edges_CCatmo[index_CCatmo]
    # get the value of the right edge of the bin for index_CCatmo from above (float):
    value_right_edge_CCatmo = bin_edges_CCatmo[index_CCatmo+1]
    # calculate the mode of the CCatmo_sample, therefore calculate the mean of value_left_edge and value_right_edge to
    # get the value in the middle of the bin (float):
    CCatmo_mode = (value_left_edge_CCatmo + value_right_edge_CCatmo) / 2

    """ Calculate the mode of the Reactor_sample distribution: """
    # get the sample-chain of the reactor background contribution (np.array of float):
    Reactor_sample = samples[:, 3]
    # put Reactor_sample in a histogram (2 np.arrays of float):
    hist_Reactor, bin_edges_Reactor = np.histogram(Reactor_sample, bins='auto', range=(0, Reactor_sample.max()))
    # get the index of the bin, where hist_Reactor is maximal (integer):
    index_Reactor = np.argmax(hist_Reactor)
    # get the value of the left edge of the bin for index_Reactor from above (float):
    value_left_edge_Reactor = bin_edges_Reactor[index_Reactor]
    # get the value of the right edge of the bin for index_Reactor from above (float):
    value_right_edge_Reactor = bin_edges_Reactor[index_Reactor+1]
    # calculate the mode of the Reactor_sample, therefore calculate the mean of value_left_edge and value_right_edge to
    # get the value in the middle of the bin (float):
    Reactor_mode = (value_left_edge_Reactor + value_right_edge_Reactor) / 2

    time_mode_1 = time.time()
    print("time to calculate the modes = {0}".format(time_mode_1-time_mode_0))

    """ Now that we have this list of samples, let’s make one of the most useful plots you can make with your MCMC 
        results: a corner plot. Generate a corner plot is as simple as: """
    # NOTE: the quantile(0.5) is equal to the np.median() and is equal to np.percentile(50) -> NOT equal to np.mean()
    fig1 = corner.corner(samples, labels=["$S$", "$B_{DSNB}$", "$B_{CCatmo}$", "$B_{reactor}$"],
                         truths=[S_true, B_DSNB_true, B_CCatmo_true, B_Reactor_true], truth_color='b',
                         show_titles=True, title_fmt='.4f', labels_args={"fontsize": 40})
    # save figure:
    if SAVE_DATA:
        fig1.savefig(path_analysis + "/Dataset{0}_fitresult.png".format(number))
    plt.close(fig1)

    """ Clear the chain and lnprobability array. Also reset the bookkeeping parameters: """
    sampler.reset()

    # to save the analysis, SAVE_DATA must be True:
    if SAVE_DATA:
        # save the output of the analysis of the dataset to txt-file:
        np.savetxt(path_analysis + '/Dataset{0}_mcmc_analysis.txt'.format(number),
                   np.array([S_mode, S_90, DSNB_mode, CCatmo_mode, Reactor_mode,
                             S_maxlikeli, B_dsnb_maxlikeli, B_ccatmo_maxlikeli, B_reactor_maxlikeli]),
                   fmt='%4.5f',
                   header='Results of the MCMC analysis of virtual experiment (Dataset_{0:d}) to the expected spectrum'
                          '(analyzed with analyze_spectra_v5_local_likelihood.py, {1}):\n'
                          'General information of the analysis are saved in info_mcmc_analysis_{2:d}_{3:d}.txt\n'
                          'Results of the analysis:\n'
                          'mode of the number of signal events,\n'
                          '90% upper limit of the number of signal events,\n'
                          'mode of the number of DSNB background events,\n'
                          'mode of the number of atmospheric CC background events,\n'
                          'mode of the number of reactor background events,\n'
                          'best-fit parameter for the number of signal events,\n'
                          'best-fit parameter for the number of DSNB background events,\n'
                          'best-fit parameter for the number of atmo. CC background events,\n'
                          'best-fit parameter for the number of reactor background events\n'
                   .format(number, now, dataset_start, dataset_stop))

# To save the general information about the analysis, SAVE_DATA must be True:
if SAVE_DATA:
    # save the general information of the analysis in txt file:
    np.savetxt(path_analysis + '/info_mcmc_analysis_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               np.array([DM_mass, min_E_cut, max_E_cut, interval_E_visible, S_true, S_max, B_DSNB_true,
                         B_CCatmo_true, B_Reactor_true, nwalkers, value_of_a, number_of_steps, step_burnin]),
               fmt='%4.5f',
               header='General information about the MCMC analysis of virtual experiment to the expected spectra '
                      '(analyzed with analyze_spectra_v5_local_likelihood.py, {0}):\n'
                      'The Datasets are saved in folder: {3}\n'
                      'Analyzed datasets: Dataset_{1:d}.txt to Dataset_{2:d}.txt\n'
                      'Input files of the simulated spectra:\n'
                      '{4},\n'
                      '{5},\n'
                      '{6},\n'
                      '{7},\n'
                      'The likelihood function is calculated, NOT the log-likelihood function!!!\n'
                      'Prior Probability of Signal: flat_distribution (1/S_max) from 0 to S_max\n'
                      'Prior Prob. of DSNB bkg: Gaussian with mean=B_DSNB_true and sigma = 2*B_DSNB_true\n'
                      'Corresponding to page 16 in the GERDA paper -> "very poorly known background"\n'
                      'Prior Prob. of atmo. CC bkg: Gaussian with mean=B_CCatmo_true and sigma = B_CCatmo_true/2\n'
                      'Prior Prob. of reactor bkg: Gaussian with mean=B_Reactor_true and sigma = B_Reactor_true/2\n'
                      'Equations 20 and 21 of GERDA paper ("fairly known background")\n'
                      'Values below:\n'
                      'Dark matter mass in MeV:\n'
                      'minimum E_cut in MeV, maximum E_cut in MeV, interval-width of the E_cut array in MeV,\n'
                      'Expected number of signal events in this energy range,\n'
                      'S_max,\n'
                      'Expected number of DSNB background events in this energy range,\n'
                      'Expected number of CC atmospheric background events in this energy range,\n'
                      'Expected number of reactor background events in this energy range,\n'
                      'Number of walkers in the Markov Chain,\n'
                      'parameter "a", which controls the step size in the Markov Chain,\n'
                      'number of steps in the chain,\n'
                      'number of step, which are used for "burning in" (the first steps are not considered in the '
                      'sample):'
               .format(now, dataset_start, dataset_stop, path_dataset, file_signal, file_DSNB, file_CCatmo,
                       file_reactor))

    # Save the mean of the acceptance fractions during sampling of every analyzed dataset to txt-file:
    np.savetxt(path_analysis + '/acceptance_fraction_sampling_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               af_sample_mean_array, fmt='%4.5f',
               header='Mean values of the acceptance fraction during sampling of the sample from the MCMC analysis of '
                      'the virt. experiments {0:d} to {1:d}:\n'
                      '(analyzed with analyze_spectra_v5_local_likelihood.py, {2})\n'
                      'General information of the analysis are saved in info_mcmc_analysis_{0:d}_{1:d}.txt\n'
                      'Thumb rule: mean of acceptance fraction should be roughly between 0.2 and 0.5:'
               .format(dataset_start, dataset_stop, now))

    # Save the mean of the acceptance fractions during burnin-phase of every analyzed dataset to txt-file:
    np.savetxt(path_analysis + '/acceptance_fraction_burnin_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               af_burnin_mean_array, fmt='%4.5f',
               header='Mean values of the acceptance fraction during burnin-phase of the sample from the MCMC analysis '
                      'of the virt. experiments {0:d} to {1:d}:\n'
                      '(analyzed with analyze_spectra_v5_local_likelihood.py, {2})\n'
                      'General information of the analysis are saved in info_mcmc_analysis_{0:d}_{1:d}.txt\n'
                      'Thumb rule: mean of acceptance fraction should be roughly between 0.2 and 0.5:'
               .format(dataset_start, dataset_stop, now))

    # Save the mean of the autocorrelation time of every analyzed dataset to txt-file:
    np.savetxt(path_analysis + '/autocorrelation_time_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               mean_acor_array, fmt='%4.5f',
               header='Mean values of the autocorrelation time of the sample from the MCMC analysis of the virt.'
                      'experiments {0:d} to {1:d}:\n'
                      '(analyzed with analyze_spectra_v5_local_likelihood.py, {2}\n'
                      'General information of the analysis are saved in info_mcmc_analysis_{0:d}_{1:d}.txt\n'
                      'IMPORTANT: value=1001001 means there was an emcee.autocorr.AutocorrError:'
               .format(dataset_start, dataset_stop, now))
