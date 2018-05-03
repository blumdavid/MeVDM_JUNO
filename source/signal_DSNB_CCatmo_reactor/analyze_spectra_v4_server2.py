#!/junofs/users/dblum/my_env/bin/python
# author: David Blum

""" Script to analyze a large number of dataset-spectra with the simulation-spectrum on the IHEP cluster in China.

    Use the script to analyze datasets corresponding to SEVERAL dark matter masses.

    This script will be included to the auto_analysis.py script and to the auto_simu_analysis.sh script to
    automatically simulate datasets and analyze them for different DM masses.

    Script is based on the analyze_spectra_v4_local.py Script, but changed a bit to be able to run it on the cluster.

    to run the script:

    python /junofs/users/dblum/work/code/analyze_spectra_v4_server.py 0 DM_mass

    give 3 arguments to the script:
    - sys.argv[0] name of the script = analyze_spectra_v4_server.py
    - sys.argv[1] number of the job
    - sys.argv[2] DM mass in MeV

    Note: the changes to the original script analyze_spectra_v4_local.py as highlighted with "### CHANGE:"
"""

# import of the necessary packages:
import datetime
import numpy as np
from scipy.special import factorial
from scipy.special import erf
import scipy.optimize as op
import matplotlib
# To generate images without having a window appear, do:
# The easiest way is use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import emcee
import corner
import sys

# INFO-me: You have to import matplotlib first, then set a non-interactive backend and then import pyplot to avoid
# INFO-me: an ERROR by building the figures!!

""" Set boolean value to define, if the result of the analysis are saved: """
# save the results and the plot of likelihood-function with fit parameters
# (if data should be saved, SAVE_DATA must be True):
SAVE_DATA = True

""" CHANGE: Set the number of the job submitted: """
job_number = int(sys.argv[1])

""" get the DM mass in MeV (float): """
DM_mass = int(sys.argv[2])

""" set the path of the correct folder: """
# TODO: check the directory "path_folder"
path_folder = "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor"

""" set the path of the output folder: """
path_output = path_folder + "/dataset_output_{0}".format(DM_mass)

""" set the path of the folder, where the datasets were saved: """
path_dataset = path_output + "/datasets"

""" set the path of the folder, where the results of the analysis should be saved: """
path_analysis = path_output + "/analysis_mcmc"

""" set the number of the first dataset (is equal to dataset_start in auto_gen_dataset.py script) and the number
    of the last dataset (is equal to dataset_stop in auto_gen_dataset.py script) (integer): """
# TODO: set number of first dataset and of last dataset ("first_dataset" and "last_dataset"):
first_dataset = 1
last_dataset = 10000

""" Go through every dataset and perform the analysis: """
# TODO: set dataset_start and dataset_stop corresponding to the number of jobs and the number of datasets per job
# define the first dataset, which will be analyzed (file: Dataset_dataset_start) (integer):
dataset_start = job_number*100 + 1
# define the last dataset, which will be analyzed (file: Dataset_dataset_stop) (integer):
dataset_stop = job_number*100 + 100


""" Load information about the generation of the datasets from file (np.array of float): """
# TODO: Check, if info-files have the same parameter:
info_dataset = np.loadtxt(path_dataset + "/info_dataset_{0}_to_{1}.txt".format(first_dataset, last_dataset))
# get the bin-width of the visible energy in MeV from the info-file (float):
interval_E_visible = info_dataset[0]
# get minimum of the visible energy in MeV from info-file (float):
min_E_visible = info_dataset[1]
# get maximum of the visible energy in MeV from info-file (float):
max_E_visible = info_dataset[2]

""" Load simulated spectra in events/MeV from file (np.array of float): """
# TODO: set the directory and folder names of the simulated spectra:
path_simu = "/junofs/users/dblum/work/simu_spectra"
file_signal = path_simu + "/signal_DMmass{0}_bin100keV.txt".format(DM_mass)
Spectrum_signal = np.loadtxt(file_signal)

file_info_signal = path_simu + "/signal_info_DMmass{0}_bin100keV.txt".format(DM_mass)
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

""" Preallocate the array, where the acceptance fraction of each analysis is appended to (empty np.array): """
af_mean_array = np.array([])


""" Define functions: """


def ln_likelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    """
    Function, which represents the log-likelihood function.
    The function is defined by the natural logarithm of the function p_spectrum_sb(), that is defined
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
    # INFO-me: until factorial(14) there is no difference between the result for exact=True and exact=False!!
    # INFO-me: from fac(15) to fac(19) the deviation is less than 3.4e-7% !!
    # INFO-me: exact=True is only valid up to factorial(20), for larger values the int64 is to 'small'
    sum_2 = np.log(factorial(data, exact=False))
    sum_3 = lamb

    return np.sum(sum_1 - sum_2 - sum_3)


def ln_priorprob(param):
    """
    Function to calculate the natural logarithm of the prior probabilities of the parameter param.

    The prior probabilities are partly considered from the GERDA paper, page 6, equ. 20 and 21.

    :param param: np.array of the 'unknown' parameters of the prior probability (param represents the number
    of signal events, the number of DSNB background events, the number of atmospheric CC background events,
    and the number of reactor background events (np.array of 4 float)

    :return: the sum of log of the prior probabilities of the different parameters param
    """
    # get the single parameters from param (float):
    s, b_dsnb, b_ccatmo, b_reactor = param

    # define the ln of the prior probability for the expected signal contribution
    # (flat probability until maximum value S_max):
    if 0.0 <= s <= S_max:
        # between 0 and S_max, flat prior probability defined by 1/S_max (float):
        p_0_s = 1/S_max
        # log of the p_0_s (float):
        ln_prior_s = np.log(p_0_s)
    else:
        # if s < 0 or s > S_max, the prior probability p_0_s is set to 0 -> ln(0) = -infinity (float):
        ln_prior_s = -np.inf

    # define the ln of the prior probability for the expected DSNB background contribution (the background
    # contribution is assumed to be 'very poorly known' -> width = 2*B_DSNB_true (poorly known background corresponds
    # to a width = B_DSNB_true, fairly known background corresponds to a width = B_DSNB_true/2).
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_DSNB_true and
    # width = 2*B_DSNB_true):
    # INFO-me: DSNB background is assumed to be Gaussian with width = 2*B_DSNB_true -> "very poorly known background"
    if b_dsnb >= 0.0:
        # define the mean and the width of the Gaussian function (float):
        mu_b_dsnb = B_DSNB_true
        sigma_b_dsnb = B_DSNB_true * 2
        # calculate the natural logarithm of the denominator of equ. 21 (integral over B from 0 to infinity of
        # exp(-(B-mu)**2 / (2*sigma**2)), the integral is given by the error function) (float):
        sum_2_dsnb = np.log(np.sqrt(np.pi/2) * (sigma_b_dsnb * (erf(mu_b_dsnb / (np.sqrt(2)*sigma_b_dsnb)) + 1)))
        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):
        sum_1_dsnb = -(b_ccatmo - mu_b_dsnb)**2 / (2 * sigma_b_dsnb**2)
        # natural logarithm of the prior probability (float):
        ln_prior_b_dsnb = sum_1_dsnb - sum_2_dsnb
    else:
        # if b_dsnb < 0, the prior probability is set to 0 -> ln(0) = -infinity (float):
        ln_prior_b_dsnb = -np.inf

    # define the ln of the prior probability for the expected atmospheric CC background contribution (the background
    # contribution is assumed to be known up to a factor of 2 -> 'fairly known background'.
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_CCatmo_true and
    # width = B_CCatmo_true/2)
    if b_ccatmo >= 0.0:
        # define the mean and the width of the Gaussian function (float):
        mu_b_ccatmo = B_CCatmo_true
        sigma_b_ccatmo = B_CCatmo_true / 2
        # calculate the natural logarithm of the denominator of equ. 21 (integral over B from 0 to infinity of
        # exp(-(B-mu)**2 / (2*sigma**2)), the integral is given by the error function) (float):
        sum_2_ccatmo = np.log(np.sqrt(np.pi/2) * (sigma_b_ccatmo *
                                                  (erf(mu_b_ccatmo / (np.sqrt(2)*sigma_b_ccatmo)) + 1)))
        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):
        sum_1_ccatmo = -(b_ccatmo - mu_b_ccatmo)**2 / (2 * sigma_b_ccatmo**2)
        # natural logarithm of the prior probability (float):
        ln_prior_b_ccatmo = sum_1_ccatmo - sum_2_ccatmo
    else:
        # if b_ccatmo < 0, the prior probability is set to 0 -> ln(o) = -infinity (float):
        ln_prior_b_ccatmo = -np.inf

    # define the ln of the prior probability for the expected reactor background contribution (the background
    # contribution is assumed to be known up to a factor of 2 -> 'fairly known background'.
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_Reactor_true and
    # width = B_events_reactor/2)
    if b_reactor >= 0.0:
        # define the mean and the width of the Gaussian function (float):
        mu_b_reactor = B_Reactor_true
        sigma_b_reactor = B_Reactor_true / 2
        # calculate the natural logarithm of the denominator of equ. 21 (integral over B from 0 to infinity of
        # exp(-(B-mu)**2 / (2*sigma**2)), the integral is given by the error function) (float):
        sum_2_reactor = np.log(np.sqrt(np.pi/2) * (sigma_b_reactor *
                                                   (erf(mu_b_reactor / (np.sqrt(2)*sigma_b_reactor)) + 1)))
        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):
        sum_1_reactor = -(b_reactor - mu_b_reactor)**2 / (2 * sigma_b_reactor**2)
        # natural logarithm of the prior probability (float):
        ln_prior_b_reactor = sum_1_reactor - sum_2_reactor
    else:
        # if b_reactor < 0, the prior probability is set to 0 -> ln(o) = -infinity (float):
        ln_prior_b_reactor = -np.inf

    # return the sum of the log of the prior probabilities (float)
    return ln_prior_s + ln_prior_b_dsnb + ln_prior_b_ccatmo + ln_prior_b_reactor


def ln_posteriorprob(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    """
    Function, which represents the natural logarithm of the full posterior probability of the Bayesian statistic.

    The function is defined by the natural logarithm of nominator of the function p_SB_spectrum(), that is defined
    on page 4, equation 12 in the GERDA paper.

    So the function is proportional to the sum of the ln_likelihood and the ln_prior

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

    :return: the value of the log of the posterior-probability function (full Bayesian probability) (float)
    (alternatively: returns the log of the posterior probability as function of the unknown parameters param)

    """
    # calculate the prior probabilities for the parameters (float):
    lnprior = ln_priorprob(param)

    # check if lnprior is finite. If not, return -infinity as full probability:
    if not np.isfinite(lnprior):
        return -np.inf

    return lnprior + ln_likelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor)


def neg_ln_likelihood(*args):
    """
    Negative of the function ln_likelihood -> only the negative of the log-likelihood can be minimized

    :param args: arguments from the log-likelihood function defined in ln_likelihood

    :return: return the negative of ln_likelihood (float)
    """
    return -ln_likelihood(*args)


# loop over the Datasets (from dataset_start to dataset_stop):
for number in np.arange(dataset_start, dataset_stop+1, 1):

    # load corresponding dataset (unit: events/bin) (np.array of float):
    Data = np.loadtxt(path_dataset + "/Dataset_{0:d}.txt".format(number))
    # dataset in the 'interesting' energy range from min_E_cut to max_E_cut
    # (you have to take (entry_max+1) to get the array, that includes max_E_cut):
    Data = Data[entry_min_E_cut: (entry_max_E_cut + 1)]

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

    """ Sample this distribution using emcee. 
        Start by initializing the walkers in a tiny Gaussian ball around the maximum likelihood result (in the example 
        on the emcee homepage they found that this tends to be a pretty good initialization ni most cases): 
        Walkers are the members of the ensemble. They are almost like separate Metropolis-Hastings chains but, of 
        course, the proposal distribution for a given walker depends on the positions of all the other walkers in 
        the ensemble. See mcmc_GoodmanWeare_2010.pdf for more details."""
    # INFO-me: nwalkers=200 might be ok
    ndim, nwalkers = 4, 200
    pos = [result["x"] + 10**(-4)*np.random.randn(ndim) for i in range(nwalkers)]

    """ Then, we can set up the sampler 
        EnsembleSampler: a generalized Ensemble sampler that uses 2 ensembles for parallelization.
        (The "a" parameter controls the step size, the default is a=2): """
    value_of_a = 2.0
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posteriorprob, a=value_of_a,
                                    args=(Data, fraction_Signal, fraction_DSNB, fraction_CCatmo, fraction_Reactor))

    """ and run the MCMC for 'number_of_steps' steps starting from the tiny ball defined above: 
        (run_mcmc iterates sample() for N iterations and returns the result of the final sample) """
    # TODO-me: the number of steps should be large (greater than around 1000) to get a reproducible result
    # INFO-me: the auto-correlation time is <~ 60s, therefore min. 13000 steps should be made in the chain
    # Info-me: PROBLEM: sampling 13000 steps with 200 walkers took around 10 minutes!!!
    number_of_steps = 3300
    sampler.run_mcmc(pos, number_of_steps)

    """ The best way to see this is to look at the time series of the parameters in the chain. 
    The sampler object now has an attribute called chain that is an array with the shape 
    (nwalkers, number_of_steps, ndim) giving the parameter values for each walker at each step in the chain. 
    The figure below shows the positions of each walker as a function of the number of steps in the chain 
    (sampler.chain[:, :, 0] has shape (nwalkers, number_of_steps), so the steps as function of the walker, 
    sampler.chain[:, :, 0].T is the transpose of the array with shape (number_of_steps, nwalkers), so the walker as 
    function of steps): """
    """
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
        fig.savefig(path_analysis + 'Dataset{0}_chain_traces.png'.format(number))
    plt.close(fig)
    """

    """ Calculate the mean acceptance fraction. The acceptance fraction is the ratio between the accepted steps 
        over the total number of steps. In general, acceptance_fraction has an entry for each walker (So 
        it is a nwalkers-dimensional vector, therefore calculate the mean). 
        (See: https://gist.github.com/banados/2254240 and 
        http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay8_MCMC_with_emcee.html)
        -   (thumb rule: the acceptance fraction should be between 0.2 and 0.5! If af < 0.2 decrease the a parameter,
            if af > 0.5 increase the a parameter)
        -   af -> 0 would mean that the chain is not proceeding and is not sampling from the posterior PDF
        -   af -> 1 would mean that the chain is performing a random walk, without effectively sampling the post. PDF
        """
    # INFO-me: mean of acceptance fraction should be roughly between 0.2 and 0.5
    # get acceptance fraction (1D np.array of float, dimension = nwalkers):
    af = sampler.acceptance_fraction
    # calculate the mean of the acceptance fraction (float):
    af_mean = np.mean(af)
    # append af_mean to the array, which will be saved to txt-file (np.array of float):
    af_mean_array = np.append(af_mean_array, af_mean)

    """ Calculate the auto-correlation time for the chain. """
    # TODO-me: include the auto-correlation time to estimate the performance and reliability of the MCMC
    # get the auto-correlation time (1D np.array of float, dimension = ndim (one entry for each dimension of
    # parameter space))
    # (The longer the auto-correlation time, the larger the number of the samples we must generate to obtain the
    # desired sampling of the posterior PDF)
    # (You should run the sampler for a few (e.g. 10) auto-correlation times. After that, you are almost completely
    # sure to have independent samples from the posterior PDF):
    # autocorr_time = sampler.acor
    # print("auto-correlation time = {0}".format(autocorr_time))

    """ we’ll just accept it and discard the initial 50 steps and flatten the chain so that we have a flat list of 
        samples: """
    # INFO-me: step_burnin should be 'correct' value
    # set number of step, which are used for "burning in" (the first 'step_burnin' steps are not considered in the
    # sample) (integer):
    step_burnin = 300
    # Take only the samples for step number greater than 'step_burnin' (np.array of floats, three-dimensional array
    # of shape (number walkers nwalkers, number of steps after step_burnin, dimensions ndim), e.g (200, 3000, 4)).
    # AND: flatten the chain along the zeroth (nwalkers) and first (steps after burnin) axis
    # (two dimensional array of shape (nwalkers*steps, ndim), so e.g. (200*3000, 4) = (600000, 4)):
    samples = sampler.chain[:, step_burnin:, :].reshape((-1, ndim))

    """ Calculate the mode and the 90% upper limit of the signal_sample distribution: """
    # get the sample-chain of the signal contribution (np.array of float):
    signal_sample = samples[:, 0]
    # put signal_sample in a histogram (2 np.arrays of float):
    hist_S, bins_S = np.histogram(signal_sample, bins='auto', range=(0, signal_sample.max()))
    # get the index, where hist_S is maximal (integer):
    index_S = np.argmax(hist_S)
    # get the mode of the signal_sample (float):
    S_mode = bins_S[index_S]
    # Calculate the 90 percent upper limit of the signal contribution (float)
    S_90 = np.percentile(signal_sample, 90)

    """ Calculate the mode of the DSNB_sample distribution: """
    # get the sample-chain of the DSNB background contribution (np.array of float):
    DSNB_sample = samples[:, 1]
    # put DSNB_sample in a histogram (2 np.arrays of float):
    hist_DSNB, bins_DSNB = np.histogram(DSNB_sample, bins='auto', range=(0, DSNB_sample.max()))
    # get the index, where hist_DSNB is maximal (integer):
    index_DSNB = np.argmax(hist_DSNB)
    # get the mode of the DSNB_sample (float):
    DSNB_mode = bins_DSNB[index_DSNB]

    """ Calculate the mode of the CCatmo_sample distribution: """
    # get the sample-chain of the atmo. CC background contribution (np.array of float):
    CCatmo_sample = samples[:, 2]
    # put CCatmo_sample in a histogram (2 np.arrays of float):
    hist_CCatmo, bins_CCatmo = np.histogram(CCatmo_sample, bins='auto', range=(0, CCatmo_sample.max()))
    # get the index, where hist_CCatmo is maximal (integer):
    index_CCatmo = np.argmax(hist_CCatmo)
    # get the mode of the CCatmo_sample (float):
    CCatmo_mode = bins_CCatmo[index_CCatmo]

    """ Calculate the mode of the Reactor_sample distribution: """
    # get the sample-chain of the reactor background contribution (np.array of float):
    Reactor_sample = samples[:, 3]
    # put Reactor_sample in a histogram (2 np.arrays of float):
    hist_Reactor, bins_Reactor = np.histogram(Reactor_sample, bins='auto', range=(0, Reactor_sample.max()))
    # get the index, where hist_Reactor is maximal (integer):
    index_Reactor = np.argmax(hist_Reactor)
    # get the mode of the Reactor_sample (float):
    Reactor_mode = bins_Reactor[index_Reactor]

    """ Now that we have this list of samples, let’s make one of the most useful plots you can make with your MCMC 
        results: a corner plot. Generate a corner plot is as simple as: """
    # NOTE: the quantile(0.5) is equal to the np.median() and is equal to np.percentile(50) -> NOT equal to np.mean()
    # fig1 = corner.corner(samples, labels=["$S$", "$B_{DSNB}$", "$B_{CCatmo}$", "$B_{reactor}$"],
    #                      truths=[S_true, B_DSNB_true, B_CCatmo_true, B_Reactor_true], truth_color='b',
    #                      quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.4f',
    #                      labels_args={"fontsize": 40})
    # save figure:
    # if SAVE_DATA:
    #     fig1.savefig(path_analysis + "/Dataset{0}_fitresult.png".format(number))
    # plt.close(fig1)

    """ Clear the chain and lnprobability array. Also reset the bookkeeping parameters: """
    sampler.reset()

    # to save the analysis, SAVE_DATA must be True:
    if SAVE_DATA:
        # save the output of the analysis of the dataset to txt-file:
        np.savetxt(path_analysis + '/Dataset{0}_mcmc_analysis.txt'.format(number),
                   np.array([S_mode, S_90, DSNB_mode, CCatmo_mode, Reactor_mode,
                            S_maxlikeli, B_dsnb_maxlikeli, B_ccatmo_maxlikeli, B_reactor_maxlikeli]),
                   fmt='%4.5f',
                   header='Results of the MCMC analysis of vir. experiment (Dataset_{0:d}) to the expected spectrum '
                          '(job_number = {4}, analyzed with analyze_spectra_v4_local.py, {1}):\n'
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
                          'best-fit parameter for the number of reactor background events:'
                   .format(number, now, dataset_start, dataset_stop, job_number))

# To save the general information about the analysis, SAVE_DATA must be True:
if SAVE_DATA:
    # save the general information of the analysis in txt file:
    np.savetxt(path_analysis + '/info_mcmc_analysis_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               np.array([DM_mass, min_E_cut, max_E_cut, interval_E_visible, S_true, S_max, B_DSNB_true,
                         B_CCatmo_true, B_Reactor_true, nwalkers, value_of_a, number_of_steps, step_burnin]),
               fmt='%4.5f',
               header='General information about the MCMC analysis of virtual experiment to the expected spectra '
                      '(job_number = {8}, analyzed with analyze_spectra_v4_local.py, {0}):\n'
                      'The Datasets are saved in folder: {3}\n'
                      'Analyzed datasets: Dataset_{1:d}.txt to Dataset_{2:d}.txt\n'
                      'Input files of the simulated spectra:\n'
                      '{4},\n'
                      '{5},\n'
                      '{6},\n'
                      '{7},\n'
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
                      'B_DSNB_max,\n'
                      'Expected number of CC atmospheric background events in this energy range,\n'
                      'Expected number of reactor background events in this energy range,\n'
                      'Number of walkers in the Markov Chain,\n'
                      'parameter "a", which controls the step size in the Markov Chain,\n'
                      'number of steps in the chain,\n'
                      'number of step, which are used for "burning in" (the first steps are not considered in the '
                      'sample):'
               .format(now, dataset_start, dataset_stop, path_dataset, file_signal, file_DSNB, file_CCatmo,
                       file_reactor, job_number))

    # Save the mean of the acceptance fractions of every analyzed dataset to txt-file:
    np.savetxt(path_analysis + '/acceptance_fraction_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               af_mean_array, fmt='%4.5f',
               header='Mean values of the acceptance fraction from the MCMC analysis of the virt. experiments'
                      ' {0:d} to {1:d} (job_number = {3}):\n'
                      '(analyzed with analyze_spectra_v4_local.py, {2})\n'
                      'General information of the analysis are saved in info_mcmc_analysis_{0:d}_{1:d}.txt\n'
                      'Thumb rule: mean of acceptance fraction should be roughly between 0.2 and 0.5:'
               .format(dataset_start, dataset_stop, now, job_number))
