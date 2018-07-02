#!/junofs/users/dblum/my_env/bin/python
# author: David Blum

""" Script to analyze a large number of dataset-spectra with the simulation-spectrum on the IHEP cluster in China.

    Use the script to analyze datasets corresponding to ONE dark matter mass.

    Only the signal contribution is fitted to the data. The three backgrounds are fixed to the value of the expected
    number of events!!!!

    The Condor description file is called "condor_desc_file".

    Script is based on the analyze_spectra_v5_local.py Script, but changed a bit to be able to run it on the cluster.

    to run the script:

    python /junofs/users/dblum/work/code/analyze_spectra_v5_server.py 0 /junofs/users/dblum/work/
    dataset_output_20/datasets/ dataset_output_20/analysis_mcmc/

    give 7 arguments to the script:
    - sys.argv[0] name of the script = analyze_spectra_v5_server.py
    - sys.argv[1] number of the job = 0
    - sys.argv[2] directory of the work-folder = /junofs/users/dblum/work/
    - sys.argv[3] directory of the datasets = dataset_output_20/datasets/
    - sys.argv[4] directory of the output files = dataset_output_20/analysis_mcmc/
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

""" set the path of the folder, where the datasets were saved: """
path_dataset = str(sys.argv[2]) + str(sys.argv[3])
### path_dataset = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor/dataset_output_30/datasets/"

""" set the path of the folder, where the results of the analysis should be saved: """
path_analysis = str(sys.argv[2]) + str(sys.argv[4])
### path_analysis = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor/dataset_output_30/" \
###                 "analysis_mcmc_fixBkg/"

""" Go through every dataset and perform the analysis: """
# define the first dataset, which will be analyzed (file: Dataset_dataset_start) (integer):
dataset_start = job_number*50 + 1
### dataset_start = 1

#  define the last dataset, which will be analyzed (file: Dataset_dataset_stop) (integer):
dataset_stop = job_number*50 + 50
### dataset_stop = 1

""" Load information about the generation of the datasets from file (np.array of float): """
# TODO: Check, if info-files have the same parameter:
info_dataset = np.loadtxt(path_dataset + "info_dataset_1_to_10000.txt")
# get the bin-width of the visible energy in MeV from the info-file (float):
interval_E_visible = info_dataset[0]
# get minimum of the visible energy in MeV from info-file (float):
min_E_visible = info_dataset[1]
# get maximum of the visible energy in MeV from info-file (float):
max_E_visible = info_dataset[2]

""" Load simulated spectra in events/MeV from file (np.array of float): """
file_signal = str(sys.argv[2]) + "simu_spectra/signal_DMmass30_bin100keV.txt"
### file_signal = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/signal_DMmass30_bin100keV.txt"
Spectrum_signal = np.loadtxt(file_signal)

file_info_signal = str(sys.argv[2]) + "simu_spectra/signal_info_DMmass30_bin100keV.txt"
### file_info_signal = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/signal_info_DMmass30_bin100keV.txt"
info_signal = np.loadtxt(file_info_signal)

file_DSNB = str(sys.argv[2]) + "simu_spectra/DSNB_EmeanNuXbar22_bin100keV.txt"
### file_DSNB = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/DSNB_EmeanNuXbar22_bin100keV.txt"
Spectrum_DSNB = np.loadtxt(file_DSNB)

file_reactor = str(sys.argv[2]) + "simu_spectra/Reactor_NH_power36_bin100keV.txt"
### file_reactor = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/Reactor_NH_power36_bin100keV.txt"
Spectrum_reactor = np.loadtxt(file_reactor)

file_CCatmo = str(sys.argv[2]) + "simu_spectra/CCatmo_Osc1_bin100keV.txt"
### file_CCatmo = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/CCatmo_Osc1_bin100keV.txt"
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


def ln_likelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    """
    Function, which represents the log-likelihood function.

    The log-likelihood function only depends on the number of signal events. All other parameters are fixed.

    The function is defined by the natural logarithm of the function p_spectrum_sb(), that is defined
    on page 3, equation 11 in the GERDA paper.

    The likelihood function is given in the GERDA paper (equation 11: p_spectrum_sb) and in the paper arXiv:1208.0834
    'A novel way of constraining WIMPs annihilation in the Sun: MeV neutrinos', page 15, equation (24)

    :param param: 'unknown' parameter of the log-likelihood function (param represents the number
    of signal events (float)

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
    # get the parameter from param (float):
    s = param
    # calculate the variable lambda_i(s), which is defined on page 3, equ. 9 of the
    # GERDA paper 'A Bayesian approach to the analysis of sparsely populated spectra, Calculating the sensitivity of
    # the GERDA experiment' (np.array of float):
    lamb = (fraction_signal*s + fraction_dsnb*B_DSNB_true + fraction_ccatmo*B_CCatmo_true +
            fraction_reactor*B_Reactor_true)

    # calculate the addends (Summanden) of the log-likelihood-function defined on page 3, equ. 11 of the GERDA paper
    # (np.array of float):
    sum_1 = data*np.log(lamb)

    """ From documentation scipy 0.14: scipy.special.factorial: (on the server scipy 0.19 is installed)
    scipy.special.factorial is the factorial function, n! = special.gamma(n+1).
    If exact is 0, then floating point precision is used, otherwise exact long integer is computed.
    Array argument accepted only for exact=False case. If n<0, the return value is 0.
    """
    # INFO-me: until factorial(14) there is no difference between the result for exact=True and exact=False!!
    # INFO-me: from fac(15) to fac(19) the deviation is less than 3.4e-7% !!
    # INFO-me: exact=True is only valid up to factorial(20), for larger values the int64 is to 'small'
    sum_2 = np.log(factorial(data, exact=False))
    sum_3 = lamb

    return np.sum(sum_1 - sum_2 - sum_3)


def ln_priorprob(param):
    """
    Function to calculate the natural logarithm of the prior probability of the parameter param.

    The prior probabilities are partly considered from the GERDA paper, page 6, equ. 20 and 21.

    :param param: 'unknown' parameter of the prior probability (param represents the number
    of signal events (float)

    :return: the sum of log of the prior probabilities of the different parameters param
    """
    # get the single parameters from param (float):
    s = param

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

    # return the log of the prior probability (float)
    return ln_prior_s


def ln_posteriorprob(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    """
    Function, which represents the natural logarithm of the full posterior probability of the Bayesian statistic.

    The function is defined by the natural logarithm of nominator of the function p_SB_spectrum(), that is defined
    on page 4, equation 12 in the GERDA paper.

    So the function is proportional to the sum of the ln_likelihood and the ln_prior

    IMPORTANT:  the denominator of equ. 12 (integral over p_spectrum_SB * p_0_S * p_0_b) is not considered
                -> it is just a normalization constant

    :param param: 'unknown' parameter of the log-likelihood function (param represents the number
    of signal events (float)

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
    # calculate the prior probabilities for the parameter (float):
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
    Data = np.loadtxt(path_dataset + "Dataset_{0:d}.txt".format(number))
    # dataset in the 'interesting' energy range from min_E_cut to max_E_cut
    # (you have to take (entry_max+1) to get the array, that includes max_E_cut):
    Data = Data[entry_min_E_cut: (entry_max_E_cut + 1)]

    # guess of the parameters (as guess, the total number of events from the simulated spectrum are used)
    # (np.array of float):
    parameter_guess = np.array([S_true])

    # bounds of the parameter (parameters have to be positive or zero) (tuple):
    bnds = ((0, None),)

    # Minimize the negative log-likelihood function with the L-BFGS-B method for the defined bounds above:
    result = op.minimize(neg_ln_likelihood, parameter_guess,
                         args=(Data, fraction_Signal, fraction_DSNB, fraction_CCatmo, fraction_Reactor),
                         method='L-BFGS-B', bounds=bnds, options={'disp': None})

    # get the best-fit parameters from the minimization (float):
    S_maxlikeli = result["x"]

    """ Sample this distribution using emcee. 
        Start by initializing the walkers in a tiny Gaussian ball around the maximum likelihood result (in the example 
        on the emcee homepage they found that this tends to be a pretty good initialization in most cases): 
        Walkers are the members of the ensemble. They are almost like separate Metropolis-Hastings chains but, of 
        course, the proposal distribution for a given walker depends on the positions of all the other walkers in 
        the ensemble. See mcmc_GoodmanWeare_2010.pdf for more details.
        Run with large number of walkers -> more independent samples per autocorrelation time. BUT: disadvantage of 
        large number of walkers is that the burnin-phase can be slow. Therefore: use the smallest number of walkers 
        for which the acceptance fraction during burn-in is good (see emcee_1202.3665.pdf): """
    # INFO-me: nwalkers=200 might be ok
    ndim, nwalkers = 1, 50
    P0 = [result["x"] + 10**(-4)*np.random.randn(ndim) for i in range(nwalkers)]

    """ Then, we can set up the sampler 
        EnsembleSampler: a generalized Ensemble sampler that uses 2 ensembles for parallelization.
        (The "a" parameter controls the step size, the default is a=2): """
    value_of_a = 3.0
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posteriorprob, a=value_of_a,
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
    # print("mean acceptance fraction during burnin-phase = {0}".format(af_burnin_mean))

    """ Reset the sampler (to get rid of the previous chain and start where the sampler left off at variable 'pos'):"""
    sampler.reset()

    """ Now run the MCMC for 'number_of_steps' steps starting, where the sampler left off in the burnin-phase: 
        (run_mcmc iterates sample() for N iterations and returns the result of the final sample) """
    # TODO-me: the number of steps should be large (greater than around 10000) to get a reproducible result
    number_of_steps = 15000
    sampler.run_mcmc(pos, number_of_steps)

    """ The best way to see this is to look at the time series of the parameters in the chain. 
    The sampler object now has an attribute called chain that is an array with the shape 
    (nwalkers, number_of_steps, ndim) giving the parameter values for each walker at each step in the chain. 
    The figure below shows the positions of each walker as a function of the number of steps in the chain 
    (sampler.chain[:, :, 0] has shape (nwalkers, number_of_steps), so the steps as function of the walker, 
    sampler.chain[:, :, 0].T is the transpose of the array with shape (number_of_steps, nwalkers), so the walker as 
    function of steps): """
    """
    fig = plt.figure(1)
    fig.set_size_inches(10, 10)
    plt.plot(sampler.chain[:, :, 0].T, '-', color='k', alpha=0.3)
    plt.axhline(y=S_true, color='b', linestyle='--')
    plt.ylabel('$S$')
    plt.legend()
    if SAVE_DATA:
        fig.savefig(path_analysis + 'Dataset{0}_chain_traces.png'.format(number))
    plt.close(fig)
    """

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
        # calculate the mean of autocorr_time (float):
        mean_autocorr_time = np.mean(autocorr_time)
    except emcee.autocorr.AutocorrError:
        # if there is an emcee.autocorr.AutocorrError, set the autocorrelation time to 1001001 (-> this means that there
        # was an error):
        mean_autocorr_time = 1001001

    # append mean_autocorr_time to the array (np.array of float):
    mean_acor_array = np.append(mean_acor_array, mean_autocorr_time)

    """ flatten the chain so that we have a flat list of samples: """
    # the chain is a three-dimensional array of shape (number walkers nwalkers, number of steps, dimensions ndim),
    # e.g (200, 3000, 4)).
    # The 'flatchain' function flattens the chain along the zeroth (nwalkers) axis and you get a two dimensional
    # array of shape (nwalkers*steps, ndim), so e.g. (200*3000, 4) = (600000, 4)):
    samples = sampler.flatchain

    """ Calculate the mode and the 90% upper limit of the signal_sample distribution: """
    # get the sample-chain of the signal contribution (np.array of float):
    signal_sample = samples
    # put signal_sample in a histogram (2 np.arrays of float), hist are the values of the histogram,
    # bins return the bin edges (length(hist)+1)
    hist_S, bin_edges_S = np.histogram(signal_sample, bins='auto', range=(0, signal_sample.max()))
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

    fig1 = plt.figure(2)
    plt.step(bin_edges_S[:-1], hist_S, label="S_mode = {0:.4f}\nS_90 = {1:.4f}".format(S_mode, S_90))
    plt.xticks(np.arange(0, 20, 0.5))
    plt.xlabel("number of 'observed' signal events S")
    plt.ylabel("counts")
    plt.title("p(S) from MCMC sampling for one dataset")
    plt.legend()
    # save figure:
    if SAVE_DATA:
        fig1.savefig(path_analysis + "Dataset{0}_fitresult.png".format(number))
    plt.close(fig1)

    """ Clear the chain and lnprobability array. Also reset the bookkeeping parameters: """
    sampler.reset()

    # to save the analysis, SAVE_DATA must be True:
    if SAVE_DATA:
        # save the output of the analysis of the dataset to txt-file:
        np.savetxt(path_analysis + 'Dataset{0}_mcmc_analysis.txt'.format(number),
                   np.array([S_mode, S_90, S_maxlikeli]),
                   fmt='%4.5f',
                   header='Results of the MCMC analysis of vir. experiment (Dataset_{0:d}) to the expected spectrum '
                          '(analyzed with analyze_spectra_v5_server_fixBkg.py, {1}):\n'
                          'General information of the analysis are saved in info_mcmc_analysis_{2:d}_{3:d}.txt\n'
                          'Results of the analysis:\n'
                          'mode of the number of signal events,\n'
                          '90% upper limit of the number of signal events,\n'
                          'best-fit parameter for the number of signal events:'
                   .format(number, now, dataset_start, dataset_stop))

# To save the general information about the analysis, SAVE_DATA must be True:
if SAVE_DATA:
    # save the general information of the analysis in txt file:
    np.savetxt(path_analysis + 'info_mcmc_analysis_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               np.array([DM_mass, min_E_cut, max_E_cut, interval_E_visible, S_true, S_max, B_DSNB_true,
                         B_CCatmo_true, B_Reactor_true, nwalkers, value_of_a, number_of_steps, step_burnin]),
               fmt='%4.5f',
               header='General information about the MCMC analysis of virtual experiment to the expected spectra '
                      '(analyzed with analyze_spectra_v5_server_fixBkg.py, {0}):\n'
                      'The Datasets are saved in folder: {3}\n'
                      'Analyzed datasets: Dataset_{1:d}.txt to Dataset_{2:d}.txt\n'
                      'Input files of the simulated spectra:\n'
                      '{4},\n'
                      '{5},\n'
                      '{6},\n'
                      '{7},\n'
                      'Prior Probability of Signal: flat_distribution (1/S_max) from 0 to S_max\n'
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
                      '(analyzed with analyze_spectra_v5_server.py, {2})\n'
                      'General information of the analysis are saved in info_mcmc_analysis_{0:d}_{1:d}.txt\n'
                      'Thumb rule: mean of acceptance fraction should be roughly between 0.2 and 0.5:'
               .format(dataset_start, dataset_stop, now))

    # Save the mean of the acceptance fractions during burnin-phase of every analyzed dataset to txt-file:
    np.savetxt(path_analysis + '/acceptance_fraction_burnin_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               af_burnin_mean_array, fmt='%4.5f',
               header='Mean values of the acceptance fraction during burnin-phase of the sample from the MCMC analysis '
                      'of the virt. experiments {0:d} to {1:d}:\n'
                      '(analyzed with analyze_spectra_v5_server.py, {2})\n'
                      'General information of the analysis are saved in info_mcmc_analysis_{0:d}_{1:d}.txt\n'
                      'Thumb rule: mean of acceptance fraction should be roughly between 0.2 and 0.5:'
               .format(dataset_start, dataset_stop, now))

    # Save the mean of the autocorrelation time of every analyzed dataset to txt-file:
    np.savetxt(path_analysis + '/autocorrelation_time_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               mean_acor_array, fmt='%4.5f',
               header='Mean values of the autocorrelation time of the sample from the MCMC analysis of the virt.'
                      'experiments {0:d} to {1:d}:\n'
                      '(analyzed with analyze_spectra_v5_server.py, {2}\n'
                      'General information of the analysis are saved in info_mcmc_analysis_{0:d}_{1:d}.txt\n'
                      'IMPORTANT: value=1001001 means there was an emcee.autocorr.AutocorrError:'
               .format(dataset_start, dataset_stop, now))
