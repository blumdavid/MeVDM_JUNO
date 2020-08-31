""" Script to calculate 90 % upper limit of spectrum with Bayesian analysis and MCMC

    Difference between version 7 and 6:
        -   No datasets are analyzed!

        -   number of observed events (n_i) is given by the background only spectrum
            (DSNB + reactor + CCatmo_p + CCatmo_C12 + NCatmo).
            No free parameters in n_i

        -   ACHTUNG:  Es wird die Fakultät von Dezimalzahlen berechnet!!!!!

    Difference between version 6 and version 5:
                atmospheric NC background and fast neutron background is also implemented.

    Difference between version 5 and version 4:
                MCMC sampling is implemented differently to version4:
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

    Dataset (virtual experiments) are generated with gen_dataset_v2_local.py
    Simulated spectra are generated with gen_spectrum_v3.py

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
from gen_spectrum_functions import limit_annihilation_crosssection_v2, limit_neutrino_flux_v2

# TODO: i have to cite the package 'emcee', when I use it to analyze

# TODO-me: Check the sensitivity of the results depending on the prior probabilities

# set DM mass that should be investigated:
# mass_DM = [15.0]
mass_DM = np.arange(15, 100+5, 5)

# output path:
path_analysis = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo/test_Bayesian_ohne_datasets"

""" Load simulated spectra from file (np.array of float): """
# load bkg spectra:
path_simu = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/"
path_simu_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
               "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
               "test_10to20_20to30_30to40_40to100_final/"
# DSNB spectrum in events/bin:
file_DSNB = path_simu + "/DSNB_bin500keV_PSD.txt"
Spectrum_DSNB = np.loadtxt(file_DSNB)
# reactor spectrum in events/bin:
file_reactor = path_simu + "/Reactor_NH_power36_bin500keV_PSD.txt"
Spectrum_reactor = np.loadtxt(file_reactor)
# atmo. CC background on proton in events/bin:
file_CCatmo_p = path_simu + "/CCatmo_onlyP_Osc1_bin500keV_PSD.txt"
Spectrum_CCatmo_p = np.loadtxt(file_CCatmo_p)
# atmo. CC background on C12 in events/bin:
file_CCatmo_C12 = path_simu + "/CCatmo_onlyC12_Osc1_bin500keV_PSD.txt"
Spectrum_CCatmo_C12 = np.loadtxt(file_CCatmo_C12)
# atmo. NC background in events/bin:
file_NCatmo = path_simu_NC + "/NCatmo_onlyC12_wPSD99_bin500keV.txt"
Spectrum_NCatmo = np.loadtxt(file_NCatmo)

""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

""" Define the energy window, where spectrum of virtual experiment and simulated spectrum is analyzed
    (from min_E_cut in MeV to max_E_cut in MeV): """
min_E_visible = 10.0
max_E_visible = 100.0
interval_E_visible = 0.5
min_E_cut = min_E_visible
max_E_cut = max_E_visible
# calculate the entry number of the array to define the energy window:
entry_min_E_cut = int((min_E_cut - min_E_visible) / interval_E_visible)
entry_max_E_cut = int((max_E_cut - min_E_visible) / interval_E_visible)

""" Simulated spectra in events/bin (multiply with interval_E_visible): """
# spectrum per bin in the 'interesting' energy range from min_E_cut to max_E_cut
# (you have to take (entry_max+1) to get the array, that includes max_E_cut):
spectrum_DSNB_per_bin = Spectrum_DSNB[entry_min_E_cut: (entry_max_E_cut + 1)]
spectrum_CCatmo_p_per_bin = Spectrum_CCatmo_p[entry_min_E_cut: (entry_max_E_cut + 1)]
spectrum_CCatmo_C12_per_bin = Spectrum_CCatmo_C12[entry_min_E_cut: (entry_max_E_cut + 1)]
spectrum_Reactor_per_bin = Spectrum_reactor[entry_min_E_cut: (entry_max_E_cut + 1)]
spectrum_NCatmo_per_bin = Spectrum_NCatmo[entry_min_E_cut: (entry_max_E_cut + 1)]

""" 'true' values of the parameters: """
# expected number of DSNB background events in the energy window (float):
B_DSNB_true = np.sum(spectrum_DSNB_per_bin)
# expected number of CCatmo background events on protons in the energy window (float):
B_CCatmo_p_true = np.sum(spectrum_CCatmo_p_per_bin)
# expected number of CCatmo background events on C12 in the energy window (float):
B_CCatmo_C12_true = np.sum(spectrum_CCatmo_C12_per_bin)
# expected number of Reactor background events in the energy window (float):
B_Reactor_true = np.sum(spectrum_Reactor_per_bin)
# expected number of NCatmo background events in the energy window (float):
B_NCatmo_true = np.sum(spectrum_NCatmo_per_bin)

""" fractions (normalized shapes) of signal and background spectra: """
# Fraction of DSNB background (np.array of float):
fraction_DSNB = spectrum_DSNB_per_bin / B_DSNB_true
# Fraction of CCatmo background on protons (np.array of float):
fraction_CCatmo_p = spectrum_CCatmo_p_per_bin / B_CCatmo_p_true
# Fraction of CCatmo background on C12 (np.array of float):
fraction_CCatmo_C12 = spectrum_CCatmo_C12_per_bin / B_CCatmo_C12_true
# Fraction of reactor background (np.array of float):
fraction_Reactor = spectrum_Reactor_per_bin / B_Reactor_true
# Fraction of NCatmo background (np.array of float):
fraction_NCatmo = spectrum_NCatmo_per_bin / B_NCatmo_true

""" Preallocate the array, where the acceptance fraction of the actual sampling of each analysis is appended to 
(empty np.array): """
af_sample_mean_array = np.array([])
""" Preallocate the array, where the acceptance fraction during burnin-phase is appended to (empty np.array): """
af_burnin_mean_array = np.array([])
""" Preallocate the array, where the mean of autocorrelation time is appended to (empty np.array): """
mean_acor_array = np.array([])

# array, where 90 % limit of signal contribution is stored:
S_90_array = []
# array, where the PSD efficiency is stored for each DM mass:
PSD_eff_array = []

""" Define functions: """
# INFO-me: emcee actually requires the logarithm of p (see http://dfm.io/emcee/current/user/quickstart/)


def ln_likelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo_p, fraction_ccatmo_c12, fraction_reactor,
                  fraction_ncatmo):
    """
    Function, which represents the log-likelihood function.
    The function is defined by the natural logarithm of the function p_spectrum_sb(), that is defined
    on page 3, equation 11 in the GERDA paper.

    The likelihood function is given in the GERDA paper (equation 11: p_spectrum_sb) and in the paper arXiv:1208.0834
    'A novel way of constraining WIMPs annihilation in the Sun: MeV neutrinos', page 15, equation (24)

    :param param: np.array of the 'unknown' parameters of the log-likelihood function (param represents the number
    of signal events, the number of DSNB background events, the number of atmospheric CC background events, the number
    of reactor background events, the number of atmospheric NC background events and the number of fast neutron
    background events (np.array of 6 float)

    :param data: represents the data, that the model will be fitted to ('observed' number of events for each bin
    from the dataset ('observed' spectrum)) (np.array of float)

    :param fraction_signal: normalized shapes of the signal spectra (represents f_S in equ. 9 of the GERDA paper),
    equivalent to the number of signal events per bin from the theoretical spectrum (np.array of float)

    :param fraction_dsnb: normalized shapes of the DSNB background spectra (represents f_B_DSNB in equ. 9 of the
    GERDA paper), equivalent to the number of DSNB background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ccatmo_p: normalized shapes of the CC atmo. background spectra on protons
    (represents f_B_CCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of CC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ccatmo_c12: normalized shapes of the CC atmo. background spectra on C12
    (represents f_B_CCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of CC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_reactor: normalized shapes of the reactor background spectra (represents f_B_reactor in equ. 9 of
    the GERDA paper), equivalent to the number of reactor background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ncatmo: normalized shapes of the NC atmo. background spectra (represents f_B_NCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of NC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :return: the value of the log-likelihood function (float) (alternatively: returns the log-likelihood function
    as function of the unknown parameters param)
    """
    # get the single parameters from param (float):
    s, b_dsnb, b_ccatmo_p, b_ccatmo_c12, b_reactor, b_ncatmo = param

    # calculate the variable lambda_i(s, b_dsnb, b_ccatmo_p, b_ccatmo_c12, b_reactor, b_ncatmo),
    # which is defined on page 3,
    # equ. 9 of the GERDA paper 'A Bayesian approach to the analysis of sparsely populated spectra, Calculating the
    # sensitivity of the GERDA experiment' (np.array of float):
    lamb = (fraction_signal*s + fraction_dsnb*b_dsnb + fraction_ccatmo_p*b_ccatmo_p + fraction_ccatmo_c12*b_ccatmo_c12
            + fraction_reactor*b_reactor + fraction_ncatmo*b_ncatmo)

    # calculate the addends (Summanden) of the log-likelihood-function defined on page 3, equ. 11 of the GERDA paper
    # (np.array of float):
    sum_1 = data * np.log(lamb)

    """ From documentation scipy 0.14: scipy.special.factorial: (on the server scipy 0.19 is installed, 
        on pipc79 scipy 1.0.0 is installed)
        scipy.special.factorial is the factorial function, n! = special.gamma(n+1).
        If exact is 0, then floating point precision is used, otherwise exact long integer is computed.
        Array argument accepted only for exact=False case. If n<0, the return value is 0.
        """
    # INFO-me: until factorial(14) there is no difference between the result for exact=True and exact=False!!
    # INFO-me: from fac(15) to fac(19) the deviation is less than 3.4e-7% !!
    # INFO-me: exact=True is only valid up to factorial(20), for larger values the int64 is to 'small'
    # TODO-me: Factorial of decimal number is calculated!!!!!!
    sum_2 = np.log(factorial(data, exact=False))
    sum_3 = lamb

    return np.sum(sum_1 - sum_2 - sum_3)


def ln_priorprob(param):
    """
    Function to calculate the natural logarithm of the prior probabilities of the parameter param.

    The prior probabilities are partly considered from the GERDA paper, page 6, equ. 20 and 21.

    :param param: np.array of the 'unknown' parameters of the prior probability (param represents the number
    of signal events, the number of DSNB background events, the number of atmospheric CC background events, the number
    of reactor background events, the number of atmospheric NC background events and the number of fast neutron
    background events (np.array of 6 float)

    :return: the sum of log of the prior probabilities of the different parameters param
    """
    # get the single parameters from param (float):
    s, b_dsnb, b_ccatmo_p, b_ccatmo_c12, b_reactor, b_ncatmo = param

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

    # # define the ln of the prior probability for the expected signal contribution (Iris-prior, 'pessimistic-prior'),
    # # that is also defined in GERDA paper:
    # if 0.0 <= s <= S_max:
    #     # between 0 and S_max, the prior probability is defined by p_0_s = X * exp(-s/10), where X is normalization
    #     # factor. X = 1 / (10 * (1 - exp(-S_max/10))). If taking ln(p_0_s), you get
    #     # ln(p_0_s) = -s/10 - ln(10) - ln(1-exp(-S_max/10))
    #     # first addend:
    #     sum_1_signal = -s/10
    #     # second addend:
    #     sum_2_signal = np.log(10)
    #     # third addend:
    #     sum_3_signal = np.log(1 - np.exp(-S_max / 10))
    #     # ln of prior probability:
    #     ln_prior_s = sum_1_signal - sum_2_signal - sum_3_signal
    # else:
    #     # if s < 0 or s > S_max, the prior probability p_0_s is set to 0 -> ln(0) = -infinity (float):
    #     ln_prior_s = -np.inf

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
        # sigma_b_dsnb = B_DSNB_true / 4.0

        # calculate the natural logarithm of the denominator of equ. 21 (integral over B from 0 to infinity of
        # exp(-(B-mu)**2 / (2*sigma**2)), the integral is given by the error function) (float):
        sum_2_dsnb = np.log(np.sqrt(np.pi/2) * (sigma_b_dsnb * (erf(mu_b_dsnb / (np.sqrt(2)*sigma_b_dsnb)) + 1)))
        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):
        # INFO-me: there was an error in sum_1_dsnb: instead of b_dsnb, there was b_ccatmo (11.06.2018)
        sum_1_dsnb = -(b_dsnb - mu_b_dsnb)**2 / (2 * sigma_b_dsnb**2)
        # natural logarithm of the prior probability (float):
        ln_prior_b_dsnb = sum_1_dsnb - sum_2_dsnb
    else:
        # if b_dsnb < 0, the prior probability is set to 0 -> ln(0) = -infinity (float):
        ln_prior_b_dsnb = -np.inf

    # define the ln of the prior probability for the expected atmospheric CC background contribution on protons
    # (the background contribution is assumed to be known up to a factor of 2 -> 'fairly known background'.
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_CCatmo_p_true and
    # width = B_CCatmo_p_true/2)
    if b_ccatmo_p >= 0.0:
        # define the mean and the width of the Gaussian function (float):
        mu_b_ccatmo_p = B_CCatmo_p_true
        sigma_b_ccatmo_p = B_CCatmo_p_true * 2
        # sigma_b_ccatmo_p = B_CCatmo_p_true / 4.0

        # calculate the natural logarithm of the denominator of equ. 21 (integral over B from 0 to infinity of
        # exp(-(B-mu)**2 / (2*sigma**2)), the integral is given by the error function) (float):
        sum_2_ccatmo_p = np.log(np.sqrt(np.pi/2) * (sigma_b_ccatmo_p *
                                                    (erf(mu_b_ccatmo_p / (np.sqrt(2)*sigma_b_ccatmo_p)) + 1)))
        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):
        sum_1_ccatmo_p = -(b_ccatmo_p - mu_b_ccatmo_p)**2 / (2 * sigma_b_ccatmo_p**2)
        # natural logarithm of the prior probability (float):
        ln_prior_b_ccatmo_p = sum_1_ccatmo_p - sum_2_ccatmo_p
    else:
        # if b_ccatmo < 0, the prior probability is set to 0 -> ln(o) = -infinity (float):
        ln_prior_b_ccatmo_p = -np.inf

    # define the ln of the prior probability for the expected atmospheric CC background contribution on C12
    # (the background contribution is assumed to be known up to a factor of 2 -> 'fairly known background'.
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_CCatmo_C12_true and
    # width = B_CCatmo_C12_true/2)
    if b_ccatmo_c12 >= 0.0:
        # define the mean and the width of the Gaussian function (float):
        mu_b_ccatmo_c12 = B_CCatmo_C12_true
        sigma_b_ccatmo_c12 = B_CCatmo_C12_true * 2
        # sigma_b_ccatmo_c12 = B_CCatmo_C12_true / 4.0

        # calculate the natural logarithm of the denominator of equ. 21 (integral over B from 0 to infinity of
        # exp(-(B-mu)**2 / (2*sigma**2)), the integral is given by the error function) (float):
        sum_2_ccatmo_c12 = np.log(np.sqrt(np.pi / 2) * (sigma_b_ccatmo_c12 *
                                                        (erf(mu_b_ccatmo_c12 / (np.sqrt(2) * sigma_b_ccatmo_c12)) + 1)))
        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):
        sum_1_ccatmo_c12 = -(b_ccatmo_c12 - mu_b_ccatmo_c12) ** 2 / (2 * sigma_b_ccatmo_c12 ** 2)
        # natural logarithm of the prior probability (float):
        ln_prior_b_ccatmo_c12 = sum_1_ccatmo_c12 - sum_2_ccatmo_c12
    else:
        # if b_ccatmo < 0, the prior probability is set to 0 -> ln(o) = -infinity (float):
        ln_prior_b_ccatmo_c12 = -np.inf

    # define the ln of the prior probability for the expected reactor background contribution (the background
    # contribution is assumed to be known up to a factor of 2 -> 'fairly known background'.
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_Reactor_true and
    # width = B_events_reactor/2)
    if b_reactor >= 0.0:
        # define the mean and the width of the Gaussian function (float):
        mu_b_reactor = B_Reactor_true
        sigma_b_reactor = B_Reactor_true * 2
        # sigma_b_reactor = B_Reactor_true / 4.0

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

    # define the ln of the prior probability for the expected atmospheric NC background contribution (the background
    # contribution is assumed to be known up to a factor of 2 -> 'fairly known background'.
    # The prior probability is chosen to be of Gaussian shape with mean value mu = B_NCatmo_true and
    # width = B_NCatmo_true/2):
    if b_ncatmo >= 0.0:
        # define the mean and the width of the Gaussian function (float):
        mu_b_ncatmo = B_NCatmo_true
        sigma_b_ncatmo = B_NCatmo_true * 2
        # sigma_b_ncatmo = B_NCatmo_true / 4.0

        # calculate the natural logarithm of the denominator of equ. 21 (integral over B from 0 to infinity of
        # exp(-(B-mu)**2 / (2*sigma**2)), the integral is given by the error function) (float):
        sum_2_ncatmo = np.log(np.sqrt(np.pi/2) * (sigma_b_ncatmo *
                                                  (erf(mu_b_ncatmo / (np.sqrt(2)*sigma_b_ncatmo)) + 1)))
        # calculate the natural logarithm of the numerator of equ. 21 (do NOT calculate exp(-(B-mu)**2)/(2*sigma**2))
        # first and then take the logarithm, because exp(-1e-3)=0.0 in python and therefore ln(0.0)=-inf for small
        # values) (float):
        sum_1_ncatmo = -(b_ncatmo - mu_b_ncatmo)**2 / (2 * sigma_b_ncatmo**2)
        # natural logarithm of the prior probability (float):
        ln_prior_b_ncatmo = sum_1_ncatmo - sum_2_ncatmo
    else:
        # if b_ncatmo < 0, the prior probability is set to 0 -> ln(o) = -infinity (float):
        ln_prior_b_ncatmo = -np.inf

    # return the sum of the log of the prior probabilities (float)
    return (ln_prior_s + ln_prior_b_dsnb + ln_prior_b_ccatmo_p + ln_prior_b_ccatmo_c12 + ln_prior_b_reactor +
            ln_prior_b_ncatmo)


def ln_posteriorprob(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo_p, fraction_ccatmo_c12,
                     fraction_reactor, fraction_ncatmo):
    """
    Function, which represents the natural logarithm of the full posterior probability of the Bayesian statistic.

    The function is defined by the natural logarithm of nominator of the function p_SB_spectrum(), that is defined
    on page 4, equation 12 in the GERDA paper.

    So the function is proportional to the sum of the ln_likelihood and the ln_prior

    IMPORTANT:  the denominator of equ. 12 (integral over p_spectrum_SB * p_0_S * p_0_b) is not considered
                -> it is just a normalization constant

    :param param: np.array of the 'unknown' parameters of the log-likelihood function (param represents the number
    of signal events, the number of DSNB background events, the number of atmospheric CC background events, the number
    of reactor background events, the number of atmospheric NC background events and the number of fast neutron
    background events (np.array of 6 float)

    :param data: represents the data, that the model will be fitted to ('observed' number of events for each bin
    from the dataset ('observed' spectrum)) (np.array of float)

    :param fraction_signal: normalized shapes of the signal spectra (represents f_S in equ. 9 of the GERDA paper),
    equivalent to the number of signal events per bin from the theoretical spectrum (np.array of float)

    :param fraction_dsnb: normalized shapes of the DSNB background spectra (represents f_B_DSNB in equ. 9 of the
    GERDA paper), equivalent to the number of DSNB background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ccatmo_p: normalized shapes of the CC atmo. background spectra on protons
     (represents f_B_CCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of CC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ccatmo_c12: normalized shapes of the CC atmo. background spectra on C12
     (represents f_B_CCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of CC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_reactor: normalized shapes of the reactor background spectra (represents f_B_reactor in equ. 9 of
    the GERDA paper), equivalent to the number of reactor background events per bin from the theoretical spectrum
    (np.array of float)

    :param fraction_ncatmo: normalized shapes of the NC atmo. background spectra (represents f_B_NCatmo in equ. 9 of the
    GERDA paper), equivalent to the number of NC atmo. background events per bin from the theoretical spectrum
    (np.array of float)

    :return: the value of the log of the posterior-probability function (full Bayesian probability) (float)
    (alternatively: returns the log of the posterior probability as function of the unknown parameters param)

    """
    # calculate the prior probabilities for the parameters (float):
    lnprior = ln_priorprob(param)

    # check if lnprior is finite. If not, return -infinity as full probability:
    if not np.isfinite(lnprior):
        return -np.inf

    return lnprior + ln_likelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo_p, fraction_ccatmo_c12,
                                   fraction_reactor, fraction_ncatmo)


def neg_ln_likelihood(*args):
    """
    Negative of the function ln_likelihood -> only the negative of the log-likelihood can be minimized

    :param args: arguments from the log-likelihood function defined in ln_likelihood

    :return: return the negative of ln_likelihood (float)
    """
    return -ln_likelihood(*args)


# loop over DM masses:
for mass in mass_DM:
    print("----------------------------------------------------------------------------")
    print("\nDM mass = {0:.0f}\n".format(mass))

    # signal spectrum before PSD cut in events/bin:
    file_signal_woPSD = path_simu + "/signal_DMmass{0:.0f}_bin500keV.txt".format(mass)
    Spectrum_signal_woPSD = np.loadtxt(file_signal_woPSD)
    spectrum_Signal_per_bin_woPSD = Spectrum_signal_woPSD[entry_min_E_cut: (entry_max_E_cut + 1)]
    # expected number of signal events before PSD cut:
    S_true_woPSD = np.sum(spectrum_Signal_per_bin_woPSD)

    # signal spectrum in events/bin:
    file_signal = path_simu + "/signal_DMmass{0:.0f}_bin500keV_PSD.txt".format(mass)
    Spectrum_signal = np.loadtxt(file_signal)
    file_info_signal = path_simu + "/signal_info_DMmass{0:.0f}_bin500keV_PSD.txt".format(mass)
    info_signal = np.loadtxt(file_info_signal)

    """ Get Dark Matter mass from the info_signal file: """
    DM_mass = info_signal[10]

    spectrum_Signal_per_bin = Spectrum_signal[entry_min_E_cut: (entry_max_E_cut + 1)]

    # expected number of signal events in the energy window (float):
    S_true = np.sum(spectrum_Signal_per_bin)

    # Calculate PSD efficiency for this DM mass and append it ot PSD_eff_array:
    PSD_eff = float(S_true) / float(S_true_woPSD)
    PSD_eff_array.append(PSD_eff)

    # maximum value of signal events consistent with existing limits (assuming the 'new' 90 % upper limit for
    # annihilation cross-section of Super-K from paper "Dark matter-neutrino interactions through the lens of their
    # cosmological implications" (PhysRevD.97.075039 of Olivares-Del Campo), for the description and calculation see
    # limit_from_SuperK.py)
    # INFO-me: S_max is assumed from the limit on the annihilation cross-section of Super-K (see limit_from_SuperK.py)
    S_max = 60

    # Fraction of DM signal (np.array of float):
    fraction_Signal = spectrum_Signal_per_bin / S_true

    S_true = 0

    # load corresponding dataset (unit: events/bin) (background-only spectrum:
    Data = spectrum_DSNB_per_bin + spectrum_CCatmo_p_per_bin + spectrum_CCatmo_C12_per_bin + spectrum_Reactor_per_bin \
           + spectrum_NCatmo_per_bin

    # guess of the parameters (as guess, the total number of events from the simulated spectrum are used)
    # (np.array of float):
    parameter_guess = np.array([S_true, B_DSNB_true, B_CCatmo_p_true, B_CCatmo_C12_true, B_Reactor_true,
                                B_NCatmo_true])

    # bounds of the parameters (parameters have to be positive or zero) (tuple):
    bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

    # Minimize the negative log-likelihood function with the L-BFGS-B method for the defined bounds above:
    result = op.minimize(neg_ln_likelihood, parameter_guess,
                         args=(Data, fraction_Signal, fraction_DSNB, fraction_CCatmo_p, fraction_CCatmo_C12,
                               fraction_Reactor, fraction_NCatmo),
                         method='L-BFGS-B', bounds=bnds, options={'disp': None})

    # get the best-fit parameters from the minimization (float):
    (S_maxlikeli, B_dsnb_maxlikeli, B_ccatmo_p_maxlikeli, B_ccatmo_c12_maxlikeli, B_reactor_maxlikeli,
     B_ncatmo_maxlikeli) = result["x"]
    print("Signal best fit = {0:.3f}".format(S_maxlikeli))
    print("DSNB best fit = {0:.3f}".format(B_dsnb_maxlikeli))
    print("CCatmo on p best fit = {0:.3f}".format(B_ccatmo_p_maxlikeli))
    print("CCatmo on C12 best fit = {0:.3f}".format(B_ccatmo_c12_maxlikeli))
    print("Reactor best fit = {0:.3f}".format(B_reactor_maxlikeli))
    print("NCatmo best fit = {0:.3f}".format(B_ncatmo_maxlikeli))

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
    ndim, nwalkers = 6, 50
    P0 = [result["x"] + 10**(-4)*np.random.randn(ndim) for i in range(nwalkers)]

    """ Then, we can set up the sampler 
        EnsembleSampler: a generalized Ensemble sampler that uses 2 ensembles for parallelization.
        (The "a" parameter controls the step size, the default is a=2): """
    value_of_a = 3.0
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posteriorprob, a=value_of_a,
                                    args=(Data, fraction_Signal, fraction_DSNB, fraction_CCatmo_p, fraction_CCatmo_C12,
                                          fraction_Reactor, fraction_NCatmo))

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
    # INFO-me: the number of steps should be large (greater than around 10000) to get a reproducible result
    number_of_steps = 40000
    sampler.run_mcmc(pos, number_of_steps)

    """ The best way to see this is to look at the time series of the parameters in the chain. 
    The sampler object now has an attribute called chain that is an array with the shape 
    (nwalkers, number_of_steps, ndim) giving the parameter values for each walker at each step in the chain. 
    The figure below shows the positions of each walker as a function of the number of steps in the chain 
    (sampler.chain[:, :, 0] has shape (nwalkers, number_of_steps), so the steps as function of the walker, 
    sampler.chain[:, :, 0].T is the transpose of the array with shape (number_of_steps, nwalkers), so the walker as 
    function of steps): """
    fig, (ax1, ax2, ax3, ax6, ax4, ax5) = plt.subplots(ndim, 1, sharex='all')
    fig.set_size_inches(10, 10)
    ax1.plot(sampler.chain[:, :, 0].T, '-', color='k', alpha=0.3)
    ax1.axhline(y=S_true, color='b', linestyle='--')
    ax1.set_ylabel('$S$')
    ax2.plot(sampler.chain[:, :, 1].T, '-', color='k', alpha=0.3)
    ax2.axhline(y=B_DSNB_true, color='b', linestyle='--')
    ax2.set_ylabel('$B_{DSNB}$')
    ax3.plot(sampler.chain[:, :, 2].T, '-', color='k', alpha=0.3)
    ax3.axhline(y=B_CCatmo_p_true, color='b', linestyle='--')
    ax3.set_ylabel('$B_{CCatmo} on p$')
    ax6.plot(sampler.chain[:, :, 3].T, '-', color='k', alpha=0.3)
    ax6.axhline(y=B_CCatmo_C12_true, color='b', linestyle='--')
    ax6.set_ylabel('$B_{CCatmo} on C12$')
    ax4.plot(sampler.chain[:, :, 4].T, '-', color='k', alpha=0.3)
    ax4.axhline(y=B_Reactor_true, color='b', linestyle='--', label='expected number of events')
    ax4.set_ylabel('$B_{Reactor}$')
    ax5.plot(sampler.chain[:, :, 5].T, '-', color='k', alpha=0.3)
    ax5.axhline(y=B_NCatmo_true, color='b', linestyle='--')
    ax5.set_ylabel('$B_{NCatmo}$')
    plt.legend()
    # plt.show()
    fig.savefig(path_analysis + '/DMmass{0:.0f}_chain_traces.png'.format(mass))
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
    # (The longer the auto-correlation time, the larger the number of the samples we must generate to obtain the
    # desired sampling of the posterior PDF) (You should run the sampler for a few (e.g. 10) auto-correlation times.
    # After that, you are almost completely sure to have independent samples from the posterior PDF).
    # Estimate the autocorrelation time for each dimension (np.array of float, dimension=ndim):
    try:
        # estimate integrated autocorrelation time, c = the minimum number of autocorrelation times needed to trust
        # the estimate (default: 10) (np.array of float):
        autocorr_time = sampler.get_autocorr_time(c=10)
        # print("auto-correlation time = {0}".format(autocorr_time))
        # calculate the mean of autocorr_time (float):
        mean_autocorr_time = np.mean(autocorr_time)
        print("mean of autocorrelation time = {0}".format(mean_autocorr_time))
    except emcee.autocorr.AutocorrError:
        # if there is an emcee.autocorr.AutocorrError, set the autocorrelation time to 1001001 (-> this means that there
        # was an error):
        # You get AutoCorrError, if the autocorrelation time can't be reliably estimated from the chain.
        # This normally means that the chain is too short -> increase number of steps!
        mean_autocorr_time = 1001001
        print("emcee.autocorr.AutocorrError")

    # append mean_autocorr_time to the array (np.array of float):
    mean_acor_array = np.append(mean_acor_array, mean_autocorr_time)

    """ flatten the chain so that we have a flat list of samples: """
    # the chain is a three-dimensional array of shape (number walkers nwalkers, number of steps, dimensions ndim),
    # e.g (200, 3000, 6)).
    # The 'flatchain' function flattens the chain along the zeroth (nwalkers) axis and you get a two dimensional
    # array of shape (nwalkers*steps, ndim), so e.g. (200*3000, 6) = (600000, 6)):
    samples = sampler.flatchain

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
    # print("integral = {0}".format(integral_hist_S))

    # plt.step(bin_middle_S, hist_S, 'x')
    # plt.xticks(np.arange(0, 20, 0.5))
    # plt.xlabel("S")
    # plt.ylabel("counts")
    # plt.title("p(S) from MCMC sampling for one dataset")
    # plt.show()

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
    # append value to array:
    S_90_array.append(S_90)
    print("S_90 = {0:.3f}".format(S_90))

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
    print("mode of DSNB_sample distribution = {0:.3f}".format(DSNB_mode))

    """ Calculate the mode of the CCatmo_sample on protons distribution: """
    # get the sample-chain of the atmo. CC background contribution (np.array of float):
    CCatmo_p_sample = samples[:, 2]
    # put CCatmo_sample in a histogram (2 np.arrays of float):
    hist_CCatmo_p, bin_edges_CCatmo_p = np.histogram(CCatmo_p_sample, bins='auto', range=(0, CCatmo_p_sample.max()))
    # get the index of the bin, where hist_CCatmo is maximal (integer):
    index_CCatmo_p = np.argmax(hist_CCatmo_p)
    # get the value of the left edge of the bin for index_CCatmo from above (float):
    value_left_edge_CCatmo_p = bin_edges_CCatmo_p[index_CCatmo_p]
    # get the value of the right edge of the bin for index_CCatmo from above (float):
    value_right_edge_CCatmo_p = bin_edges_CCatmo_p[index_CCatmo_p+1]
    # calculate the mode of the CCatmo_sample, therefore calculate the mean of value_left_edge and value_right_edge to
    # get the value in the middle of the bin (float):
    CCatmo_p_mode = (value_left_edge_CCatmo_p + value_right_edge_CCatmo_p) / 2
    print("mode of CCatmo_sample on proton distribution = {0:.3f}".format(CCatmo_p_mode))

    """ Calculate the mode of the CCatmo_sample on C12 distribution: """
    # get the sample-chain of the atmo. CC background contribution (np.array of float):
    CCatmo_c12_sample = samples[:, 3]
    # put CCatmo_sample in a histogram (2 np.arrays of float):
    hist_CCatmo_c12, bin_edges_CCatmo_c12 = np.histogram(CCatmo_c12_sample, bins='auto',
                                                         range=(0, CCatmo_c12_sample.max()))
    # get the index of the bin, where hist_CCatmo is maximal (integer):
    index_CCatmo_c12 = np.argmax(hist_CCatmo_c12)
    # get the value of the left edge of the bin for index_CCatmo from above (float):
    value_left_edge_CCatmo_c12 = bin_edges_CCatmo_c12[index_CCatmo_c12]
    # get the value of the right edge of the bin for index_CCatmo from above (float):
    value_right_edge_CCatmo_c12 = bin_edges_CCatmo_c12[index_CCatmo_c12 + 1]
    # calculate the mode of the CCatmo_sample, therefore calculate the mean of value_left_edge and value_right_edge to
    # get the value in the middle of the bin (float):
    CCatmo_c12_mode = (value_left_edge_CCatmo_c12 + value_right_edge_CCatmo_c12) / 2
    print("mode of CCatmo_sample on C12 distribution = {0:.3f}".format(CCatmo_c12_mode))

    """ Calculate the mode of the Reactor_sample distribution: """
    # get the sample-chain of the reactor background contribution (np.array of float):
    Reactor_sample = samples[:, 4]
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
    print("mode of Reactor_sample distribution = {0:.3f}".format(Reactor_mode))

    """ Calculate the mode of the NCatmo_sample distribution: """
    # get the sample-chain of the atmo. NC background contribution (np.array of float):
    NCatmo_sample = samples[:, 4]
    # put NCatmo_sample in a histogram (2 np.arrays of float):
    hist_NCatmo, bin_edges_NCatmo = np.histogram(NCatmo_sample, bins='auto', range=(0, NCatmo_sample.max()))
    # get the index of the bin, where hist_NCatmo is maximal (integer):
    index_NCatmo = np.argmax(hist_NCatmo)
    # get the value of the left edge of the bin for index_NCatmo from above (float):
    value_left_edge_NCatmo = bin_edges_NCatmo[index_NCatmo]
    # get the value of the right edge of the bin for index_NCatmo from above (float):
    value_right_edge_NCatmo = bin_edges_NCatmo[index_NCatmo+1]
    # calculate the mode of the NCatmo_sample, therefore calculate the mean of value_left_edge and value_right_edge to
    # get the value in the middle of the bin (float):
    NCatmo_mode = (value_left_edge_NCatmo + value_right_edge_NCatmo) / 2
    print("mode of NCatmo_sample distribution = {0:.3f}".format(NCatmo_mode))

    """ Now that we have this list of samples, let’s make one of the most useful plots you can make with your MCMC 
        results: a corner plot. Generate a corner plot is as simple as: """
    # NOTE: the quantile(0.5) is equal to the np.median() and is equal to np.percentile(50) -> NOT equal to np.mean()
    fig1 = corner.corner(samples,
                         labels=["$S$", "$B_{DSNB}$", "$B_{CCatmo,p}$", "$B_{CCatmo,^{12}C}$", "$B_{reactor}$",
                                 "$B_{NCatmo}$"],
                         truths=[S_true, B_DSNB_true, B_CCatmo_p_true, B_CCatmo_C12_true, B_Reactor_true,
                                 B_NCatmo_true],
                         truth_color='b',
                         labels_args={"fontsize": 80})
    # plt.show(fig1)
    # save figure:
    fig1.savefig(path_analysis + "/DMmass{0:.0f}_fitresult.png".format(mass))
    plt.close(fig1)

    """ Clear the chain and lnprobability array. Also reset the bookkeeping parameters: """
    sampler.reset()

    # save the output of the analysis of the dataset to txt-file:
    np.savetxt(path_analysis + '/DMmass{0:.0f}_mcmc_analysis.txt'.format(mass),
               np.array([S_mode, S_90, DSNB_mode, CCatmo_p_mode, CCatmo_c12_mode, Reactor_mode, NCatmo_mode,
                         S_maxlikeli, B_dsnb_maxlikeli, B_ccatmo_p_maxlikeli, B_ccatmo_c12_maxlikeli,
                         B_reactor_maxlikeli, B_ncatmo_maxlikeli]),
               fmt='%4.5f',
               header='Results of the MCMC analysis for DM mass {0:.0f} MeV to the expected spectrum'
                      '(analyzed with analyze_spectra_v7_wo_datasets.py, {1}):\n'
                      'Results of the analysis:\n'
                      'mode of the number of signal events,\n'
                      '90% upper limit of the number of signal events,\n'
                      'mode of the number of DSNB background events,\n'
                      'mode of the number of atmospheric CC background events on protons,\n'
                      'mode of the number of atmospheric CC background events on C12,\n'
                      'mode of the number of reactor background events,\n'
                      'mode of the number of atmospheric NC background events,\n'
                      'best-fit parameter for the number of signal events,\n'
                      'best-fit parameter for the number of DSNB background events,\n'
                      'best-fit parameter for the number of atmo. CC background events on protons,\n'
                      'best-fit parameter for the number of atmo. CC background events on C12,\n'
                      'best-fit parameter for the number of reactor background events\n'
                      'best-fit parameter for the number of atmo. NC background events,\n'
               .format(mass, now))


plt.plot(mass_DM, S_90_array)
plt.show()

# Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
# mass of "mass" MeV in cm**2 (float)
J_avg = 5
# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536

N_target = 1.450000000e+33
time_year = 10
time_s = time_year * 3.156 * 10 ** 7
epsilon_IBD = 0.67005
epsilon_mu_veto = 0.9717
sigma_anni_natural = 3 * 10 ** (-26)

limit_anni_90_array = []
limit_flux_90_array = []


for index in range(len(mass_DM)):
    limit_anni_90 = limit_annihilation_crosssection_v2(S_90_array[index], mass_DM[index], J_avg, N_target, time_s,
                                                       epsilon_IBD, epsilon_mu_veto, PSD_eff_array[index],
                                                       MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_anni_90_array.append(limit_anni_90)

    limit_flux_90 = limit_neutrino_flux_v2(S_90_array[index], mass_DM[index], N_target, time_s,
                                                      epsilon_IBD, epsilon_mu_veto, PSD_eff_array[index],
                                                      MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_flux_90_array.append(limit_flux_90)

limit_anni_90_array = np.asarray(limit_anni_90_array)
limit_flux_90_array = np.asarray(limit_flux_90_array)

""" Save values of limit_sigma_anni to txt file: """
np.savetxt(path_analysis + "/limit_annihilation_JUNO_wo_datasets.txt", limit_anni_90_array, fmt='%.5e',
           header="90 % upper limit of the DM annihilation cross-section from JUNO experiment in cm^3/2:\n"
                  "Values calculated with script analyze_spectra_v7_wo_datasets.py;\n"
                  "0 datasets have been analyzed, background-only spectrum is used as data;\n"
                  "Output in folder {0};\n"
                  "Spectra used for the analysis:\n"
                  "signal_DMmass()_bin500keV_PSD.txt,\n"
                  "{7},\n"
                  "{8},\n"
                  "{9},\n"
                  "{10},\n"
                  "{11};\n"
                  "DM masses in MeV = {1};\n"
                  "{2} years of data taking, number of free protons = {3};\n"
                  "IBD efficiency = {4}, muon veto efficiency = {5};\n"
                  "canonical value J_avg = {6}:"
           .format(path_analysis, mass_DM, time_year, N_target, epsilon_IBD, epsilon_mu_veto, J_avg, file_DSNB,
                   file_reactor, file_CCatmo_p, file_CCatmo_C12, file_NCatmo))

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way, obtained from Super-K Data.
    New Super-K limit of 2853 days = 7.82 years of data (the old limit from 0710.5420 was for only 1496 days of data).
    (they have used the canonical value J_avg = 5, the results are digitized from figure 1, on page 4 of the paper 
    'Dark matter-neutrino interactions through the lens of their cosmological implications', PhysRevD.97.075039)
    The digitized data is saved in "/home/astro/blum/PhD/paper/MeV_DM/SuperK_limit_new.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_SuperK = np.array([10.5384, 10.9966, 10.9966, 11.2257, 11.9129, 12.8293, 14.433, 15.1203, 16.2658, 16.953,
                           17.4112, 18.0985, 18.5567, 19.244, 20.3895, 21.5349, 22.9095, 23.8259, 24.9714, 25.8877,
                           27.2623, 28.1787, 28.866, 30.2405, 32.5315, 33.9061, 35.7388, 37.5716, 39.4044, 41.4662,
                           43.5281, 45.3608, 47.1936, 49.2554, 50.4009, 51.7755, 53.3792, 54.9828, 57.9611, 60.9393,
                           62.543, 64.3757, 66.8958, 69.874, 72.6231, 75.3723, 78.5796, 81.7869, 85.2234, 88.6598,
                           91.4089, 94.1581, 98.5109, 100.802])
# 90% limit of the self-annihilation cross-section in cm^3/s (np.array of float):
sigma_anni_SuperK = np.array([2.50761E-22, 2.20703E-22, 1.90157E-22, 1.38191E-22, 4.19682E-23, 8.15221E-24, 1.5502E-24,
                              8.18698E-25, 4.32374E-25, 3.3493E-25, 3.01122E-25, 2.70727E-25, 2.59447E-25, 2.59447E-25,
                              2.76551E-25, 2.94782E-25, 3.01122E-25, 2.88575E-25, 2.70727E-25, 2.53984E-25, 2.48636E-25,
                              2.53984E-25, 2.70727E-25, 3.14215E-25, 4.51174E-25, 5.23647E-25, 5.82438E-25, 6.07762E-25,
                              5.94965E-25, 5.94965E-25, 6.34188E-25, 7.51892E-25, 9.30201E-25, 1.10284E-24, 1.17555E-24,
                              1.17555E-24, 1.1508E-24, 1.07962E-24, 9.10614E-25, 7.51892E-25, 6.90537E-25, 6.61763E-25,
                              6.75997E-25, 7.3606E-25, 8.36308E-25, 9.10614E-25, 9.30201E-25, 8.91441E-25, 8.54296E-25,
                              8.7267E-25, 9.70646E-25, 1.1508E-24, 1.72424E-24, 2.22589E-24])

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way expected for 
    Hyper-Kamiokande after 10 years.
    (they have used the canonical value J_avg = 5, the results are digitized from figure 1, on page 2 of the paper 
    'Implications of a Dark Matter-Neutrino Coupling at Hyper–Kamiokande', Arxiv:1805.09830)
    The digitized data is saved in "/home/astro/blum/PhD/paper/MeV_DM/HyperK_limit_no_Gd.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_HyperK = np.array([11.4202, 11.4898, 11.5711, 11.6313, 11.6313, 11.7241, 11.7859, 11.9793, 12.0118, 12.1336,
                           12.2554, 12.4642, 12.4642, 12.6034, 12.7426, 12.7774, 12.8795, 13.0142, 13.0906, 13.2298,
                           13.4386, 13.6473, 13.7865, 13.9953, 14.1323, 14.2737, 14.4129, 14.5544, 14.6936, 14.7771,
                           14.9719, 15.1785, 15.4337, 15.6657, 16.5727, 17.1665, 17.8641, 19.0992, 20.8513, 22.2884,
                           23.9136, 25.4049, 26.8199, 28.4739, 30.005, 31.5361, 33.0673, 34.5984, 36.1922, 37.6607,
                           39.1919, 40.723, 42.3168, 43.8201, 45.3164, 46.8476, 48.4251, 49.7794, 52.0071, 54.2249,
                           55.7561, 57.1908, 59.418, 60.9394, 62.4374, 63.9686, 65.4997, 67.0309, 68.562, 70.0931,
                           71.6243, 73.1554, 74.7534, 76.281, 77.782, 79.3132, 80.8443, 82.3423, 83.8734, 85.4046,
                           86.9357, 88.4234, 89.998, 91.5292, 93.123, 94.5915, 96.0961, 97.6537, 99.1849, 100.716])

# 90 % limit of the self-annihilation cross-section in cm^3/s (array of float):
sigma_anni_HyperK = np.array([2.81969E-23, 2.3984E-23, 2.07369E-23, 1.79792E-23, 1.5389E-23, 1.34158E-23, 1.08796E-23,
                              9.07639E-24, 7.67308E-24, 6.07612E-24, 5.10954E-24, 4.29904E-24, 3.72832E-24, 3.1529E-24,
                              2.75079E-24, 2.42671E-24, 2.07014E-24, 1.76387E-24, 1.53958E-24, 1.34482E-24,
                              1.07845E-24, 8.76268E-25, 7.47593E-25, 6.49867E-25, 5.60194E-25, 4.82895E-25,
                              4.18958E-25, 3.64793E-25, 3.08801E-25, 2.6732E-25, 2.2808E-25, 2.00699E-25, 1.5469E-25,
                              1.32772E-25, 6.85055E-26, 5.56656E-26, 4.55841E-26, 3.74336E-26, 3.91269E-26,
                              4.31199E-26, 4.78795E-26, 5.27168E-26, 5.75659E-26, 6.33309E-26, 6.96262E-26,
                              7.63859E-26, 8.27672E-26, 8.99002E-26, 9.75563E-26, 1.05156E-25, 1.12134E-25, 1.207E-25,
                              1.28126E-25, 1.34863E-25, 1.40667E-25, 1.42365E-25, 1.41514E-25, 1.35923E-25,
                              1.23903E-25, 1.07922E-25, 9.82287E-26, 8.90453E-26, 8.01206E-26, 7.52633E-26,
                              7.20222E-26, 7.08082E-26, 7.08747E-26, 7.22607E-26, 7.50706E-26, 7.95367E-26,
                              8.48912E-26, 9.1121E-26, 1.00294E-25, 1.12329E-25, 1.24761E-25, 1.38262E-25,
                              1.54204E-25, 1.7043E-25, 1.80407E-25, 1.87869E-25, 1.90304E-25, 1.90679E-25, 1.92784E-25,
                              1.96469E-25, 2.02367E-25, 2.13703E-25, 2.27027E-25, 2.43049E-25, 2.62985E-25,
                              2.81874E-25])

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way obtained from KamLAND data 
    for 2343 days = 6.42 years of data taking.
    (they have used the canonical value J_avg = 5, the results are digitized from figure 5, on page 19 of the paper 
    'Search for extraterrestrial antineutrino sources with the KamLAND detector', Arxiv:1105.3516)
    The digitized data is saved in "/home/astro/blum/PhD/paper/KamLand/limit_KamLAND.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_Kamland = np.array([10.321, 11.8765, 12.4691, 13.284, 13.8025, 14.2469, 14.7654, 15.358, 15.358, 15.6543,
                            15.9506, 16.6173, 17.8025, 18.4691, 19.2099, 19.9506, 20.6173, 21.358, 22.0988, 22.3951,
                            22.6173, 23.0617, 23.4321, 23.9506, 24.6914, 25.4321, 26.0988, 26.8395, 27.5802, 28.321,
                            29.8765])

# 90 % limit of the self-annihilation cross-section in cm^3/s (array of float):
sigma_anni_Kamland = np.array([1.18275E-24, 9.19504E-25, 1.34141E-24, 1.95691E-24, 2.21943E-24, 2.36361E-24,
                               2.29039E-24, 1.89629E-24, 2.01948E-24, 1.89629E-24, 1.72545E-24, 1.42856E-24,
                               1.04285E-24, 9.48901E-25, 8.91017E-25, 9.79238E-25, 1.22056E-24, 1.52136E-24,
                               1.72545E-24, 1.67199E-24, 1.57E-24, 1.29986E-24, 1.04285E-24, 9.19504E-25, 8.91017E-25,
                               9.79238E-25, 1.1106E-24, 1.18275E-24, 1.14611E-24, 9.79238E-25, 3.55838E-24])

""" Semi-log. plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO, Super-K, Hyper-K and 
KamLAND: """
# maximum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_max = 10 ** (-22)
# minimum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_min = 10 ** (-26)

h3 = plt.figure(3, figsize=(10, 6))
plt.semilogy(mass_DM, limit_anni_90_array, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
             label='90% C.L. limit simulated for JUNO (10 years)')
plt.fill_between(mass_DM, limit_anni_90_array, y_max, facecolor='red', alpha=0.4)
plt.semilogy(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black', linewidth=2.0,
             label="90% C.L. limit from Super-K data (7.82 years)")
plt.semilogy(DM_mass_HyperK, sigma_anni_HyperK, linestyle=":", color='black', linewidth=2.0,
             label="90% C.L. limit simulated for Hyper-K (10 years)")
plt.semilogy(DM_mass_Kamland, sigma_anni_Kamland, linestyle="-", color='black', linewidth=2.0,
             label="90% C.L. limit from KamLAND data (6.42 years)")
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='$<\\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$')
plt.fill_between(DM_mass_SuperK, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(10, np.max(DM_mass_SuperK))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$<\\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
plt.title("90% upper limit on the total DM self-annihilation cross-section \nfrom the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()


y_max_flux = 10
y_min_flux = 0.01
h2 = plt.figure(2, figsize=(10, 6))
plt.semilogy(mass_DM, limit_flux_90_array, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
             label='90% C.L. limit simulated for JUNO (10 years)')
plt.fill_between(mass_DM, limit_flux_90_array, y_max_flux, facecolor='red', alpha=0.4)
plt.xlim(10, 100)
plt.ylim(y_min_flux, y_max_flux)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$\\phi_{90}$ in $1/(cm^2 s)$", fontsize=13)
plt.title("90% upper limit of the electron anti-neutrino flux from DM self-annihilation\nin the entire Milky Way",
          fontsize=15)
plt.legend(fontsize=12)
plt.grid()


plt.show()


