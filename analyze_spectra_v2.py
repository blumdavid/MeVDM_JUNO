""" Script to statistically analyze a large number of dataset-spectra with the simulation-spectrum.

    Version 2:  based on the GERDA paper 'A bayesian approach to the analysis of sparely populated spectra,
                Calculating the sensitivity of the GERDA experiment' by Caldwell and Kröninger (2006)

    Dataset (virtual experiments) are generated with gen_dataset_v1.py
    Simulated spectra are generated with gen_spectrum_v2.py

"""

# import of the necessary packages:
import datetime
import time
import numpy as np
# from scipy.optimize import minimize
from scipy.special import factorial
from scipy import integrate
# from scipy import stats
from matplotlib import pyplot

# TODO-me: improve the performance of the script -> scipy.LowLevelCallable!!

""" Set boolean value to define, if plot of p_S_spectrum is displayed: """
DISPLAY_PLOT = False

""" Set boolean value to define, if the result of the analysis are saved: """
# save the results and the plot of function p_S_spectrum (if data should be saved, SAVE_DATA must be True):
SAVE_DATA = True

""" set the path of the output folder: """
path_output = "dataset_output_20"

""" set the path of the folder, where the datasets were saved: """
path_dataset = path_output + "/datasets"

""" set the path of the folder, where the results of the analysis should be saved: """
path_analysis = path_output + "/analysis_integrate"

""" Go through every dataset and perform the analysis described in the GERDA paper: """
# define the first dataset, which will be analyzed (file: Dataset_dataset_start) (integer):
dataset_start = 953
# define the last dataset, which will be analyzed (file: Dataset_dataset_stop) (integer):
dataset_stop = 1000

""" Load information about the generation of the datasets from file (np.array of float): """
# TODO: Check, if info-files have the same parameter:
info_dataset = np.loadtxt(path_dataset + "/info_dataset_1_to_5.txt")
# get the bin-width of the visible energy in MeV from the info-file (float):
interval_E_visible = info_dataset[0]
# get minimum of the visible energy in MeV from info-file (float):
min_E_visible = info_dataset[1]
# get maximum of the visible energy in MeV from info-file (float):
max_E_visible = info_dataset[2]

""" Load simulated spectra in events/MeV from file (np.array of float): """
file_signal = "gen_spectrum_v2/signal_DMmass20_bin100keV.txt"
Spectrum_signal = np.loadtxt(file_signal)
file_info_signal = "gen_spectrum_v2/signal_info_DMmass20_bin100keV.txt"
info_signal = np.loadtxt(file_info_signal)
file_DSNB = "gen_spectrum_v2/DSNB_EmeanNuXbar22_bin100keV.txt"
Spectrum_DSNB = np.loadtxt(file_DSNB)
file_reactor = "gen_spectrum_v2/Reactor_NH_power36_bin100keV.txt"
Spectrum_reactor = np.loadtxt(file_reactor)
file_CCatmo = "gen_spectrum_v2/CCatmo_Osc1_bin100keV.txt"
Spectrum_CCatmo = np.loadtxt(file_CCatmo)


""" Define functions: """


def p_spectrum_sb(b_dsnb, b_ccatmo, b_reactor, s, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor,
                  dataset):
    """
    equation 11 of the GERDA paper.
    probability to observe the measured spectrum given by S,B (in case H_bar is true, signal and background)

    Assumption: the fluctuations in the bins of the spectrum are uncorrelated

    :param b_dsnb: number of expected DSNB background events (variable of the function) (float)
    :param b_ccatmo: number of expected CC atmospheric background events (variable of the function) (float)
    :param b_reactor: number of expected reactor background events (variable of the function) (float)
    :param s: number of expected signal events (variable of the function) (float)
    :param fraction_signal: normalized shapes of the signal spectra * bin-width for each bin, equivalent to the number
    of signal events per bin from the theoretical spectrum (np.array of float)
    :param fraction_dsnb: normalized shapes of the DSNB background spectra * bin-width for each bin, equivalent to the
    number of DSNB background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_ccatmo: normalized shapes of the CC atmo. background spectra * bin-width for each bin, equivalent
    to the number of CC atmo. background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_reactor: normalized shapes of the reactor background spectra * bin-width for each bin, equivalent
    to the number of reactor background events per bin from the theoretical spectrum (np.array of float)
    :param dataset: 'observed' number of events for each bin from the dataset ('observed' spectrum) (np.array of float)

    :return: conditional probability to obtain the measured spectrum given the parameters s, b_dsnb, b_ccatmo, b_reactor
    (float)
    """
    # lambda value of equation 9 of the Gerda paper (np.array of float):
    lamb = s * fraction_signal + b_dsnb * fraction_dsnb + b_ccatmo * fraction_ccatmo + b_reactor * fraction_reactor
    # value of the equation (np.array of float):
    factor = lamb ** dataset / factorial(dataset, exact=True) * np.exp(-lamb)
    # product over the value factor (float):
    result = np.prod(factor)

    return result


def p_spectrum_b(b_dsnb, b_ccatmo, b_reactor, fraction_dsnb, fraction_ccatmo, fraction_reactor, dataset):
    """
    equation 10 of the GERDA paper.
    probability to observe the measured spectrum given by B (in case H is true, only background)
    Here s = 0.

    Assumption: the fluctuations in the bins of the spectrum are uncorrelated

    :param b_dsnb: number of expected DSNB background events (variable of the function) (float)
    :param b_ccatmo: number of expected CC atmospheric background events (variable of the function) (float)
    :param b_reactor: number of expected reactor background events (variable of the function) (float)
    :param fraction_dsnb: normalized shapes of the DSNB background spectra * bin-width for each bin, equivalent to the
    number of DSNB background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_ccatmo: normalized shapes of the CC atmo. background spectra * bin-width for each bin, equivalent
    to the number of CC atmo. background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_reactor: normalized shapes of the reactor background spectra * bin-width for each bin, equivalent
    to the number of reactor background events per bin from the theoretical spectrum (np.array of float)
    :param dataset: 'observed' number of events for each bin from the dataset ('observed' spectrum) (np.array of float)

    :return: conditional probability to obtain the measured spectrum given the parameters b_dsnb, b_ccatmo, b_reactor
    (float)
    """
    # lambda value of equation 9 of the Gerda paper (np.array of float):
    lamb = b_dsnb * fraction_dsnb + b_ccatmo * fraction_ccatmo + b_reactor * fraction_reactor
    # value of the equation (np.array of float):
    factor = lamb ** dataset / factorial(dataset, exact=True) * np.exp(-lamb)
    # product over the value factor (float):
    result = np.prod(factor)

    return result


def p_spectrum_h(p_0_b_dsnb, b_dsnb_min, b_dsnb_max, p_0_b_ccatmo, b_ccatmo_min, b_ccatmo_max, p_0_b_reactor,
                 b_reactor_min, b_reactor_max, fraction_dsnb, fraction_ccatmo, fraction_reactor, dataset):
    """
    equation 6 of the GERDA paper
    conditional probabilities to find the observed spectrum given the hypothesis H (only background) is true or not true

    :param p_0_b_dsnb: prior probabilities for the number of expected DSNB background events (float)
    :param b_dsnb_min: lower integration limit for the DSNB background (float)
    :param b_dsnb_max: upper integration limit for the DSNB background (float)
    :param p_0_b_ccatmo: prior probabilities for the number of expected CC atmospheric background events (float)
    :param b_ccatmo_min: lower integration limit for the CC atmospheric background (float)
    :param b_ccatmo_max: upper integration limit for the CC atmospheric background (float)
    :param p_0_b_reactor: prior probabilities for the number of expected reactor background events (float)
    :param b_reactor_min: lower integration limit for the reactor background (float)
    :param b_reactor_max: upper integration limit for the reactor background (float)
    :param fraction_dsnb: normalized shapes of the DSNB background spectra * bin-width for each bin, equivalent to the
    number of DSNB background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_ccatmo: normalized shapes of the CC atmo. background spectra * bin-width for each bin, equivalent
    to the number of CC atmo. background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_reactor: normalized shapes of the reactor background spectra * bin-width for each bin, equivalent
    to the number of reactor background events per bin from the theoretical spectrum (np.array of float)
    :param dataset: 'observed' number of events for each bin from the dataset ('observed' spectrum) (np.array of float)

    :return: conditional probability to find the observed spectrum given the hypothesis H is true or not true (float)
    """
    # calculate integral of p_spectrum_b over b_dsnb, b_ccatmo and b_reactor (np.array of two floats):
    integral = integrate.nquad(p_spectrum_b, [[b_dsnb_min, b_dsnb_max], [b_ccatmo_min, b_ccatmo_max],
                                              [b_reactor_min, b_reactor_max]],
                               args=(fraction_dsnb, fraction_ccatmo, fraction_reactor, dataset),
                               full_output=True)
    # print(integral)
    # multiply the integral with the prior probabilities (float):
    result = integral[0] * p_0_b_dsnb * p_0_b_ccatmo * p_0_b_reactor

    return result


def p_spectrum_hbar(p_0_b_dsnb, b_dsnb_min, b_dsnb_max, p_0_b_ccatmo, b_ccatmo_min,
                    b_ccatmo_max, p_0_b_reactor, b_reactor_min, b_reactor_max, p_0_s, s_min, s_max,
                    fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor, dataset):
    """
    equation 7 of the GERDA paper
    conditional probabilities to find the observed spectrum given the hypothesis Hbar (signal and background) is true or
    not true

    :param p_0_b_dsnb: prior probability for the number of expected DSNB background events (float)
    :param b_dsnb_min: lower integration limit for the DSNB background (float)
    :param b_dsnb_max: upper integration limit for the DSNB background (float)
    :param p_0_b_ccatmo: prior probabilities for the number of expected CC atmospheric background events (float)
    :param b_ccatmo_min: lower integration limit for the CC atmospheric background (float)
    :param b_ccatmo_max: upper integration limit for the CC atmospheric background (float)
    :param p_0_b_reactor: prior probabilities for the number of expected reactor background events (float)
    :param b_reactor_min: lower integration limit for the reactor background (float)
    :param b_reactor_max: upper integration limit for the reactor background (float)
    :param p_0_s: prior probability for the number of expected signal events (float)
    :param s_min: lower integration limit for the signal events (float)
    :param s_max: upper integration limit for the signal events (float)
    :param fraction_signal: normalized shapes of the signal spectra * bin-width for each bin, equivalent to the number
    of signal events per bin from the theoretical spectrum (np.array of float)
    :param fraction_dsnb: normalized shapes of the DSNB background spectra * bin-width for each bin, equivalent to the
    number of DSNB background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_ccatmo: normalized shapes of the CC atmo. background spectra * bin-width for each bin, equivalent
    to the number of CC atmo. background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_reactor: normalized shapes of the reactor background spectra * bin-width for each bin, equivalent
    to the number of reactor background events per bin from the theoretical spectrum (np.array of float)
    :param dataset: 'observed' number of events for each bin from the dataset ('observed' spectrum) (np.array of float)

    :return: conditional probability to find the observed spectrum given the hypothesis Hbar is true or not true (float)
    """
    # calculate integral over log_p_spectrum_sb over s, b_dsnb, b_ccatmo and b_reactor (float):
    integral = integrate.nquad(p_spectrum_sb, [[b_dsnb_min, b_dsnb_max], [b_ccatmo_min, b_ccatmo_max],
                                               [b_reactor_min, b_reactor_max], [s_min, s_max]],
                               args=(fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor, dataset),
                               full_output=True)
    # print(integral)
    # multiply the integral with the prior probabilities (float):
    result = integral[0] * p_0_s * p_0_b_dsnb * p_0_b_ccatmo * p_0_b_reactor

    return result


def p_h_spectrum(p_spectrum_h_value, p_0_h, p_spectrum_hbar_value, p_0_hbar):
    """
    equation 3 of the GERDA paper
    conditional probability for the hypothesis H (only background) to be true or not, given the measured
    spectrum/dataset

    :param p_spectrum_h_value: conditional probability to find the observed spectrum given the hypothesis H is true or
     not true, return value of the function p_spectrum_h() (float)
    :param p_0_h: prior probability for H (float)
    :param p_spectrum_hbar_value: conditional probability to find the observed spectrum given the hypothesis Hbar
    is true or not true, return value of the function p_spectrum_hbar() (float)
    :param p_0_hbar: prior probability of Hbar (float)

    :return: conditional probability for the hypothesis H to be true or not (float)
    """
    # calculate p_spectrum (equation 5) (float):
    p_spectrum = p_spectrum_h_value * p_0_h + p_spectrum_hbar_value * p_0_hbar
    # calculate p_h_spectrum from equation 3 (float):
    result = p_spectrum_h_value * p_0_h / p_spectrum

    return result


def p_hbar_spectrum(p_spectrum_h_value, p_0_h, p_spectrum_hbar_value, p_0_hbar):
    """
    equation 4 of the GERDA paper
    conditional probability for the hypothesis Hbar (signal + background) to be true or not, given the measured
    spectrum/dataset

    :param p_spectrum_h_value: conditional probability to find the observed spectrum given the hypothesis H is true or
     not true, return value of the function p_spectrum_h() (float)
    :param p_0_h: prior probability for H (float)
    :param p_spectrum_hbar_value: conditional probability to find the observed spectrum given the hypothesis Hbar
    is true or not true, return value of the function p_spectrum_hbar() (float)
    :param p_0_hbar: prior probability of Hbar (float)

    :return: conditional probability for the hypothesis H to be true or not (float)
    """
    # calculate p_spectrum (equation 5) (float):
    p_spectrum = p_spectrum_h_value * p_0_h + p_spectrum_hbar_value * p_0_hbar
    # calculate p_h_spectrum from equation 4 (float):
    result = p_spectrum_hbar_value * p_0_hbar / p_spectrum

    return result


def p_s_spectrum(s_array, p_0_s, p_0_b_dsnb, b_dsnb_min, b_dsnb_max, p_0_b_ccatmo, b_ccatmo_min, b_ccatmo_max,
                 p_0_b_reactor, b_reactor_min, b_reactor_max,
                 p_spectrum_hbar_value, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor, dataset):
    """
    equation 13 of the paper
    To estimate the signal contribution the probability p_SB_spectrum is marginalized with respect to B_DSNB, B_CCatmo,
    B_Reactor
    p_SB_spectrum: probability that the 'observed' spectrum can be explained by the set of parameters S and B_DSNB,
    B_CCatmo, B_Reactor

    :param s_array: array which corresponds to the values of the number of signal events S (np.array of float)
    :param p_0_s: prior probability for the number of expected signal events (float)
    :param p_0_b_dsnb: prior probability for the number of expected DSNB background events (float)
    :param b_dsnb_min: lower integration limit for the DSNB background (float)
    :param b_dsnb_max: upper integration limit for the DSNB background (float)
    :param p_0_b_ccatmo: prior probabilities for the number of expected CC atmospheric background events (float)
    :param b_ccatmo_min: lower integration limit for the CC atmospheric background (float)
    :param b_ccatmo_max: upper integration limit for the CC atmospheric background (float)
    :param p_0_b_reactor: prior probabilities for the number of expected reactor background events (float)
    :param b_reactor_min: lower integration limit for the reactor background (float)
    :param b_reactor_max: upper integration limit for the reactor background (float)
    :param p_spectrum_hbar_value: conditional probability to find the observed spectrum given the hypothesis Hbar
    is true or not true, return value of the function p_spectrum_hbar() (float)
    :param fraction_signal: normalized shapes of the signal spectra * bin-width for each bin, equivalent to the number
    of signal events per bin from the theoretical spectrum (np.array of float)
    :param fraction_dsnb: normalized shapes of the DSNB background spectra * bin-width for each bin, equivalent to the
    number of DSNB background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_ccatmo: normalized shapes of the CC atmo. background spectra * bin-width for each bin, equivalent
    to the number of CC atmo. background events per bin from the theoretical spectrum (np.array of float)
    :param fraction_reactor: normalized shapes of the reactor background spectra * bin-width for each bin, equivalent
    to the number of reactor background events per bin from the theoretical spectrum (np.array of float)
    :param dataset: 'observed' number of events for each bin from the dataset ('observed' spectrum) (np.array of float)

    :return: signal contribution of the probability p_SB_spectrum (np.array of float)
    """
    # calculate the integral of p_sb_spectrum over B for every entry in the S_array:
    # (you get p_s_spectrum as function of S_array)
    # preallocate the array (np.array of ones with len(S_array)):
    integral_p_sb_spectrum = np.ones(len(s_array))

    for index in enumerate(s_array):

        # integrate the function p_spectrum_sb() over the backgrounds for one fix value of S (np.array with 2 floats):
        p_s_spectrum_value = integrate.nquad(p_spectrum_sb, [[b_dsnb_min, b_dsnb_max], [b_ccatmo_min, b_ccatmo_max],
                                                             [b_reactor_min, b_reactor_max]],
                                             args=(s_array[index[0]], fraction_signal, fraction_dsnb, fraction_ccatmo,
                                                   fraction_reactor, dataset),
                                             full_output=True)
        # print(p_s_spectrum_value)
        # Overwrite the value of the integral with the result value (np.array of floats):
        integral_p_sb_spectrum[index[0]] = p_s_spectrum_value[0]

    # calculate the numerator (Zaehler) of equation 13 and equation 12 (np.array of float):
    numerator = integral_p_sb_spectrum * p_0_s * p_0_b_dsnb * p_0_b_ccatmo * p_0_b_reactor
    # calculate the result of equation 13 (the denominator of equation 13 is exactly the value of p_spectrum_hbar,
    # which can be calculated with function p_spectrum_hbar() (np.array of float):
    result = numerator / p_spectrum_hbar_value

    return result


""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

""" Get Dark Matter mass from the info_signal file: """
DM_mass = info_signal[9]
print("Dark matter mass = {0:.2f} MeV".format(DM_mass))

""" Define the energy window, where spectrum of virtual experiment and simulated spectrum is analyzed
    (from min_E_cut in MeV to max_E_cut in MeV): """
min_E_cut = DM_mass - 5
max_E_cut = DM_mass + 5
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

# Fraction of DM signal (np.array of float):
N_events_signal = np.sum(spectrum_Signal_per_bin)
print("Number of signal events in the energy window: {0}".format(N_events_signal))
fraction_Signal = spectrum_Signal_per_bin / N_events_signal

# Fraction of DSNB background (np.array of float):
N_events_DSNB = np.sum(spectrum_DSNB_per_bin)
print("Number of DSNB events in the energy window: {0}".format(N_events_DSNB))
fraction_DSNB = spectrum_DSNB_per_bin / N_events_DSNB

# Fraction of CCatmo background (np.array of float)::
N_events_CCatmo = np.sum(spectrum_CCatmo_per_bin)
print("Number of CCatmo events in the energy window: {0}".format(N_events_CCatmo))
fraction_CCatmo = spectrum_CCatmo_per_bin / N_events_CCatmo

# Fraction of reactor background (np.array of float)::
N_events_reactor = np.sum(spectrum_Reactor_per_bin)
print("Number of reactor events in the energy window: {0}".format(N_events_reactor))
fraction_Reactor = spectrum_Reactor_per_bin / N_events_reactor

""" define the prior probabilities for the hypothesis H and Hbar and for the signal and background contributions: """
# Set minimum and maximum values of signal and backgrounds, which are consistent with existing limits:
# These value 'define' the prior probabilities of the number of signal and backgrounds
# TODO-me: use a more general approach!
S_min = 0
S_max = 10
B_DSNB_min = 0
B_DSNB_max = N_events_DSNB + N_events_DSNB
B_CCatmo_min = N_events_CCatmo - N_events_CCatmo/2
B_CCatmo_max = N_events_CCatmo + N_events_CCatmo/2
B_reactor_min = N_events_reactor - N_events_reactor/2
B_reactor_max = N_events_reactor + N_events_reactor/2

# define prior probabilities of the signal and background contributions:
# TODO-me: use a more general approach!
p_0_S = 1 / (S_max - S_min)
p_0_B_DSNB = 1 / (B_DSNB_max - B_DSNB_min)
p_0_B_CCatmo = 1 / (B_CCatmo_max - B_CCatmo_min)
p_0_B_reactor = 1 / (B_reactor_max - B_reactor_min)

# define array which corresponds to the values of S (np.array of float):
# TODO-me: change the interval width -> smaller steps?
interval_S_array = 0.1
S_array = np.arange(S_min, S_max, interval_S_array)

# Prior probability of the hypothesis H:
p_0_H = 0.5
# Prior probability of the hypothesis H_bar:
p_0_Hbar = 0.5

# loop over the Datasets (from dataset_start to dataset_stop):
for number in np.arange(dataset_start, dataset_stop+1, 1):
    print("Analyze dataset_{0:d}".format(number))
    # load corresponding dataset (unit: events/bin) (np.array of float):
    Dataset = np.loadtxt(path_dataset + "/Dataset_{0:d}.txt".format(number))
    # dataset in the 'interesting' energy range from min_E_cut to max_E_cut
    # (you have to take (entry_max+1) to get the array, that includes max_E_cut):
    Dataset = Dataset[entry_min_E_cut: (entry_max_E_cut + 1)]

    # calculate p_spectrum_H with function p_spectrum_h() from equation 6 (integral of p_spectrum_b over backgrounds)
    # (float):
    start_time_p_H = time.time()
    p_spectrum_H = p_spectrum_h(p_0_B_DSNB, B_DSNB_min, B_DSNB_max, p_0_B_CCatmo, B_CCatmo_min, B_CCatmo_max,
                                p_0_B_reactor, B_reactor_min, B_reactor_max,
                                fraction_DSNB, fraction_CCatmo, fraction_Reactor, Dataset)
    print("time elapsed for p_spectrum_H: {:.2f}s".format(time.time() - start_time_p_H))

    # calculate p_spectrum_Hbar with function p_spectrum_hbar() from equation 7 (integral of p_spectrum_sb over signal
    # and backgrounds) (float):
    start_time_p_Hbar = time.time()
    p_spectrum_Hbar = p_spectrum_hbar(p_0_B_DSNB, B_DSNB_min, B_DSNB_max, p_0_B_CCatmo, B_CCatmo_min, B_CCatmo_max,
                                      p_0_B_reactor, B_reactor_min, B_reactor_max, p_0_S, S_min, S_max,
                                      fraction_Signal, fraction_DSNB, fraction_CCatmo, fraction_Reactor, Dataset)
    print("time elapsed for p_spectrum_Hbar: {:.2f}s".format(time.time() - start_time_p_Hbar))

    """ calculate p_H_spectrum with function p_h_spectrum() from equation 3: """
    # (float number:)
    p_H_spectrum = p_h_spectrum(p_spectrum_H, p_0_H, p_spectrum_Hbar, p_0_Hbar)

    """ calculate p_Hbar_spectrum with function p_hbar_spectrum() from equation 4: """
    # (float number:)
    p_Hbar_spectrum = p_hbar_spectrum(p_spectrum_H, p_0_H, p_spectrum_Hbar, p_0_Hbar)

    # calculate the decimal logarithm of p_H_spectrum (float):
    log_p_H_spectrum = np.log10(p_H_spectrum)
    print("logarithm of p_H_spectrum = {0:.6f}".format(log_p_H_spectrum))
    # calculate the decimal logarithm of p_Hbar_spectrum (float):
    log_p_Hbar_spectrum = np.log10(p_Hbar_spectrum)
    print("logarithm of p_Hbar_spectrum = {0:.6f}".format(log_p_Hbar_spectrum))

    # TODO: check if the sum is 1
    print("sum of p_H_spectrum and p_Hbar_spectrum = {0:.6f}".format(p_H_spectrum + p_Hbar_spectrum))

    """ Calculate p_S_spectrum as function of the S_array (integral of p_SB_spectrum over backgrounds): """
    # (np.array of float:)
    start_time_p_S = time.time()
    p_S_spectrum = p_s_spectrum(S_array, p_0_S, p_0_B_DSNB, B_DSNB_min, B_DSNB_max, p_0_B_CCatmo, B_CCatmo_min,
                                B_CCatmo_max, p_0_B_reactor, B_reactor_min, B_reactor_max,
                                p_spectrum_Hbar, fraction_Signal, fraction_DSNB, fraction_CCatmo, fraction_Reactor,
                                Dataset)
    print("time elapsed for p_S_spectrum: {:.2f}s".format(time.time() - start_time_p_S))

    # Calculate the integral of p_S_spectrum over the whole array S_array as test:
    total_integral_p_S_spectrum = np.trapz(p_S_spectrum, S_array)
    print("integral of p_S_spectrum over S = {0:.6f}".format(total_integral_p_S_spectrum))

    """ Calculate the mode (most probable value) of S from p_S_spectrum: """
    # get the index, where p_S_spectrum has its maximum value (integer):
    index_mode = np.argmax(p_S_spectrum)
    # take the value of S_array for the calculated index_max (float):
    mode_of_S = S_array[index_mode]
    print("Mode of S from p_S_spectrum = {0:.6f}".format(mode_of_S))

    """ Calculate the 90 percent limit of the probability distribution p_S_spectrum: """
    # define the 90 percent limit of the probability (integral of p_S_spectrum from 0 to S_90 = 0.9),
    # defines the break condition of the while loop (float):
    limit_p = 0.9
    # preallocate the index number in S_array (integer):
    index_S_array = 1
    # preallocate the integral of p_S_spectrum from 0 to a specific index number of S_array (float):
    integral_p_S_spectrum = 0
    while integral_p_S_spectrum < limit_p:
        # calculate the integral of p_S_spectrum from S_array[0] to S_array[index_S_array] (float):
        integral_p_S_spectrum = np.trapz(p_S_spectrum[0:(index_S_array+1)], S_array[0:(index_S_array+1)])
        # increment the index number by 1 (integer):
        index_S_array = index_S_array + 1
    # take the value of S_array for the calculated index = (index_S_array - 1) (float):
    upperlimit_S_90 = S_array[index_S_array - 1]
    print("90 percent limit of S. S_90 = {0:.4f}".format(upperlimit_S_90))

    # TODO-me: Calculation of probability level is missing

    """ Display the function p_S_spectrum in a plot: """
    h1 = pyplot.figure(1)
    # TODO-me: Insert the values of the probability levels
    pyplot.plot(S_array, p_S_spectrum, label='mode of p(S|spectrum): S_mode = {0:.3f}\n'
                                             '90% limit of p(S|spectrum): S_90 = {1:.3f}'
                .format(mode_of_S, upperlimit_S_90))
    pyplot.xlabel('number of signal events S')
    pyplot.ylabel('p(S|spectrum)')
    pyplot.title('p(S|spectrum) as function of S for Dataset_{0:d}'.format(number))
    pyplot.ylim(ymin=0)
    pyplot.grid()
    pyplot.legend()
    if DISPLAY_PLOT:
        pyplot.show()
    else:
        pyplot.savefig(path_analysis + '/Dataset{0:d}_p_S_spectrum.png'.format(number))
        pyplot.close(h1)

    # to save the data set values, SAVE_DATA must be True:
    if SAVE_DATA:
        # Save the results of the analysis in a txt file:
        # TODO-me: Insert the values of the probability levels
        np.savetxt(path_analysis + '/Dataset{0:d}_analysis.txt'.format(number),
                   np.array([log_p_H_spectrum, log_p_Hbar_spectrum, mode_of_S, upperlimit_S_90]), fmt='%4.5f',
                   header='Results of the analysis of virtual experiment (Dataset_{3:d}) to the expected spectrum '
                          '(analyzed with analyze_spectra_v2.py, {0}):\n'
                          'General information of the analysis are saved in info_analysis_{1:d}_{2:d}.txt\n'
                          'Values of the function p_S_spectrum are saved in Dataset{3:d}_p_S_spectrum.txt\n'
                          'Results of the analysis:\n'
                          'log_p_H_spectrum,\n'
                          'log_p_Hbar_spectrum,\n'
                          'mode of S,\n'
                          '90 percent upper limit of S:'
                   .format(now, dataset_start, dataset_stop, number))

        # save the p_S_spectrum array in a txt file:
        np.savetxt(path_analysis + '/Dataset{0:d}_p_S_spectrum.txt'.format(number),
                   p_S_spectrum, fmt='%4.5f',
                   header='Probability function p_S_spectrum of the analysis of virtual experiment (Dataset_{0:d}) '
                          'to the expected spectrum:\n'
                          '(analyzed with analyze_spectra_v2.py, {1}):\n'
                          'General information of the analysis are saved in info_analysis_{2:d}_{3:d}.txt\n'
                          'Result values of the analysis are save in Dataset{0:d}_analysis.txt\n'
                          'Array of number of signal events:\n'
                          'S_min = {4:.3f}, S_max = {5:.3f}, interval_S = {6:.3f}.\n'
                          'p_S_spectrum as function of S_array:'
                   .format(number, now, dataset_start, dataset_stop, S_min, S_max, interval_S_array))

if SAVE_DATA:
    # save the general information of the analysis in txt file:
    np.savetxt(path_analysis + '/info_analysis_{0:d}_{1:d}.txt'.format(dataset_start, dataset_stop),
               np.array([min_E_cut, max_E_cut, interval_E_visible, N_events_signal, S_min, S_max, interval_S_array,
                         p_0_S, N_events_DSNB, B_DSNB_min, B_DSNB_max, p_0_B_DSNB, N_events_CCatmo, B_CCatmo_min,
                         B_CCatmo_max, p_0_B_CCatmo, N_events_reactor, B_reactor_min, B_reactor_max, p_0_B_reactor,
                         p_0_H, p_0_Hbar]),
               fmt='%4.5f',
               header='General information about the analysis of virtual experiment to the expected spectra '
                      '(analyzed with analyze_spectra_v2.py, {0}):\n'
                      'Analyzed datasets: Dataset_{1:d}.txt to Dataset_{2:d}.txt\n'
                      'Values below:\n'
                      'minimum E_cut in MeV, maximum E_cut in MeV, interval-width of the E_cut array in MeV,\n'
                      'Expected number of signal events in this energy range,\n'
                      'S_min (lower integration bound), S_max (higher integration bound), interval-width S_array,\n'
                      '(S_array is defined as np.arange(S_min, S_max, interval_S_array)),\n'
                      'prior probability for the number of expected signal events (flat distribution),\n'
                      'Expected number of DSNB background events in this energy range,\n'
                      'B_DSNB_min (lower integration bound), B_DSNB_max (higher integration bound),\n'
                      'prior probability for the number of DSNB background events (flat distribution),\n'
                      'Expected number of CC atmospheric background events in this energy range,\n'
                      'B_CCatmo_min (lower integration bound), B_CCatmo_max (higher integration bound),\n'
                      'prior probability for the number of CC atmo. background events (flat distribution),\n'
                      'Expected number of reactor background events in this energy range,\n'
                      'B_reactor_min (lower integration bound), B_reactor_max (higher integration bound),\n'
                      'prior probability of the number of reactor background events (flat distribution),\n'
                      'prior probability of the hypothesis H (background only), '
                      'prior probability of the hypothesis Hbar (signal and background):'
               .format(now, dataset_start, dataset_stop))
