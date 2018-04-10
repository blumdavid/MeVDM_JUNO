""" Script to statistically analyze a large number of dataset-spectra with the simulation-spectrum.
    Dataset (virtual experiments) are generated with gen_dataset_v1.py
    Simulated spectra are generated with gen_spectrum_v2.py

    Data of the virtual experiments is fitted to the simulated spectrum with a negative log-likelihood fit
    Best-fit parameters of the total number of signal and backgrounds are saved to txt-file
"""

# import of the necessary packages:
# import datetime
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
from matplotlib import pyplot

""" Set boolean values to define, which plots are displayed: """
DISPLAY_HISTO = True
DISPLAY_LOG_LIKELIHOOD = False

""" Set boolean value to define, if the datasets are saved: """
# save the dataset (if data should be saved, SAVE_DATA must be True):
SAVE_DATA = True
# set the path of the folder, where the datasets are saved and where the best-fit parameter should be saved:
path = "dataset_output_20"

""" Go through every dataset and perform a log-likelihood fit to the parameters, which describe the total number of 
    DM signal events, DSNB background events, atmospheric CC background events and reactor background events: """
# define the first dataset, which will be analyzed (file: Dataset_dataset_start) (integer):
dataset_start = 1
# define the last dataset, which will be analyzed (file: Dataset_dataset_stop) (integer):
dataset_stop = 1000

# Load information about the generation of the datasets from file (np.array of float):
# TODO: Check, if info-files have the same parameter:
info_dataset = np.loadtxt(path + "/info_dataset_1_to_5.txt")
# get the bin-width of the visible energy in MeV from the info-file (float):
interval_E_visible = info_dataset[0]
# get minimum of the visible energy in MeV from info-file (float):
min_E_visible = info_dataset[1]
# get maximum of the visible energy in MeV from info-file (float):
max_E_visible = info_dataset[2]

""" Define the energy window, where spectrum of virtual experiment and simulated spectrum is analyzed
    (from min_E_cut in MeV to max_E_cut in MeV): """
min_E_cut = min_E_visible
max_E_cut = max_E_visible

# Load simulated spectra in events/MeV from file (np.array of float):
file_signal = "gen_spectrum_v2/signal_DMmass20_bin100keV.txt"
Spectrum_signal = np.loadtxt(file_signal)
file_DSNB = "gen_spectrum_v2/DSNB_EmeanNuXbar22_bin100keV.txt"
Spectrum_DSNB = np.loadtxt(file_DSNB)
file_reactor = "gen_spectrum_v2/Reactor_NH_power36_bin100keV.txt"
Spectrum_reactor = np.loadtxt(file_reactor)
file_CCatmo = "gen_spectrum_v2/CCatmo_Osc1_bin100keV.txt"
Spectrum_CCatmo = np.loadtxt(file_CCatmo)

# Simulated spectra in events/bin (multiply with interval_E_visible):
spectrum_signal_per_bin = Spectrum_signal * interval_E_visible
spectrum_DSNB_per_bin = Spectrum_DSNB * interval_E_visible
spectrum_CCatmo_per_bin = Spectrum_CCatmo * interval_E_visible
spectrum_reactor_per_bin = Spectrum_reactor * interval_E_visible

# Fraction of DM signal (np.array of float):
N_event_signal = np.sum(spectrum_signal_per_bin)
fraction_signal = spectrum_signal_per_bin / N_event_signal
# Fraction of DSNB background (np.array of float)::
N_event_DSNB = np.sum(spectrum_DSNB_per_bin)
fraction_DSNB = spectrum_DSNB_per_bin / N_event_DSNB
# Fraction of CCatmo background (np.array of float)::
N_event_CCatmo = np.sum(spectrum_CCatmo_per_bin)
fraction_CCatmo = spectrum_CCatmo_per_bin / N_event_CCatmo
# Fraction of reactor background (np.array of float)::
N_event_reactor = np.sum(spectrum_reactor_per_bin)
fraction_reactor = spectrum_reactor_per_bin / N_event_reactor

# preallocate the array, where the best-fit-parameter of the total number of events are saved:
# total number of DM signal events (best-fit-parameter) (np.array of float):
number_signal = np.array([])
# 90 percent limit (probability) of signal, upper limit on the number of signal events (np.array of float):
limit_90_signal = np.array([])
# total number of DSNB background events (best-fit-parameter) (np.array of float):
number_DSNB = np.array([])
# total number of atmospheric CC background events (best-fit-parameter) (np.array of float):
number_CCatmo = np.array([])
# total number of reactor background events (best-fit-parameter) (np.array of float):
number_reactor = np.array([])

# loop over the Datasets (from dataset_start to dataset_stop):
for number in np.arange(dataset_start, dataset_stop+1, 1):
    print("Fit dataset_{0:d} to simulated spectrum".format(number))
    # load corresponding dataset (unit: events/bin) (np.array of float):
    dataset = np.loadtxt(path + "/Dataset_{0:d}.txt".format(number))

    # Define function, which calculates the negative-log-likelihood with the fractions defined above and the loaded
    # dataset:
    def log_likelihood(param):
        """
        function, which calculates the negative log-likelihood function for the given fit-parameters and fix-parameters.
        Fix-parameters are the fractions of DM signal, DSNB background, atmospheric CC background and reactor
        background, and the number of 'detected' events from the dataset.
        The function L is taken from arXiv:1208.0834 'A novel way of constraining WIMPs annihilation in the Sun: MeV
        neutrinos', page 15, equation (24).
        The negative-log-likelihood function is calculated as -ln(L).

        :param param: array of parameters that will be estimated. Correspond to the total number of events of DM signal,
        DSNB background, atmospheric CC background, reactor background (np.array of float)

        :return: neg-log-likelihood: returns the negative of the log-likelihood function for the input parameters
        """
        # one part of the neg-log-likelihood function (float):
        sum_1 = sum(np.log(factorial(dataset, exact=True)))
        # another part of the neg-log-likelihood function (float):
        sum_2 = sum(dataset * np.log(param[0]*fraction_signal + param[1]*fraction_DSNB +
                                     param[2]*fraction_CCatmo + param[3]*fraction_reactor))
        # define the neg-log-likelihood function depending on parameters param (float):
        neg_log_likelihood = param[0] + param[1] + param[2] + param[3] + sum_1 - sum_2

        return neg_log_likelihood

    # guess of the fit-parameters (as guess, the total number of events from the simulated spectrum are used)
    # (np.array of float):
    parameter = np.array([N_event_signal, N_event_DSNB, N_event_CCatmo, N_event_reactor])

    # bounds of the fitting parameters (all parameters are positive or zero):
    bnds = ((0, None), (0, None), (0, None), (0, None))

    # Minimize the negative log-likelihood function with the L-BFGS-B method for the defined bounds above and print the
    # convergence message. (result contains the optimization result as a OptimizeResult object):
    result = minimize(log_likelihood, parameter, method='L-BFGS-B', bounds=bnds, options={'disp': None})

    # Print Boolean flag, that indicates if the optimizer exited successfully:
    # print(result.success)

    """
    def integral_log_likelihood(alpha_90):
        integral_alpha90 = integrate.nquad(log_likelihood, [[0, alpha_90], [0, np.inf], [0, np.inf], [0, np.inf]])
        integral_total = integrate.nquad(log_likelihood, [[0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf]])

        func_to_solve = integral_alpha90[0] / integral_total[0] - 0.9

        return func_to_solve

    alpha_90_signal = fsolve(integral_log_likelihood, np.array([1]))

    limit_90_signal = np.append(limit_90_signal, alpha_90_signal)
    """

    """
    # define an array, that represents the variable alpha (np.array of float):
    alpha_array = np.arange(0, 50, 0.1)

    # calculate neg-log-likelihood as function of alpha and the fix parameter for the backgrounds from the fit:
    p_S_spectrum = np.array([])
    for index in alpha_array:
        p_S = log_likelihood([alpha_array[index], result.x[1], result.x[2], result.x[3]])
        p_S_spectrum = np.append(p_S_spectrum, p_S)

    print(p_S_spectrum)

    # Sum over all entries in p_S_spectrum, is used to normalize the function:
    p_S_total = np.sum(p_S_spectrum)
    print(p_S_total)

    # calculate the upper limit of events (90 percent):
    bin_index = alpha_array[0]
    sum_signal = 0
    while sum_signal < 0.9 * p_S_total:
        sum_signal = sum_signal + p_S_spectrum[bin_index]
        bin_index = bin_index + 1
    limit_90_signal = (alpha_array[1] - alpha_array[0]) * (bin_index - 1)
    print(limit_90_signal)
    """

    # TODO: best-fit-parameters are float number and not integers! -> is this a problem?
    # The solution of the optimization is given in result.x
    # append the best-fit-parameters to the arrays:
    number_signal = np.append(number_signal, result.x[0])
    number_DSNB = np.append(number_DSNB, result.x[1])
    number_CCatmo = np.append(number_CCatmo, result.x[2])
    number_reactor = np.append(number_reactor, result.x[3])

    # TODO: Error of the best-fit-parameters???
    # TODO: 90 percent confidence level for alpha (total number of DM signal events)?
    # TODO: How does 90 percent confidence level affect the annihilation cross section of Dark Matter?
    # TODO: Number of Datasets have to be increased! -> What is a good number?
    # TODO: Documentation of the Analysis!
    # TODO: Only use a small energy-window in the region of interest

""" Save the best-fit-parameters to file for each signal/backgrounds: """
if SAVE_DATA:
    # save best-fit parameters of total number of DM signal events to file:
    np.savetxt(path + "/Best_fit_signal.txt", number_signal, fmt='%4.5f',
               header="Values of the best-fit parameter of the total number of events of DM signal:\n"
                      "(fit is made by minimizing neg-log-likelihood-function with script analyze_spectra_v1.py)\n"
                      "Input simulated spectra:\n"
                      "{0},\n{1},\n{2},\n{3}\n"
                      "Input datasets:\n"
                      "in folder {4} from dataset {5:d} to {6:d}.\n"
                      "Energy window from {7:.2f} MeV to {8:.2f} MeV"
               .format(file_signal, file_DSNB, file_CCatmo, file_reactor, path, dataset_start, dataset_stop,
                       min_E_cut, max_E_cut))

    # save best-fit parameters of total number of DSNB background events to file:
    np.savetxt(path + "/Best_fit_DSNB.txt", number_DSNB, fmt='%4.5f',
               header="Values of the best-fit parameter of the total number of events of DSNB background:\n"
                      "(fit is made by minimizing neg-log-likelihood-function with script analyze_spectra_v1.py)\n"
                      "Input simulated spectra:\n"
                      "{0},\n{1},\n{2},\n{3}\n"
                      "Input datasets:\n"
                      "in folder {4} from dataset {5:d} to {6:d}.\n"
                      "Energy window from {7:.2f} MeV to {8:.2f} MeV."
               .format(file_signal, file_DSNB, file_CCatmo, file_reactor, path, dataset_start, dataset_stop,
                       min_E_cut, max_E_cut))

    # save best-fit parameters of total number of reactor background events to file:
    np.savetxt(path + "/Best_fit_reactor.txt", number_reactor, fmt='%4.5f',
               header="Values of the best-fit parameter of the total number of events of reactor background:\n"
                      "(fit is made by minimizing neg-log-likelihood-function with script analyze_spectra_v1.py)\n"
                      "Input simulated spectra:\n"
                      "{0},\n{1},\n{2},\n{3}\n"
                      "Input datasets:\n"
                      "in folder {4} from dataset {5:d} to {6:d}.\n"
                      "Energy window from {7:.2f} MeV to {8:.2f} MeV."
               .format(file_signal, file_DSNB, file_CCatmo, file_reactor, path, dataset_start, dataset_stop,
                       min_E_cut, max_E_cut))

    # save best-fit parameters of total number of CCatmo background events to file:
    np.savetxt(path + "/Best_fit_CCatmo.txt", number_CCatmo, fmt='%4.5f',
               header="Values of the best-fit parameter of the total number of events of CCatmo background:\n"
                      "(fit is made by minimizing neg-log-likelihood-function with script analyze_spectra_v1.py)\n"
                      "Input simulated spectra:\n"
                      "{0},\n{1},\n{2},\n{3}\n"
                      "Input datasets:\n"
                      "in folder {4} from dataset {5:d} to {6:d}.\n"
                      "Energy window from {7:.2f} MeV to {8:.2f} MeV."
               .format(file_signal, file_DSNB, file_CCatmo, file_reactor, path, dataset_start, dataset_stop,
                       min_E_cut, max_E_cut))

if DISPLAY_HISTO:
    # Display the best-fit-parameters in a histogram:
    # TODO: What is the correct binning?
    Bin_width = 1
    Bins = np.arange(0, 100, Bin_width)

    h1 = pyplot.figure(1)
    entries_signal, bins_signal, patches_signal = pyplot.hist(number_signal, bins=Bins, histtype='step', color='k',
                                                              label='best-fit-parameters of total number of DM signal '
                                                                    'events')
    # total number of entries in the histogram:
    total_number_signal = np.sum(entries_signal)
    # calculate the upper limit of events (90 percent):
    bin_index = 0
    sum_signal = 0
    while sum_signal < 0.9 * total_number_signal:
        sum_signal = sum_signal + entries_signal[bin_index]
        bin_index = bin_index + 1
    limit_90_signal = Bin_width * (bin_index - 1)
    print(limit_90_signal)

    pyplot.axvline(limit_90_signal, linestyle='--', color='k', label='90% upper limit on the number of signal events '
                                                                     '({0:.1f})'.format(limit_90_signal))

    entries_DSNB, bins_DSNB, patches_DSNB = pyplot.hist(number_DSNB, bins=Bins, histtype='step', color='g',
                                                        label='best-fit-parameters of total number of DSNB background '
                                                              'events')
    entries_CCatmo, bins_CCatmo, patches_CCatmo = pyplot.hist(number_CCatmo, bins=Bins, histtype='step', color='r',
                                                              label='best-fit-parameters of total number of atmo. CC '
                                                                    'background events')
    entries_reactor, bins_reactor, patches_reactor = pyplot.hist(number_reactor, bins=Bins, histtype='step', color='b',
                                                                 label='best-fit-parameters of total number of reactor '
                                                                       'background events')

    # TODO: Is it useful to fit a poisson function to the histogram of the best-fit-parameters? -> NO

    pyplot.xlabel('total number of events')
    pyplot.ylabel('counts')
    pyplot.title('Histogram of best-fit parameters of the total number of events for DM signal and backgrounds')
    pyplot.legend()
    pyplot.show()

if DISPLAY_LOG_LIKELIHOOD:
    # Calculate the neg-log-likelihood function and display it:
    y = np.arange(0, 80, 1)
    # neg-log-likelihood function for best-fit-parameters of DSNB, CCatmo and reactor as function of DM signal (alpha):
    alpha = np.array([])
    for index in np.arange(len(y)):
        alpha_1 = log_likelihood((y[index], result.x[1], result.x[2], result.x[3]))
        alpha = np.append(alpha, alpha_1)
    # neg-log-likelihood function for best-fit-parameters of DM signal, CCatmo and reactor as function of DSNB (beta):
    beta = np.array([])
    for index in np.arange(len(y)):
        beta_1 = log_likelihood((result.x[0], y[index], result.x[2], result.x[3]))
        beta = np.append(beta, beta_1)
    # neg-log-likelihood function for best-fit-parameters of DM signal, DSNB and reactor as function of CCatmo (gamma):
    gamma = np.array([])
    for index in np.arange(len(y)):
        gamma_1 = log_likelihood((result.x[0], result.x[1], y[index], result.x[3]))
        gamma = np.append(gamma, gamma_1)
    # neg-log-likelihood function for best-fit-parameters of DM signal, DSNB and CCatmo as function of reactor (delta):
    delta = np.array([])
    for index in np.arange(len(y)):
        delta_1 = log_likelihood((result.x[0], result.x[1], result.x[2], y[index]))
        delta = np.append(delta, delta_1)

    h2 = pyplot.figure(2)
    pyplot.plot(y, alpha, label='neg-log-likelihood as function of alpha')
    pyplot.plot(y, beta, label='neg-log-likelihood as function of beta')
    pyplot.plot(y, gamma, label='neg-log-likelihood as function of gamma')
    pyplot.plot(y, delta, label='neg-log-likelihood as function of delta')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()
