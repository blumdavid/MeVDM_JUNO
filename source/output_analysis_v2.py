""" Script to display and analyze the results of the analysis done with analyze_spectra_v3.py.

    In analyze_spectra_v3.py the dataset and the simulated spectrum were analyzed and the result of the analysis
    are saved in the files DatasetX_analysis.txt

    In analyze_spectra_v3.py the total number of signal and background events are determined by minimizing the
    neg-log-likelihood function

"""

import numpy as np
from matplotlib import pyplot as plt
import datetime

# TODO-me: at the end check the script again

# TODO-me: Binning of the histograms
# TODO-me: consider the correlation of the fit-parameter
# TODO-me: is it better to calculate the standard deviation or the confidence interval of the distribution?
# INFO-me: 1 sigma level is equal to a 68 % confidence level (16% to 84% interval) for normal distribution
# INFO-me: -> also for these distributions????


""" Set boolean value to define, if the result of output_analysis_v1.py is saved: """
SAVE_DATA = True

""" Set the path of the folder, where the results of the analysis are saved: """
number_dataset_output = 1
path_dataset_output = "dataset_output_{0:d}".format(number_dataset_output)
path_analysis = path_dataset_output + "/analysis_fit"

""" Set the path of the file, where the information about the analysis is saved: """
file_info_analysis = path_analysis + "/info_analysis_1_10.txt"

# Set the number of the files, that should be read in:
file_number_start = 1
file_number_stop = 10000

""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

# calculate the number of files, that are read in (Equivalent to the number of entries in the result array) (integer):
number_of_entries = file_number_stop - file_number_start + 1

""" Preallocate the arrays, where the results of the analysis of the different datasets should be appended: """
# values of log_p_H_spectrum (np.array of float):
log_p_H_spectrum = np.array([])
# best-fit values of the total number of signal events (np.array of float):
bestfit_S = np.array([])
# values of upper 90 percent limit of the number of signal events (np.array of float):
limit_S_90 = np.array([])
# best-fit values of the total number of DSNB background events (np.array of float):
bestfit_DSNB = np.array([])
# best-fit values of the total number of CCatmo background events (np.array of float):
bestfit_CCatmo = np.array([])
# best-fit values of the total number of reactor background events (np.array of float):
bestfit_Reactor = np.array([])

""" Read in the files, where the results of the analysis are saved and read the result-values: """
for number in np.arange(file_number_start, file_number_stop+1, 1):
    # load the file corresponding to Dataset{number}_analysis.txt:
    result_analysis = np.loadtxt(path_analysis + "/Dataset{0:d}_analysis.txt".format(number))
    # get value of log_p_H_spectrum (float):
    value_log_p_H_spectrum = result_analysis[0]
    # get value of best-fit parameter of S (float):
    value_bestfit_S = result_analysis[1]
    # get value of 90 percent limit S (float):
    value_limit_S_90 = result_analysis[2]
    # get value of best-fit parameter for B_DSNB (float):
    value_bestfit_DSNB = result_analysis[3]
    # get value of best-fit parameter for B_CCatmo (float):
    value_bestfit_CCatmo = result_analysis[4]
    # get value of best-fit parameter for B_reactor (float):
    value_bestfit_Reactor = result_analysis[5]

    # Append the values to the arrays (np.array of float):
    log_p_H_spectrum = np.append(log_p_H_spectrum, value_log_p_H_spectrum)
    bestfit_S = np.append(bestfit_S, value_bestfit_S)
    limit_S_90 = np.append(limit_S_90, value_limit_S_90)
    bestfit_DSNB = np.append(bestfit_DSNB, value_bestfit_DSNB)
    bestfit_CCatmo = np.append(bestfit_CCatmo, value_bestfit_CCatmo)
    bestfit_Reactor = np.append(bestfit_Reactor, value_bestfit_Reactor)

# calculate the mean of log_p_H_spectrum (float):
mean_log_p_H_spectrum = np.mean(log_p_H_spectrum)
# calculate the mean and the standard deviation of the array bestfit_S (float):
mean_bestfit_S = np.mean(bestfit_S)
sigma_bestfit_S = np.std(bestfit_S)
# calculate the mean and the standard deviation of the array limit_S_90 (float):
mean_limit_S_90 = np.mean(limit_S_90)
sigma_limit_S_90 = np.std(limit_S_90)
# calculate the mean and the standard deviation of the array bestfit_DSNB (float):
mean_bestfit_DSNB = np.mean(bestfit_DSNB)
sigma_bestfit_DSNB = np.std(bestfit_DSNB)
# calculate the mean and the standard deviation of the array bestfit_CCatmo (float):
mean_bestfit_CCatmo = np.mean(bestfit_CCatmo)
sigma_bestfit_CCatmo = np.std(bestfit_CCatmo)
# calculate the mean and the standard deviation of the array bestfit_reactor (float):
mean_bestfit_Reactor = np.mean(bestfit_Reactor)
sigma_bestfit_Reactor = np.std(bestfit_Reactor)

""" Set the discovery criterion and calculate the number of elements, which pass this criterion: """
# Set the discovery criterion (float):
discovery_crit = 0.01/100
# calculate the decimal logarithm of the discovery criterion (float):
log_discovery_crit = np.log10(discovery_crit)
# calculate the number of elements of log_p_H_spectrum, that pass the discovery criterion
# (are smaller than -4=log(0.01%)) (log_p_H_spectrum < log_discovery_crit returns an array that consists of boolean
# variables True and False. With sum() all entries that are True are summed up)(float):
num_entries_discovery = (log_p_H_spectrum < log_discovery_crit).sum()

""" Set the criterion for evidence of Hbar and calculate the number of elements, which pass this criterion: """
# Set the criterion for evidence for Hbar (float):
evidence_crit = 1/100
# calculate the decimal logarithm of the criterion for evidence (float):
log_evidence_crit = np.log10(evidence_crit)
# calculate the number of elements of log_p_H_spectrum, that give evidence for Hbar ('Hinweis auf Entdeckung')
# (are smaller than -2=log(1%) (float):
num_entries_evidence = (log_p_H_spectrum < log_evidence_crit).sum()
# You have to subtract num_entries_discovery to get the number of elements of log_p_H_spectrum, that are between
# -4 and -2 (float)
num_entries_evidence = num_entries_evidence - num_entries_discovery

""" Calculate the number of elements in log_p_H_spectrum, which do not pass a criterion: """
num_entries_nodiscovery = number_of_entries - num_entries_evidence - num_entries_discovery

""" Load the analysis information file and get the expected number of events in the energy window and to get the 
    energy window: """
# load the txt file (np.array of float):
information_analysis = np.loadtxt(file_info_analysis)
# lower bound of the energy window in MeV (float):
lower_energy_bound = information_analysis[0]
# upper bound of the energy window in MeV (float):
upper_energy_bound = information_analysis[1]
# number of expected signal events in the energy window (float):
signal_expected = information_analysis[3]
# number of expected DSNB backgrounds events in the energy window (float):
DSNB_expected = information_analysis[8]
# number of expected atmospheric CC background events in the energy window (float):
CCatmo_expected = information_analysis[12]
# number of expected reactor background events in the energy window (float):
Reactor_expected = information_analysis[16]

""" Save the results in txt file: """
if SAVE_DATA:
    np.savetxt(path_dataset_output + "/result/result_dataset_output_{0:d}.txt".format(number_dataset_output),
               np.array([lower_energy_bound, upper_energy_bound, number_of_entries, mean_bestfit_S, sigma_bestfit_S,
                         mean_limit_S_90, sigma_limit_S_90,
                         mean_bestfit_DSNB, sigma_bestfit_DSNB, mean_bestfit_CCatmo, sigma_bestfit_CCatmo,
                         mean_bestfit_Reactor, sigma_bestfit_Reactor, mean_log_p_H_spectrum,
                         log_discovery_crit, num_entries_discovery, log_evidence_crit, num_entries_evidence,
                         num_entries_nodiscovery]), fmt="%4.5f",
               header="Results of the analysis of the spectra in dataset_output_20 (with output_analysis_v1.py, {0}):\n"
                      "Analysis of Dataset_{1:d}.txt to Dataset_{2:d}.txt\n"
                      "Information to the values below:\n"
                      "Lower bound of the energy window in MeV, upper bound of the energy window in MeV\n"
                      "Number of datasets that were analyzed,\n"
                      "Mean of the observed number of signal events,\n"
                      "Standard deviation of the observed number of signal events,\n"
                      "Mean of the 90% probability limit of the observed number of signal events,\n"
                      "Standard deviation of the 90% probability limit of the observed number of signal events,\n"
                      "Mean of the observed number of DSNB background events,\n"
                      "Standard deviation of the observed number of DSNB background events,\n"
                      "Mean of the observed number of atmo. CC background events,\n"
                      "Standard deviation of the observed number of atmo. CC background events,\n"
                      "Mean of the observed number of Reactor background events,\n"
                      "Standard deviation of the observed number of Reactor background events,\n"
                      "Mean of log_p_H_spectrum,\n"
                      "Decimal logarithm of the discovery criterion,\n"
                      "Number of datasets that claim a discovery,\n"
                      "Decimal logarithm of criterion for evidence,\n"
                      "Number of datasets that give evidence for a discovery,\n"
                      "Number of datasets that do not claim a discovery."
               .format(now, file_number_start, file_number_stop))

    # print message, that result data is saved in file:
    print("result data is saved in the file result_dataset_output_{0:d}.txt".format(number_dataset_output))

# Display log_p_H_spectrum in histogram:
h1 = plt.figure(1)
Bins1 = 'auto'
log_p_H, bins1, patches1 = plt.hist(log_p_H_spectrum, bins=Bins1, histtype='step', color='b',
                                    label='number of virt. experiments = {0:d},\n'
                                          'mean of the distribution = {4:.3f},\n'
                                          'number of virt. experiments that claim a discovery = {1:d},\n'
                                          'number of virt. experiments that give evidence for a discovery = {2:d}\n'
                                          'number of virt. experiments that do not claim a discovery = {3:d}'
                                    .format(number_of_entries, num_entries_discovery, num_entries_evidence,
                                            num_entries_nodiscovery, mean_log_p_H_spectrum))
plt.axvline(x=log_discovery_crit, color='r', linestyle='--', label='discovery criterion')
plt.axvline(x=log_evidence_crit, color='k', linestyle='--', label='criterion for evidence')
plt.xlabel("log[p(H|spectrum)]")
plt.ylabel("counts")
plt.title("Distribution of the conditional probability for the hypothesis,\nthat the spectra of virtual experiments "
          "are due to background only (in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound))
plt.legend(loc='upper left', framealpha=1)


# Display bestfit_S in histogram:
h2 = plt.figure(2)
# Bins2 = 'auto'
Bins2 = np.arange(0, 9, 0.1)
mode_S, bins2, patches2 = plt.hist(bestfit_S, bins=Bins2, histtype='step', color='b',
                                   label='number of virt. experiments = {0:d},\n'
                                         'mean of the distribution = {1:.3f},\n'
                                         'standard deviation of the distribution = {2:.3f}'
                                   .format(number_of_entries, mean_bestfit_S, sigma_bestfit_S))
plt.axvline(signal_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(signal_expected), color='r')
plt.xlabel("total number of observed signal events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of signal events in virtual experiments "
          "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)".format(lower_energy_bound, upper_energy_bound))
plt.legend()


# Display limit_S_90 in histogram:
h3 = plt.figure(3)
# Bins3 = 'auto'
Bins3 = np.arange(2, 11, 0.1)
limit_S, bins3, patches3 = plt.hist(limit_S_90, bins=Bins3, histtype='step', color='b',
                                    label='number of virt. experiments = {0:d},\n'
                                          'mean of the distribution = {1:.3f},\n'
                                          'standard deviation of the distribution = {2:.3f}'
                                    .format(number_of_entries, mean_limit_S_90, sigma_limit_S_90))
plt.xlabel("90 percent limit of number of signal events S")
plt.ylabel("counts")
plt.title("Distribution of the 90 % upper limit of the signal contribution")
plt.legend()


# Display bestfit_DSNB in histogram
h4 = plt.figure(4)
# Bins4 = 'auto'
Bins4 = np.arange(0, 25, 1)
mode_DSNB, bins4, patches4 = plt.hist(bestfit_DSNB, bins=Bins4, histtype='step', color='b',
                                      label='number of virt. experiments = {0:d},\n'
                                            'mean of the distribution = {1:.3f},\n'
                                            'standard deviation of the distribution = {2:.3f}'
                                      .format(number_of_entries, mean_bestfit_DSNB, sigma_bestfit_DSNB))
plt.axvline(DSNB_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(DSNB_expected), color='r')
plt.xlabel("total number of observed DSNB background events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of DSNB background events in virtual experiments "
          "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)".format(lower_energy_bound, upper_energy_bound))
plt.legend()


# Display bestfit_CCatmo in histogram:
h5 = plt.figure(5)
# Bins5 = 'auto'
Bins5 = np.arange(0, 3, 0.1)
mode_CCatmo, bins5, patches5 = plt.hist(bestfit_CCatmo, bins=Bins5, histtype='step', color='b',
                                        label='number of virt. experiments = {0:d},\n'
                                              'mean of the distribution = {1:.3f},\n'
                                              'standard deviation of the distribution = {2:.3f}'
                                        .format(number_of_entries, mean_bestfit_CCatmo, sigma_bestfit_CCatmo))
plt.axvline(CCatmo_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(CCatmo_expected), color='r')
plt.xlabel("total number of observed atmo. CC background events events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of atmospheric CC background events in virtual experiments "
          "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)".format(lower_energy_bound, upper_energy_bound))
plt.legend()


# Display bestfit_Reactor in histogram:
h6 = plt.figure(6)
# Bins6 = 'auto'
Bins6 = np.arange(0, 0.5, 0.01)
mode_Reactor, bins6, patches6 = plt.hist(bestfit_Reactor, bins=Bins6, histtype='step', color='b',
                                         label='number of virt. experiments = {0:d},\n'
                                               'mean of the distribution = {1:.3f},\n'
                                               'standard deviation of the distribution = {2:.3f}'
                                         .format(number_of_entries, mean_bestfit_Reactor, sigma_bestfit_Reactor))
plt.axvline(Reactor_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(Reactor_expected), color='r')
plt.xlabel("total number of observed reactor background events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of reactor background events in virtual experiments "
          "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)".format(lower_energy_bound, upper_energy_bound))
plt.legend()

plt.show()
