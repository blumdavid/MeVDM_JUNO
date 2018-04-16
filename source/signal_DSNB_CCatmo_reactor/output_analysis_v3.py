""" Script to display and analyze the results of the MCMC analysis done with analyze_spectra_v4.py or
    analyze_spectra_v4_server.py!

    In analyze_spectra_v4.py the dataset and the simulated spectrum were analyzed with Markov Chain Monte Carlo (MCMC)
    sampling and the results of the analysis are saved in the files DatasetX_mcmc_analysis.txt

    In analyze_spectra_v4.py the mode of the total number of signal and background events are determined by MCMC
    sampling of the posterior probability

"""

import numpy as np
from matplotlib import pyplot as plt
import datetime

# TODO-me: also update the file: output_analysis_v3.py

""" Set boolean value to define, if the result of output_analysis_v3.py is saved: """
SAVE_DATA = True

""" Set the path of the folder, where the results of the analysis are saved: """
number_dataset_output = 20
path_dataset_output = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor/dataset_output_{0:d}"\
    .format(number_dataset_output)
path_analysis = path_dataset_output + "/analysis_mcmc"

""" Set the path of the file, where the information about the analysis is saved: """
# TODO: Check the file-path
file_info_analysis = path_analysis + "/info_mcmc_analysis_1_100.txt"

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
# mode of the total number of signal events (np.array of float):
S_mode = np.array([])
# values of upper 90 percent limit of the number of signal events (np.array of float):
S_90_limit = np.array([])
# mode of the total number of DSNB background events (np.array of float):
DSNB_mode = np.array([])
# mode of the total number of CCatmo background events (np.array of float):
CCatmo_mode = np.array([])
# mode of the total number of reactor background events (np.array of float):
Reactor_mode = np.array([])

""" Read in the files, where the results of the analysis are saved and read the result-values: """
for number in np.arange(file_number_start, file_number_stop+1, 1):
    # load the file corresponding to Dataset{number}_mcmc_analysis.txt:
    result_analysis = np.loadtxt(path_analysis + "/Dataset{0:d}_mcmc_analysis.txt".format(number))
    # get value of mode of S (float):
    value_mode_S = result_analysis[0]
    # get value of 90 percent limit S (float):
    value_S_90_limit = result_analysis[1]
    # get value of mode of B_DSNB (float):
    value_mode_DSNB = result_analysis[2]
    # get value of mode of B_CCatmo (float):
    value_mode_CCatmo = result_analysis[3]
    # get value of mode of B_reactor (float):
    value_mode_Reactor = result_analysis[4]

    # Append the values to the arrays (np.array of float):
    S_mode = np.append(S_mode, value_mode_S)
    S_90_limit = np.append(S_90_limit, value_S_90_limit)
    DSNB_mode = np.append(DSNB_mode, value_mode_DSNB)
    CCatmo_mode = np.append(CCatmo_mode, value_mode_CCatmo)
    Reactor_mode = np.append(Reactor_mode, value_mode_Reactor)


""" Calculate the mean and probability interval: """
# calculate the mean and 16% and 84% confidence level of the array S_mean (float):
S_50 = np.mean(S_mode)
S_50_sigma = np.std(S_mode)
S_50_16, S_50_84 = np.percentile(S_mode, [16, 84])

# calculate the mean and 16% and 84% confidence level of the array S_90_limit (float):
S_90 = np.mean(S_90_limit)
S_90_sigma = np.std(S_90_limit)
S_90_16, S_90_84 = np.percentile(S_90_limit, [16, 84])

# calculate the mean and 16% and 84% confidence level of the array DSNB_mean (float):
DSNB_50 = np.mean(DSNB_mode)
DSNB_50_sigma = np.std(DSNB_mode)
DSNB_50_16, DSNB_50_84 = np.percentile(DSNB_mode, [16, 84])

# calculate the mean and 16% and 84% confidence level of the array CCatmo_mean (float):
CCatmo_50 = np.mean(CCatmo_mode)
CCatmo_50_sigma = np.std(CCatmo_mode)
CCatmo_50_16, CCatmo_50_84 = np.percentile(CCatmo_mode, [16, 84])

# calculate the mean and 16% and 84% confidence level of the array Reactor_mean (float):
Reactor_50 = np.mean(Reactor_mode)
Reactor_50_sigma = np.std(Reactor_mode)
Reactor_50_16, Reactor_50_84 = np.percentile(Reactor_mode, [16, 84])

""" Load the analysis information file to get the expected/true number of events in the energy window and to get the 
    energy window: """
# load the txt file (np.array of float):
information_analysis = np.loadtxt(file_info_analysis)
# get the assumed DM mass in MeV (float):
DM_mass = information_analysis[0]
# lower bound of the energy window in MeV (float):
lower_energy_bound = information_analysis[1]
# upper bound of the energy window in MeV (float):
upper_energy_bound = information_analysis[2]
# number of expected signal events in the energy window (float):
signal_expected = information_analysis[4]
# number of expected DSNB backgrounds events in the energy window (float):
DSNB_expected = information_analysis[6]
# number of expected atmospheric CC background events in the energy window (float):
CCatmo_expected = information_analysis[7]
# number of expected reactor background events in the energy window (float):
Reactor_expected = information_analysis[8]


if SAVE_DATA:
    np.savetxt(path_dataset_output + "/result_mcmc/result_dataset_output_{0:d}.txt".format(number_dataset_output),
               np.array([lower_energy_bound, upper_energy_bound, number_of_entries,
                         S_50, S_50_sigma, S_50_16, S_50_84,
                         S_90, S_90_sigma, S_90_16, S_90_84,
                         DSNB_50, DSNB_50_sigma, DSNB_50_16, DSNB_50_84,
                         CCatmo_50, CCatmo_50_sigma, CCatmo_50_16, CCatmo_50_84,
                         Reactor_50, Reactor_50_sigma, Reactor_50_16, Reactor_50_84]), fmt="%4.5f",
               header="Results of the analysis of the spectra in dataset_output_20 (with output_analysis_v3.py, {0}):\n"
                      "Analysis of Dataset_{1:d}.txt to Dataset_{2:d}.txt\n"
                      "Information to the values below:\n"
                      "Lower bound of the energy window in MeV, upper bound of the energy window in MeV\n"
                      "Number of datasets that were analyzed,\n"
                      "Mean of the observed number of signal events,\n"
                      "Standard deviation of the observed number of signal events,\n"
                      "16 % confidence level of the observed number of signal events,\n"
                      "84 % confidence level of the observed number of signal events,\n"
                      "Mean of the 90% probability limit of the observed number of signal events,\n"
                      "Standard deviation of the 90% probability limit of the observed number of signal events,\n"
                      "16 % confidence level of the 90% probability limit of the observed number of signal events,\n"
                      "84 % confidence level of the 90% probability limit of the observed number of signal events,\n"
                      "Mean of the observed number of DSNB background events,\n"
                      "Standard deviation of the observed number of DSNB background events,\n"
                      "16 % confidence level of the observed number of DSNB background events,\n"
                      "84 % confidence level of the observed number of DSNB background events,\n"
                      "Mean of the observed number of atmo. CC background events,\n"
                      "Standard deviation of the observed number of atmo. CC background events,\n"
                      "16 % confidence level of the observed number of atmo. CC background events,\n"
                      "84 % confidence level of the observed number of atmo. CC background events,\n"
                      "Mean of the observed number of Reactor background events,\n"
                      "Standard deviation of the observed number of Reactor background events,\n"
                      "16 % confidence level of the observed number of Reactor background events,\n"
                      "84 % confidence level of the observed number of Reactor background events:"
               .format(now, file_number_start, file_number_stop))

    # print message, that result data is saved in file:
    print("result data is saved in the file result_dataset_output_{0:d}.txt".format(number_dataset_output))

""" Display the results in histograms: """
# Display S_mean in histogram:
h1 = plt.figure(1)
# Bins1 = 'auto'
Bins1 = np.arange(0, np.max(S_mode)+0.1, 0.05)
n_S, bins1, patches1 = plt.hist(S_mode, bins=Bins1, histtype='step', color='b',
                                label='number of virt. experiments = {0:d},\n'
                                      'mean of the distribution = {1:.4f}'
                                .format(number_of_entries, S_50))
plt.axvline(signal_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(signal_expected), color='r')
# plt.xticks(np.arange(0, np.max(S_mode)+0.5, 0.5))
plt.xlabel("total number of observed signal events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of signal events from DM with mass={2:.1f}MeV in virtual experiments "
          "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()


# Display S_90_limit in histogram:
h2 = plt.figure(2)
# Bins2 = 'auto'
Bins2 = np.arange(2, np.max(S_90_limit)+0.1, 0.1)
n_limit_S, bins2, patches2 = plt.hist(S_90_limit, bins=Bins2, histtype='step', color='b',
                                      label='number of virt. experiments = {0:d},\n'
                                            'mean of the distribution = {1:.3f}'
                                      .format(number_of_entries, S_90))
plt.xticks(np.arange(2, np.max(S_90_limit)+0.5, 0.5))
plt.xlabel("90 percent limit of number of observed signal events S")
plt.ylabel("counts")
plt.title("Distribution of the 90 % upper limit of the signal contribution for DM with mass = {0:.1f} MeV"
          .format(DM_mass))
plt.legend()


# Display DSNB_mean in histogram:
h3 = plt.figure(3)
Bins3 = 'auto'
# Bins3 = np.arange(0, 25, 1)
n_DSNB, bins3, patches3 = plt.hist(DSNB_mode, bins=Bins3, histtype='step', color='b',
                                   label='number of virt. experiments = {0:d}'
                                   .format(number_of_entries))
plt.axvline(DSNB_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(DSNB_expected), color='r')
plt.axvline(DSNB_50, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(DSNB_50), color='b')
plt.axvline(DSNB_50_16, linestyle=':', label='16% probability limit = {0:.4f}'.format(DSNB_50_16), color='b')
plt.axvline(DSNB_50_84, linestyle='-.', label='84% probability limit = {0:.4f}'.format(DSNB_50_84), color='b')
plt.xlabel("total number of observed DSNB background events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of DSNB background events in virtual experiments "
          "(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()


# Display CCatmo_mean in histogram:
h4 = plt.figure(4)
Bins4 = 'auto'
# Bins4 = np.arange(0, 3, 0.1)
n_CCatmo, bins4, patches4 = plt.hist(CCatmo_mode, bins=Bins4, histtype='step', color='b',
                                     label='number of virt. experiments = {0:d}'
                                     .format(number_of_entries))
plt.axvline(CCatmo_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(CCatmo_expected), color='r')
plt.axvline(CCatmo_50, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(CCatmo_50), color='b')
plt.axvline(CCatmo_50_16, linestyle=':', label='16% probability limit = {0:.4f}'.format(CCatmo_50_16), color='b')
plt.axvline(CCatmo_50_84, linestyle='-.', label='84% probability limit = {0:.4f}'.format(CCatmo_50_84), color='b')
plt.xlabel("total number of observed atmo. CC background events events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of atmospheric CC background events in virtual experiments "
          "(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()


# Display Reactor_mean in histogram:
h5 = plt.figure(5)
Bins5 = 'auto'
# Bins5 = np.arange(0, 0.5, 0.01)
n_Reactor, bins5, patches5 = plt.hist(Reactor_mode, bins=Bins5, histtype='step', color='b',
                                      label='number of virt. experiments = {0:d}'
                                      .format(number_of_entries))
plt.axvline(Reactor_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(Reactor_expected), color='r')
plt.axvline(Reactor_50, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(Reactor_50), color='b')
plt.axvline(Reactor_50_16, linestyle=':', label='16% probability limit = {0:.4f}'.format(Reactor_50_16), color='b')
plt.axvline(Reactor_50_84, linestyle='-.', label='84% probability limit = {0:.4f}'.format(Reactor_50_84), color='b')
plt.xlabel("total number of observed reactor background events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of reactor background events in virtual experiments "
          "(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()

plt.show()
