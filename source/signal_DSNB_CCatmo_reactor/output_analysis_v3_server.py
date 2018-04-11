""" Script to display and analyze the results of the MCMC analysis on the IHEP cluster in China
    done with analyze_spectra_v4_server2.py!

    Script is based on the output_analysis_v3.py Script, but changed a bit to be able to run it on the cluster.

    In analyze_spectra_v4_server2.py the dataset and the simulated spectrum were analyzed with
    Markov Chain Monte Carlo (MCMC) sampling and the results of the analysis are saved in the
    files DatasetX_mcmc_analysis.txt

    In analyze_spectra_v4_server2.py the mode of the total number of signal and background events are determined by
    MCMC sampling of the posterior probability

    give 7 arguments to the script:
    - sys.argv[0] name of the script = output_analysis_v3_server.py
    - sys.argv[1] Dark matter mass in MeV
    - sys.argv[2] directory of the correct folder = "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor"
    - sys.argv[3] dataset_output folder = "dataset_output_{DM_mass}"
    - sys.argv[4] number of datasets analyzed per job
    - sys.argv[5] dataset_start
    - sys.argv[6] dataset_stop
"""

import numpy as np
import datetime
import sys


""" Set boolean value to define, if the result of output_analysis_v3.py is saved: """
SAVE_DATA = True

""" get the DM mass in MeV (float): """
DM_mass = int(sys.argv[1])

""" set the path to the correct folder: """
path_folder = str(sys.argv[2])

""" set the path of the output folder: """
path_output = path_folder + "/" + str(sys.argv[3])

""" set the path of the analysis: """
path_analysis = path_output + "/analysis_mcmc"

""" set the number of datasets analyzed per job: """
number_of_datasets_per_job = int(sys.argv[4])

""" Set the path of the file, where the information about the analysis is saved: """
# TODO: Check the file-path
file_info_analysis = path_analysis + "/info_mcmc_analysis_1_{0}.txt".format(number_of_datasets_per_job)

# Set the number of the files, that should be read in (is equal to dataset_start and dataset_stop):
file_number_start = int(sys.argv[5])
file_number_stop = int(sys.argv[6])

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
    np.savetxt(path_output + "/result_mcmc/result_dataset_output_{0:d}.txt".format(DM_mass),
               np.array([lower_energy_bound, upper_energy_bound, number_of_entries,
                         signal_expected, S_50, S_50_sigma, S_50_16, S_50_84,
                         S_90, S_90_sigma, S_90_16, S_90_84,
                         DSNB_expected, DSNB_50, DSNB_50_sigma, DSNB_50_16, DSNB_50_84,
                         CCatmo_expected, CCatmo_50, CCatmo_50_sigma, CCatmo_50_16, CCatmo_50_84,
                         Reactor_expected, Reactor_50, Reactor_50_sigma, Reactor_50_16, Reactor_50_84]), fmt="%4.5f",
               header="Results of the analysis of the spectra in dataset_output_20 (with output_analysis_v3.py, {0}):\n"
                      "Analysis of Dataset_{1:d}.txt to Dataset_{2:d}.txt\n"
                      "Information to the values below:\n"
                      "Lower bound of the energy window in MeV, upper bound of the energy window in MeV\n"
                      "Number of datasets that were analyzed,\n"
                      "Expected number of signal events from simulation,\n"
                      "Mean of the observed number of signal events,\n"
                      "Standard deviation of the observed number of signal events,\n"
                      "16 % confidence level of the observed number of signal events,\n"
                      "84 % confidence level of the observed number of signal events,\n"
                      "Mean of the 90% probability limit of the observed number of signal events,\n"
                      "Standard deviation of the 90% probability limit of the observed number of signal events,\n"
                      "16 % confidence level of the 90% probability limit of the observed number of signal events,\n"
                      "84 % confidence level of the 90% probability limit of the observed number of signal events,\n"
                      "Expected number of DSNB background events from simulation,\n"
                      "Mean of the observed number of DSNB background events,\n"
                      "Standard deviation of the observed number of DSNB background events,\n"
                      "16 % confidence level of the observed number of DSNB background events,\n"
                      "84 % confidence level of the observed number of DSNB background events,\n"
                      "Expected number of CCatmo background events from simulation,\n"
                      "Mean of the observed number of atmo. CC background events,\n"
                      "Standard deviation of the observed number of atmo. CC background events,\n"
                      "16 % confidence level of the observed number of atmo. CC background events,\n"
                      "84 % confidence level of the observed number of atmo. CC background events,\n"
                      "Expected number of reactor background events from simulation,\n"
                      "Mean of the observed number of Reactor background events,\n"
                      "Standard deviation of the observed number of Reactor background events,\n"
                      "16 % confidence level of the observed number of Reactor background events,\n"
                      "84 % confidence level of the observed number of Reactor background events:"
               .format(now, file_number_start, file_number_stop))

    # print message, that result data is saved in file:
    print("result data is saved in the file result_dataset_output_{0:d}.txt".format(DM_mass))

if SAVE_DATA:
    # Save the array S_mode to txt file:
    np.savetxt(path_output + "/result_mcmc/S_mode_DMmass{0}.txt".format(DM_mass),
               S_mode, fmt="%4.5f",
               header="Observed number of signal events (mode of S) from DM with mass={2:.1f}MeV in virtual "
                      "experiments\n (in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
               .format(lower_energy_bound, upper_energy_bound, DM_mass))

    # Save the array S_90_limit to txt file:
    np.savetxt(path_output + "/result_mcmc/S_90_limit_DMmass{0}.txt".format(DM_mass),
               S_90_limit, fmt="%4.5f",
               header="90 % upper limit of number of observed signal events S from DM with mass={0:.1f}MeV in virtual"
                      "experiments"
               .format(DM_mass))

    # Save the array DSNB_mode to txt file:
    np.savetxt(path_output + "/result_mcmc/DSNB_mode_DMmass{0}.txt".format(DM_mass),
               DSNB_mode, fmt="%4.5f",
               header="Observed number of DSNB background events (mode of B_DSNB) in virtual experiments\n"
                      "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
               .format(lower_energy_bound, upper_energy_bound))

    # Save the array CCatmo_mode to txt file:
    np.savetxt(path_output + "/result_mcmc/CCatmo_mode_DMmass{0}.txt".format(DM_mass),
               CCatmo_mode, fmt="%4.5f",
               header="Observed number of CCatmo background events (mode of B_CCatmo) in virtual experiments\n"
                      "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
               .format(lower_energy_bound, upper_energy_bound))

    # Save the array Reactor_mode to txt file:
    np.savetxt(path_output + "/result_mcmc/Reactor_mode_DMmass{0}.txt".format(DM_mass),
               Reactor_mode, fmt="%4.5f",
               header="Observed number of reactor background events (mode of B_reactor) in virtual experiments\n"
                      "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
               .format(lower_energy_bound, upper_energy_bound))
