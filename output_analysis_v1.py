""" Script to display and analyze the results of the analysis done with analyze_spectra_v2.py.

    In analyze_spectra_v2.py the dataset and the simulated spectrum were analyzed and the result of the analysis
    are saved in the files DatasetX_analysis.txt

    In analyze_spectra_v2.py the total number of signal events is determined calculating the function p_S_spectrum
    defined in the GERDA paper (by integrating p_spectrum_SB over the background contributions)

"""

import numpy as np
from matplotlib import pyplot as plt
import datetime

""" Set boolean value to define, if the result of output_analysis_v1.py is saved: """
SAVE_DATA = True

""" Set the path of the folder, where the results of the analysis are saved: """
number_dataset_output = 1
path_dataset_output = "dataset_output_{0:d}".format(number_dataset_output)
path_analysis = path_dataset_output + "/analysis_integrate"

# Set the number of the files, that should be read in:
file_number_start = 1
file_number_stop = 1000

""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

""" Preallocate the arrays, where the results of the analysis of the different datasets should be appended: """
# value of log_p_H_spectrum (np.array of float):
log_p_H_spectrum = np.array([])
# value of log_p_Hbar_spectrum (np.array of float):
log_p_Hbar_spectrum = np.array([])
# value of the mode of number of signal events (np.array of float):
mode_of_S = np.array([])
# value of upper 90 percent limit of the number of signal events (np.array of float):
limit_S_90 = np.array([])

""" Read in the files, where the results of the analysis are saved and read the result-values: """
for number in np.arange(file_number_start, file_number_stop+1, 1):
    # load the file corresponding to Dataset{number}_analysis.txt:
    result_analysis = np.loadtxt(path_analysis + "/Dataset{0:d}_analysis.txt".format(number))
    # get value of log_p_H_spectrum (float):
    value_log_p_H_spectrum = result_analysis[0]
    # get value of log_p_Hbar_spectrum (float):
    value_log_p_Hbar_spectrum = result_analysis[1]
    # get value of mode_of_S (float):
    value_mode_of_S = result_analysis[2]
    # get value of 90 percent limit S (float):
    value_limit_S_90 = result_analysis[3]

    # Append the values to the preallocated arrays (np.array of float):
    log_p_H_spectrum = np.append(log_p_H_spectrum, value_log_p_H_spectrum)
    log_p_Hbar_spectrum = np.append(log_p_Hbar_spectrum, value_log_p_Hbar_spectrum)
    mode_of_S = np.append(mode_of_S, value_mode_of_S)
    limit_S_90 = np.append(limit_S_90, value_limit_S_90)


# calculate the number of file, that are read in. Equivalent to the number of entries in the result array
number_of_entries = np.size(log_p_H_spectrum)
# calculate the mean of log_p_H_spectrum (float):
mean_log_p_H_spectrum = np.mean(log_p_H_spectrum)
# calculate the mean and the standard deviation of the array mode_of_S (float):
mean_mode_of_S = np.mean(mode_of_S)
sigma_mode_of_S = np.std(mode_of_S)
# calculate the mean and the standard deviation of the array limit_S_90 (float):
mean_limit_S_90 = np.mean(limit_S_90)
sigma_limit_S_90 = np.std(limit_S_90)

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

""" Save the results in txt file: """
if SAVE_DATA:
    np.savetxt(path_dataset_output + "/result/result_dataset_output_{0:d}_integrate.txt"
               .format(number_dataset_output),
               np.array([number_of_entries, mean_mode_of_S, sigma_mode_of_S, mean_limit_S_90, sigma_limit_S_90,
                         log_discovery_crit, num_entries_discovery, log_evidence_crit, num_entries_evidence,
                         num_entries_nodiscovery]), fmt="%4.5f",
               header="Results of the analysis of the spectra in dataset_output_20 (with output_analysis_v1.py, {0}):\n"
                      "Analysis of Dataset_{1:d}.txt to Dataset_{2:d}.txt\n"
                      "Information to the values below:\n"
                      "Number of datasets that were analyzed,\n"
                      "Mean of the observed number of signal events,\n"
                      "Standard deviation of the observed number of signal events,\n"
                      "Mean of the 90% probability limit of the observed number of signal events,\n"
                      "Standard deviation of the 90% probability limit of the observed number of signal events,\n"
                      "Decimal logarithm of the discovery criterion,\n"
                      "Number of datasets that claim a discovery,\n"
                      "Decimal logarithm of criterion for evidence,\n"
                      "Number of datasets that give evidence for a discovery,\n"
                      "Number of datasets that do not claim a discovery.\n"
               .format(now, file_number_start, file_number_stop))

# Display the distribution in histograms:
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
          "are due to background only")
plt.legend(loc='upper left', framealpha=1)

h2 = plt.figure(2)
Bins2 = 'auto'
mode_S, bins2, patches2 = plt.hist(mode_of_S, bins=Bins2, histtype='step', color='b',
                                   label='number of virt. experiments = {0:d},\n'
                                         'mean of the distribution = {1:.3f},\n'
                                         'standard deviation of the distribution = {2:.3f}'
                                   .format(number_of_entries, mean_mode_of_S, sigma_mode_of_S))
plt.xlabel("mode(S)")
plt.ylabel("counts")
plt.title("Distribution of the observed number of signal events in virtual experiments")
plt.legend()

h3 = plt.figure(3)
Bins3 = 'auto'
limit_S, bins3, patches3 = plt.hist(limit_S_90, bins=Bins3, histtype='step', color='k',
                                    label='number of virt. experiments = {0:d},\n'
                                          'mean of the distribution = {1:.3f},\n'
                                          'standard deviation of the distribution = {2:.3f}'
                                    .format(number_of_entries, mean_limit_S_90, sigma_limit_S_90))
plt.xlabel("90 percent limit of S")
plt.ylabel("counts")
plt.title("Distribution of the 90 % probability limit of the signal contribution")
plt.legend()

plt.show()
