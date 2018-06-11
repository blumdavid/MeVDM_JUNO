""" Script to analyze the txt-file with the mean of the autocorrelation time from analysis of virt.
    experiments with analyze_spectra_v5_server.py

"""

import numpy as np
from matplotlib import pyplot as plt

# DM mass in MeV:
DM_mass = 25

# number of the test:
test_number = 9

# path to the folder, where the txt files are saved:
path_files = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor/dataset_output_{0}/" \
             "analysis_mcmc_test{1}/".format(DM_mass, test_number)

# variable, that includes, for how many datasets the autocorrelation time could be calculated (integer):
number_acor_calculated = 0

# array, that includes the values of the autocorrelation time, which could me calculated (empty np.array):
acor_time_calculated = np.array([])

# variable, that includes, for how many datasets the autocorrelation time could NOT be calculated
# (because the chain was too short -> AutocorrError occurred) (integer):
number_acor_failed = 0

# loop over the txt files:
for index in np.arange(1, 10000, 100):

    # load the mean autocorrelation time from the txt files (np.array of float):
    autocorr_time = np.loadtxt(path_files + "autocorrelation_time_{0}_{1}.txt".format(index, index+99))

    for index2 in np.arange(0, len(autocorr_time), 1):

        # if AutocorrError occurred:
        if autocorr_time[index2] == 1001001:
            # increment the number of failed autocorr. time:
            number_acor_failed = number_acor_failed + 1

        else:
            # increment the number of calculated autocorr. time:
            number_acor_calculated = number_acor_calculated + 1

            # add value of autocorr. time to array:
            acor_time_calculated = np.append(acor_time_calculated, autocorr_time[index2])

    print("number of AutocorrErrors for dataset {0} to {1} = {2}".format(index, index+99, number_acor_failed))
    print("number of dataset, where autocorr. time was calculated = {0} (for dataset {1} to {2})"
          .format(number_acor_calculated, index, index + 99))

print("number of AutocorrErrors for 10000 datasets = {0}".format(number_acor_failed))
print("number of dataset, where autocorr. time was calculated = {0} (for 10000 datasets)".format(number_acor_calculated))

# mean of the calculated autocorr. times (float):
mean_acor_time_calculated = np.mean(acor_time_calculated)
print("mean of the calculated mean autocorrelation times = {0}".format(mean_acor_time_calculated))

# Display S_mean in histogram:
h1 = plt.figure(1, figsize=(15, 8))
Bins1 = 'auto'
# Bins1 = np.arange(0, np.max(S_mode)+0.1, 0.05)
n1, bins1, patches1 = plt.hist(acor_time_calculated, bins=Bins1, histtype='step', color='b',
                               label="number of entries = {0}".format(number_acor_calculated))
plt.xlabel("calculated mean autocorrelation time")
plt.ylabel("counts")
plt.title("Mean autocorrelation time for 10000 datasets in dataset_output_{0}".format(DM_mass))
plt.legend()
plt.show()









