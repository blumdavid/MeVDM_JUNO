""" Script "autocorr_time_analysis_v2.py" from 30.10.2019.

    It was used for simulation and analysis of "S90_DSNB_CCatmo_reactor_NCatmo_FN".

    Script to analyze the txt-file with the mean of the autocorrelation time from analysis of virt.
    experiments with analyze_spectra_v6_server2.py

    Also the acceptance fraction for burnin phase and sampling phase can be analyzed.

"""

import numpy as np
from matplotlib import pyplot as plt

# DM mass in MeV:
DM_mass = 100

path_output = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo_FN/simulation_1/dataset_output_{0}"\
    .format(DM_mass)

# path to the folder, where the txt files are saved:
path_files = path_output + "/analysis_mcmc"

# variable, that includes, for how many datasets the autocorrelation time could be calculated (integer):
number_acor_calculated = 0

# array, that includes the values of the autocorrelation time, which could me calculated (empty np.array):
acor_time_calculated = []

# array, that includes the values of the acceptance fraction during burnin:
acc_frac_burnin = []

# array, that includes the values of the acceptance fraction during sampling:
acc_frac_sampling = []

# variable, that includes, for how many datasets the autocorrelation time could NOT be calculated
# (because the chain was too short -> AutocorrError occurred) (integer):
number_acor_failed = 0

# loop over the txt files:
for index in np.arange(0, 9999, 100):

    # load the mean autocorrelation time from the txt files (np.array of float):
    autocorr_time = np.loadtxt(path_files + "/autocorrelation_time_{0}_{1}.txt".format(index, index+99))
    # load acceptance fraction during burnin from txt file:
    fraction_burnin = np.loadtxt(path_files + "/acceptance_fraction_burnin_{0}_{1}.txt".format(index, index+99))
    # load acceptance fraction during sampling from txt file:
    fraction_sampling = np.loadtxt(path_files + "/acceptance_fraction_sampling_{0}_{1}.txt".format(index, index+99))

    # check if lengths are equal:
    if len(autocorr_time) != len(fraction_burnin) or len(autocorr_time) != len(fraction_sampling):
        print("ERROR: files length not equal!!")

    for index2 in np.arange(0, len(autocorr_time), 1):

        # if AutocorrError occurred:
        if autocorr_time[index2] == 1001001:
            # increment the number of failed autocorr. time:
            number_acor_failed = number_acor_failed + 1

        else:
            # increment the number of calculated autocorr. time:
            number_acor_calculated = number_acor_calculated + 1

            # add value of autocorr. time to array:
            acor_time_calculated.append(autocorr_time[index2])

    # append fraction_burnin and fraction_sampling to array:
    acc_frac_burnin = np.append(acc_frac_burnin, fraction_burnin)
    acc_frac_sampling = np.append(acc_frac_sampling, fraction_sampling)

print("number of AutocorrErrors for 10000 datasets = {0}".format(number_acor_failed))
print("number of dataset, where autocorr. time was calculated = {0} (for 5000 datasets)".format(number_acor_calculated))

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
plt.savefig(path_output + "/result_mcmc/result_autocorr.png")

# Display S_mean in histogram:
h2 = plt.figure(2, figsize=(15, 8))
Bins2 = 'auto'
# Bins1 = np.arange(0, np.max(S_mode)+0.1, 0.05)
n2, bins2, patches2 = plt.hist(acc_frac_burnin, bins=Bins2, histtype='step',
                               label="number of entries = {0}".format(len(acc_frac_burnin)))
plt.xlabel("calculated acceptance fraction")
plt.ylabel("counts")
plt.title("Acceptance fraction during burnin phase for 10000 datasets in dataset_output_{0}".format(DM_mass))
plt.legend()
plt.savefig(path_output + "/result_mcmc/result_acc_frac_burnin.png")

# Display S_mean in histogram:
h3 = plt.figure(3, figsize=(15, 8))
Bins3 = 'auto'
# Bins1 = np.arange(0, np.max(S_mode)+0.1, 0.05)
n3, bins3, patches3 = plt.hist(acc_frac_sampling, bins=Bins3, histtype='step',
                               label="number of entries = {0}".format(len(acc_frac_sampling)))
plt.xlabel("calculated acceptance fraction")
plt.ylabel("counts")
plt.title("Acceptance fraction during sampling for 10000 datasets in dataset_output_{0}".format(DM_mass))
plt.legend()
plt.savefig(path_output + "/result_mcmc/result_acc_frac_sampling.png")

plt.show()









