""" Script to display the results of the MCMC analysis for indirect DM search with neutrinos in JUNO:

    The datasets are generated and the analysis is made on the IHEP cluster in China for different DM masses
    automatically (see file: "info_auto_simu.txt")

    The output of the analysis (S_mode, S_90_limit, DSNB_mode, CCatmo_mode, Reactor_mode) is saved in
    txt-files for each DM mass.

    These txt-files are displayed in this script.
"""

import numpy as np
from matplotlib import pyplot as plt


""" set the DM mass in MeV (float): """
DM_mass = 100

""" set the path to the correct folder: """
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor"

""" set the path to the folder, where the results are saved: """
path_result = path_folder + "/dataset_output_{0}/result_mcmc".format(DM_mass)

""" set the path to the file, where the results are saved: """
file_result_dataset = path_result + "/result_dataset_output_{0}.txt".format(DM_mass)
file_Signal = path_result + "/S_mode_DMmass{0}.txt".format(DM_mass)
file_S_90_limit = path_result + "/S_90_limit_DMmass{0}.txt".format(DM_mass)
file_DSNB = path_result + "/DSNB_mode_DMmass{0}.txt".format(DM_mass)
file_CCatmo = path_result + "/CCatmo_mode_DMmass{0}.txt".format(DM_mass)
file_reactor = path_result + "/Reactor_mode_DMmass{0}.txt".format(DM_mass)

""" load the txt files: """
info_result = np.loadtxt(file_result_dataset)
S_mode = np.loadtxt(file_Signal)
S_90_limit = np.loadtxt(file_S_90_limit)
DSNB_mode = np.loadtxt(file_DSNB)
CCatmo_mode = np.loadtxt(file_CCatmo)
Reactor_mode = np.loadtxt(file_reactor)

""" get the information of the results from the info_result file: """
# Lower bound of the energy window in MeV (float):
lower_energy_bound = info_result[0]
# upper bound of the energy window in MeV (float):
upper_energy_bound = info_result[1]
# Number of datasets that were analyzed (float):
number_of_entries = info_result[2]
# Expected number of signal events from simulation (float):
signal_expected = info_result[3]
# Mean of the observed number of signal events (float):
S_50 = info_result[4]
# Mean of the 90% probability limit of the observed number of signal events (float):
S_90 = info_result[8]
# Expected number of DSNB background events from simulation (float):
DSNB_expected = info_result[12]
# Mean of the observed number of DSNB background events (float):
DSNB_50 = info_result[13]
# 16 % confidence level of the observed number of DSNB background events (float):
DSNB_50_16 = info_result[15]
# 84 % confidence level of the observed number of DSNB background events (float):
DSNB_50_84 = info_result[16]
# Expected number of CCatmo background events from simulation (float):
CCatmo_expected = info_result[17]
# Mean of the observed number of atmo. CC background events (float):
CCatmo_50 = info_result[18]
# 16 % confidence level of the observed number of atmo. CC background events (float):
CCatmo_50_16 = info_result[20]
# 84 % confidence level of the observed number of atmo. CC background events (float):
CCatmo_50_84 = info_result[21]
# Expected number of reactor background events from simulation (float):
Reactor_expected = info_result[22]
# Mean of the observed number of Reactor background events (float):
Reactor_50 = info_result[23]
# 16 % confidence level of the observed number of Reactor background events (float):
Reactor_50_16 = info_result[25]
# 84 % confidence level of the observed number of Reactor background events (float):
Reactor_50_84 = info_result[26]

""" Display the results in histograms: """
# Display S_mean in histogram:
h1 = plt.figure(1, figsize=(15, 8))
# Bins1 = 'auto'
Bins1 = np.arange(0, np.max(S_mode)+0.1, 0.1)
n_S, bins1, patches1 = plt.hist(S_mode, bins=Bins1, histtype='step', color='b',
                                label='number of virt. experiments = {0:.0f},\n'
                                      'mean of the distribution = {1:.4f}'
                                .format(number_of_entries, S_50))
plt.axvline(signal_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(signal_expected), color='r')
plt.xticks(np.arange(0, np.max(S_mode)+0.5, 0.5))
plt.xlabel("total number of observed signal events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of signal events from DM with mass={2:.1f}MeV in virtual experiments "
          "\n(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()
plt.savefig(path_result + "/result_signal.png")


# Display S_90_limit in histogram:
h2 = plt.figure(2, figsize=(15, 8))
# Bins2 = 'auto'
Bins2 = np.arange(2, np.max(S_90_limit)+0.1, 0.1)
n_limit_S, bins2, patches2 = plt.hist(S_90_limit, bins=Bins2, histtype='step', color='b',
                                      label='number of virt. experiments = {0:.0f}'.format(number_of_entries))
plt.axvline(S_90, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(S_90), color='r')
plt.xticks(np.arange(2, np.max(S_90_limit)+0.5, 0.5))
plt.xlabel("90 percent limit of number of observed signal events S")
plt.ylabel("counts")
plt.title("Distribution of the 90 % upper limit of the signal contribution for DM with mass = {0:.1f} MeV"
          .format(DM_mass))
plt.legend()
plt.savefig(path_result + "/result_S90.png")


# Display DSNB_mean in histogram:
h3 = plt.figure(3, figsize=(15, 8))
Bins3 = 'auto'
# Bins3 = np.arange(0, 25, 1)
n_DSNB, bins3, patches3 = plt.hist(DSNB_mode, bins=Bins3, histtype='step', color='b',
                                   label='number of virt. experiments = {0:.0f}'
                                   .format(number_of_entries))
plt.axvline(DSNB_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(DSNB_expected), color='r')
plt.axvline(DSNB_50, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(DSNB_50), color='b')
plt.axvline(DSNB_50_16, linestyle=':', label='16% probability limit = {0:.4f}'.format(DSNB_50_16), color='b')
plt.axvline(DSNB_50_84, linestyle='-.', label='84% probability limit = {0:.4f}'.format(DSNB_50_84), color='b')
plt.xticks(np.arange(0, np.max(DSNB_mode)+2.5, 2.5))
plt.xlabel("total number of observed DSNB background events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of DSNB background events in virtual experiments "
          "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()
plt.savefig(path_result + "/result_DSNB.png")


# Display CCatmo_mean in histogram:
h4 = plt.figure(4, figsize=(15, 8))
Bins4 = 'auto'
# Bins4 = np.arange(0, 3, 0.1)
n_CCatmo, bins4, patches4 = plt.hist(CCatmo_mode, bins=Bins4, histtype='step', color='b',
                                     label='number of virt. experiments = {0:.0f}'
                                     .format(number_of_entries))
plt.axvline(CCatmo_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(CCatmo_expected), color='r')
plt.axvline(CCatmo_50, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(CCatmo_50), color='b')
plt.axvline(CCatmo_50_16, linestyle=':', label='16% probability limit = {0:.4f}'.format(CCatmo_50_16), color='b')
plt.axvline(CCatmo_50_84, linestyle='-.', label='84% probability limit = {0:.4f}'.format(CCatmo_50_84), color='b')
plt.xticks(np.arange(15, np.max(CCatmo_mode)+2.5, 2.5))
plt.xlabel("total number of observed atmo. CC background events events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of atmospheric CC background events in virtual experiments "
          "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()
plt.savefig(path_result + "/result_CCatmo.png")


# Display Reactor_mean in histogram:
h5 = plt.figure(5, figsize=(15, 8))
Bins5 = 'auto'
# Bins5 = np.arange(0, 0.5, 0.01)
n_Reactor, bins5, patches5 = plt.hist(Reactor_mode, bins=Bins5, histtype='step', color='b',
                                      label='number of virt. experiments = {0:.0f}'
                                      .format(number_of_entries))
plt.axvline(Reactor_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(Reactor_expected), color='r')
plt.axvline(Reactor_50, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(Reactor_50), color='b')
plt.axvline(Reactor_50_16, linestyle=':', label='16% probability limit = {0:.4f}'.format(Reactor_50_16), color='b')
plt.axvline(Reactor_50_84, linestyle='-.', label='84% probability limit = {0:.4f}'.format(Reactor_50_84), color='b')
plt.xticks(np.arange(20, np.max(Reactor_mode)+2.5, 2.5))
plt.xlabel("total number of observed reactor background events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of reactor background events in virtual experiments "
          "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()
plt.savefig(path_result + "/result_reactor.png")

plt.show()
