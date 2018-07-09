""" Script to display and analyze the results of the MCMC analysis done with analyze_spectra_v4_local.py or
    analyze_spectra_v4_server.py!

    In analyze_spectra_v4_local.py the dataset and the simulated spectrum were analyzed with
    Markov Chain Monte Carlo (MCMC) sampling and the results of the analysis are saved in the files
    DatasetX_mcmc_analysis.txt

    In analyze_spectra_v4_local.py the mode of the total number of signal and background events are determined by MCMC
    sampling of the posterior probability

    Script uses the function output_analysis() from output_analysis_v3.py,
    but is optimized to run on the local computer.

"""

import numpy as np
from matplotlib import pyplot as plt
from work.MeVDM_JUNO.source.signal_DSNB_CCatmo_reactor.output_analysis_v3 import output_analysis

""" Set boolean value to define, if the result of output_analysis_v3.py is saved: """
SAVE_DATA = True

""" set the DM mass in MeV (float): """
DM_mass = 30

""" set the path to the correct folder: """
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor"

""" set the path of the output folder: """
path_output = path_folder + "/dataset_output_{0:d}".format(DM_mass)

""" set the path of the analysis folder: """
path_analysis = path_output + "/analysis_mcmc_moreBkgEv"

""" Set the path of the file, where the information about the analysis is saved: """
# TODO: Check the file-path
file_info_analysis = path_analysis + "/info_mcmc_analysis_1_50.txt"

# Set the number of the files, that should be read in:
file_number_start = 1
file_number_stop = 5000


""" display and analyze the results from the analysis with function output_analysis() from output_analysis_v3.py """
(number_of_entries, lower_energy_bound, upper_energy_bound, S_mode, S_50, S_50_sigma, S_50_16, S_50_84, signal_expected,
 S_90_limit, S_90, S_90_sigma, S_90_16, S_90_84, DSNB_mode, DSNB_50, DSNB_50_sigma, DSNB_50_16, DSNB_50_84,
 DSNB_expected, CCatmo_mode, CCatmo_50, CCatmo_50_sigma, CCatmo_50_16, CCatmo_50_84, CCatmo_expected, Reactor_mode,
 Reactor_50, Reactor_50_sigma, Reactor_50_16, Reactor_50_84, Reactor_expected) \
    = output_analysis(SAVE_DATA, DM_mass, path_output, path_analysis, file_info_analysis, file_number_start,
                      file_number_stop)

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

""" Display the results in histograms: """
# Display S_mean in histogram:
h1 = plt.figure(1, figsize=(15, 8))
# Bins1 = 'auto'
Bins1 = np.arange(0, np.max(S_mode)+0.1, 0.05)
n_S, bins1, patches1 = plt.hist(S_mode, bins=Bins1, histtype='step', color='b',
                                label='number of virt. experiments = {0:d}'
                                .format(number_of_entries))
plt.axvline(S_50, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(S_50))
plt.axvline(signal_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
            .format(signal_expected), color='r')
plt.xticks(np.arange(0, np.max(S_mode)+0.5, 0.5))
plt.xlabel("total number of observed signal events")
plt.ylabel("counts")
plt.title("Distribution of the observed number of signal events from DM with mass={2:.1f}MeV in virtual experiments "
          "\n(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()


# Display S_90_limit in histogram:
h2 = plt.figure(2, figsize=(15, 8))
Bins2 = 'auto'
# Bins2 = np.arange(2, np.max(S_90_limit)+0.1, 0.1)
n_limit_S, bins2, patches2 = plt.hist(S_90_limit, bins=Bins2, histtype='step', color='b',
                                      label='number of virt. experiments = {0:d}'
                                      .format(number_of_entries))
plt.axvline(S_90, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(S_90))
# plt.xticks(np.arange(2, np.max(S_90_limit)+0.5, 0.5))
plt.xlabel("90 percent limit of number of observed signal events S")
plt.ylabel("counts")
plt.title("Distribution of the 90 % upper limit of the signal contribution for DM with mass = {0:.1f} MeV"
          .format(DM_mass))
plt.legend()


# Display DSNB_mean in histogram:
h3 = plt.figure(3, figsize=(15, 8))
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
          "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()


# Display CCatmo_mean in histogram:
h4 = plt.figure(4, figsize=(15, 8))
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
          "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()


# Display Reactor_mean in histogram:
h5 = plt.figure(5, figsize=(15, 8))
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
          "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
          .format(lower_energy_bound, upper_energy_bound, DM_mass))
plt.legend()

plt.show()
