""" Script "display_results_v2.py" from 30.10.2019.

    It was used for simulation and analysis of "S90_DSNB_CCatmo_reactor_NCatmo_FN".

    Script to display the results of the MCMC analysis for indirect DM search with neutrinos in JUNO:

    The datasets are generated and the analysis is made on the IHEP cluster in China for different DM masses
    automatically.

    The output of the analysis (S_mode, S_90_limit, DSNB_mode, CCatmo_mode, Reactor_mode) is saved in
    txt-files for each DM mass.

    These txt-files are displayed in this script.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


""" set the DM mass in MeV (float): """
# DarkMatter_mass = 15
DarkMatter_mass = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

""" set the path to the correct folder: """
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo_FN_newIBD/simulation_3"

for DM_mass in DarkMatter_mass:

    print("\nsave results for DM mass = {0:d} MeV...".format(DM_mass))

    """ set the path to the folder, where the results are saved: """
    path_result = path_folder + "/dataset_output_{0}/result_mcmc".format(DM_mass)

    """ set the path to the file, where the results are saved: """
    file_result_dataset = path_result + "/result_dataset_output_{0}.txt".format(DM_mass)
    file_Signal = path_result + "/S_mode_DMmass{0}.txt".format(DM_mass)
    file_S_90_limit = path_result + "/S_90_limit_DMmass{0}.txt".format(DM_mass)
    file_DSNB = path_result + "/DSNB_mode_DMmass{0}.txt".format(DM_mass)
    file_CCatmo = path_result + "/CCatmo_mode_DMmass{0}.txt".format(DM_mass)
    file_reactor = path_result + "/Reactor_mode_DMmass{0}.txt".format(DM_mass)
    file_NCatmo = path_result + "/NCatmo_mode_DMmass{0}.txt".format(DM_mass)
    file_FN = path_result + "/FN_mode_DMmass{0}.txt".format(DM_mass)

    """ load the txt files: """
    info_result = np.loadtxt(file_result_dataset)
    S_mode = np.loadtxt(file_Signal)
    S_90_limit = np.loadtxt(file_S_90_limit)
    DSNB_mode = np.loadtxt(file_DSNB)
    CCatmo_mode = np.loadtxt(file_CCatmo)
    Reactor_mode = np.loadtxt(file_reactor)
    NCatmo_mode = np.loadtxt(file_NCatmo)
    FN_mode = np.loadtxt(file_FN)

    """ get the information of the results from the info_result file: """
    # Lower bound of the energy window in MeV (float):
    lower_energy_bound = info_result[0]
    # upper bound of the energy window in MeV (float):
    upper_energy_bound = info_result[1]
    # Number of datasets that were analyzed (float):
    number_of_entries = info_result[2]

    """ Signal events: """
    # Expected number of signal events from simulation (float):
    signal_expected = info_result[3]
    # Mean of the observed number of signal events (float):
    S_50_mean = info_result[4]
    # Standard deviation of the observed number of signal events (float):
    S_50_std = info_result[5]
    # 16 % confidence level of the observed number of signal events (float):
    S_50_16 = info_result[6]
    # 84 % confidence level of the observed number of signal events
    S_50_84 = info_result[7]
    # -1 sigma value of the observed number of signal events (float):
    S_50_minus_1sigma = S_50_mean - S_50_std
    # +1 sigma value of the observed number of signal events (float):
    S_50_plus_1sigma = S_50_mean + S_50_std
    # 2.5 % confidence level of the observed number of signal events (float):
    S_50_2_5 = np.percentile(S_mode, 2.5)
    print(S_50_2_5)
    # 97.5 % confidence level of the observed number of signal events (float):
    S_50_97_5 = np.percentile(S_mode, 97.5)
    print(S_50_97_5)
    # -2 sigma value of the observed number of signal events (float):
    S_50_minus_2sigma = S_50_mean - 2*S_50_std
    # +2 sigma value of the observed number of signal events (float):
    S_50_plus_2sigma = S_50_mean + 2*S_50_std
    # 0.15 % confidence level of the observed number of signal events (float):
    S_50_0_15 = np.percentile(S_mode, 0.15)
    print(S_50_0_15)
    # 99.85 % confidence level of the observed number of signal events (float):
    S_50_99_85 = np.percentile(S_mode, 99.85)
    print(S_50_99_85)
    # -3 sigma value of the observed number of signal events (float):
    S_50_minus_3sigma = S_50_mean - 3*S_50_std
    # +3 sigma value of the observed number of signal events (float):
    S_50_plus_3sigma = S_50_mean + 3*S_50_std

    """ 90 % limit on signal events: """
    # Mean of the 90% probability limit of the observed number of signal events (float):
    S_90_mean = info_result[8]
    # Standard deviation of the 90% probability limit of the observed number of signal events (float):
    S_90_std = info_result[9]
    # 16 % confidence level of the 90% probability limit of the observed number of signal events (float):
    S_90_16 = info_result[10]
    # 84 % confidence level of the 90% probability limit of the observed number of signal events (float):
    S_90_84 = info_result[11]
    # -1 sigma value of the 90% probability limit of the observed number of signal events (float):
    S_90_minus_1sigma = S_90_mean - S_90_std
    # +1 sigma value of the 90% probability limit of the observed number of signal events (float):
    S_90_plus_1sigma = S_90_mean + S_90_std
    # 2.5 % confidence level of the 90% probability limit of the observed number of signal events (float):
    S_90_2_5 = np.percentile(S_90_limit, 2.5)
    print(S_90_2_5)
    # 97.5 % confidence level of the 90% probability limit of the observed number of signal events (float):
    S_90_97_5 = np.percentile(S_90_limit, 97.5)
    print(S_90_97_5)
    # -2 sigma value of the 90% probability limit of the observed number of signal events (float):
    S_90_minus_2sigma = S_90_mean - 2*S_90_std
    # +2 sigma value of the 90% probability limit of the observed number of signal events (float):
    S_90_plus_2sigma = S_90_mean + 2*S_90_std
    # 0.15 % confidence level of the 90% probability limit of the observed number of signal events (float):
    S_90_0_15 = np.percentile(S_90_limit, 0.15)
    print(S_90_0_15)
    # 99.85 % confidence level of the 90% probability limit of the observed number of signal events (float):
    S_90_99_85 = np.percentile(S_90_limit, 99.85)
    print(S_90_99_85)
    # -3 sigma value of the observed number of the 90% probability limit of the signal events (float):
    S_90_minus_3sigma = S_90_mean - 3*S_90_std
    # +3 sigma value of the observed number of the 90% probability limit of the signal events (float):
    S_90_plus_3sigma = S_90_mean + 3*S_90_std

    """ DSNB background events: """
    # Expected number of DSNB background events from simulation (float):
    DSNB_expected = info_result[12]
    # Mean of the observed number of DSNB background events (float):
    DSNB_mean = info_result[13]
    # Standard deviation of the observed number of DSNB background events (float):
    DSNB_std = info_result[14]
    # 16 % confidence level of the observed number of DSNB background events (float):
    DSNB_16 = info_result[15]
    # 84 % confidence level of the observed number of DSNB background events (float):
    DSNB_84 = info_result[16]
    # -1 sigma value of the observed number of DSNB background events (float):
    DSNB_minus_1sigma = DSNB_mean - DSNB_std
    # +1 sigma value of the observed number of DSNB background events (float):
    DSNB_plus_1sigma = DSNB_mean + DSNB_std
    # 2.5 % confidence level of the observed number of DSNB background events (float):
    DSNB_2_5 = np.percentile(DSNB_mode, 2.5)
    # 97.5 % confidence level of the observed number of DSNB background events (float):
    DSNB_97_5 = np.percentile(DSNB_mode, 97.5)
    # -2 sigma value of the observed number of DSNB background events (float):
    DSNB_minus_2sigma = DSNB_mean - 2*DSNB_std
    # +2 sigma value of the observed number of DSNB background events (float):
    DSNB_plus_2sigma = DSNB_mean + 2*DSNB_std
    # 0.15 % confidence level of the observed number of DSNB background events (float):
    DSNB_0_15 = np.percentile(DSNB_mode, 0.15)
    # 99.85 % confidence level of the observed number of DSNB background events (float):
    DSNB_99_85 = np.percentile(DSNB_mode, 99.85)
    # -3 sigma value of the observed number of DSNB background events (float):
    DSNB_minus_3sigma = DSNB_mean - 3*DSNB_std
    # +3 sigma value of the observed number of DSNB background events (float):
    DSNB_plus_3sigma = DSNB_mean + 3*DSNB_std

    """ Atmospheric CC background events: """
    # Expected number of CCatmo background events from simulation (float):
    CCatmo_expected = info_result[17]
    # Mean of the observed number of atmo. CC background events (float):
    CCatmo_mean = info_result[18]
    # Standard deviation of the observed number of atmo. CC background events (float):
    CCatmo_std = info_result[19]
    # 16 % confidence level of the observed number of atmo. CC background events (float):
    CCatmo_16 = info_result[20]
    # 84 % confidence level of the observed number of atmo. CC background events (float):
    CCatmo_84 = info_result[21]
    # -1 sigma value of the observed number of atmo. CC background events (float):
    CCatmo_minus_1sigma = CCatmo_mean - CCatmo_std
    # +1 sigma value of the observed number of atmo. CC background events (float):
    CCatmo_plus_1sigma = CCatmo_mean + CCatmo_std
    # 2.5 % confidence level of the observed number of atmo. CC background events (float):
    CCatmo_2_5 = np.percentile(CCatmo_mode, 2.5)
    # 97.5 % confidence level of the observed number of atmo. CC background events (float):
    CCatmo_97_5 = np.percentile(CCatmo_mode, 97.5)
    # -2 sigma value of the observed number of atmo. CC background events (float):
    CCatmo_minus_2sigma = CCatmo_mean - 2*CCatmo_std
    # +2 sigma value of the observed number of atmo. CC background events (float):
    CCatmo_plus_2sigma = CCatmo_mean + 2*CCatmo_std
    # 0.15 % confidence level of the observed number of atmo. CC background events (float):
    CCatmo_0_15 = np.percentile(CCatmo_mode, 0.15)
    # 99.85 % confidence level of the observed number of atmo. CC background events (float):
    CCatmo_99_85 = np.percentile(CCatmo_mode, 99.85)
    # -3 sigma value of the observed number of atmo. CC background events (float):
    CCatmo_minus_3sigma = CCatmo_mean - 3*CCatmo_std
    # +3 sigma value of the observed number of atmo. CC background events (float):
    CCatmo_plus_3sigma = CCatmo_mean + 3*CCatmo_std

    """ Reactor background events: """
    # Expected number of reactor background events from simulation (float):
    Reactor_expected = info_result[22]
    # Mean of the observed number of Reactor background events (float):
    Reactor_mean = info_result[23]
    # Standard deviation of the observed number of Reactor background events (float):
    Reactor_std = info_result[24]
    # 16 % confidence level of the observed number of Reactor background events (float):
    Reactor_16 = info_result[25]
    # 84 % confidence level of the observed number of Reactor background events (float):
    Reactor_84 = info_result[26]
    # -1 sigma value of the observed number of Reactor background events (float):
    Reactor_minus_1sigma = Reactor_mean - Reactor_std
    # +1 sigma value of the observed number of Reactor background events (float):
    Reactor_plus_1sigma = Reactor_mean + Reactor_std
    # 2.5 % confidence level of the observed number of Reactor background events (float):
    Reactor_2_5 = np.percentile(Reactor_mode, 2.5)
    # 97.5 % confidence level of the observed number of Reactor background events (float):
    Reactor_97_5 = np.percentile(Reactor_mode, 97.5)
    # -2 sigma value of the observed number of Reactor background events (float):
    Reactor_minus_2sigma = Reactor_mean - 2*Reactor_std
    # +2 sigma value of the observed number of Reactor background events (float):
    Reactor_plus_2sigma = Reactor_mean + 2*Reactor_std
    # 0.15 % confidence level of the observed number of Reactor background events (float):
    Reactor_0_15 = np.percentile(Reactor_mode, 0.15)
    # 99.85 % confidence level of the observed number of Reactor background events (float):
    Reactor_99_85 = np.percentile(Reactor_mode, 99.85)
    # -3 sigma value of the observed number of Reactor background events (float):
    Reactor_minus_3sigma = Reactor_mean - 3*Reactor_std
    # +3 sigma value of the observed number of Reactor background events (float):
    Reactor_plus_3sigma = Reactor_mean + 3*Reactor_std

    """ Atmospheric NC background events: """
    # Expected number of NCatmo background events from simulation (float):
    NCatmo_expected = info_result[27]
    # Mean of the observed number of atmo. NC background events (float):
    NCatmo_mean = info_result[28]
    # Standard deviation of the observed number of atmo. NC background events (float):
    NCatmo_std = info_result[29]
    # 16 % confidence level of the observed number of atmo. NC background events (float):
    NCatmo_16 = info_result[30]
    # 84 % confidence level of the observed number of atmo. NC background events (float):
    NCatmo_84 = info_result[31]
    # -1 sigma value of the observed number of atmo. NC background events (float):
    NCatmo_minus_1sigma = NCatmo_mean - NCatmo_std
    # +1 sigma value of the observed number of atmo. NC background events (float):
    NCatmo_plus_1sigma = NCatmo_mean + NCatmo_std
    # 2.5 % confidence level of the observed number of atmo. NC background events (float):
    NCatmo_2_5 = np.percentile(NCatmo_mode, 2.5)
    # 97.5 % confidence level of the observed number of atmo. NC background events (float):
    NCatmo_97_5 = np.percentile(NCatmo_mode, 97.5)
    # -2 sigma value of the observed number of atmo. NC background events (float):
    NCatmo_minus_2sigma = NCatmo_mean - 2*NCatmo_std
    # +2 sigma value of the observed number of atmo. NC background events (float):
    NCatmo_plus_2sigma = NCatmo_mean + 2*NCatmo_std
    # 0.15 % confidence level of the observed number of atmo. NC background events (float):
    NCatmo_0_15 = np.percentile(NCatmo_mode, 0.15)
    # 99.85 % confidence level of the observed number of atmo. NC background events (float):
    NCatmo_99_85 = np.percentile(NCatmo_mode, 99.85)
    # -3 sigma value of the observed number of atmo. NC background events (float):
    NCatmo_minus_3sigma = NCatmo_mean - 3*NCatmo_std
    # +3 sigma value of the observed number of atmo. NC background events (float):
    NCatmo_plus_3sigma = NCatmo_mean + 3*NCatmo_std

    """ fast neutron background events: """
    # Expected number of FN background events from simulation (float):
    FN_expected = info_result[32]
    # Mean of the observed number of FN background events (float):
    FN_mean = info_result[33]
    # Standard deviation of the observed number of FN background events (float):
    FN_std = info_result[34]
    # 16 % confidence level of the observed number of FN background events (float):
    FN_16 = info_result[35]
    # 84 % confidence level of the observed number of FN background events (float):
    FN_84 = info_result[36]
    # -1 sigma value of the observed number of FN background events (float):
    FN_minus_1sigma = FN_mean - FN_std
    # +1 sigma value of the observed number of FN background events (float):
    FN_plus_1sigma = FN_mean + FN_std
    # 2.5 % confidence level of the observed number of FN background events (float):
    FN_2_5 = np.percentile(FN_mode, 2.5)
    # 97.5 % confidence level of the observed number of FN background events (float):
    FN_97_5 = np.percentile(FN_mode, 97.5)
    # -2 sigma value of the observed number of FN background events (float):
    FN_minus_2sigma = FN_mean - 2*FN_std
    # +2 sigma value of the observed number of FN background events (float):
    FN_plus_2sigma = FN_mean + 2*FN_std
    # 0.15 % confidence level of the observed number of FN background events (float):
    FN_0_15 = np.percentile(FN_mode, 0.15)
    # 99.85 % confidence level of the observed number of FN background events (float):
    FN_99_85 = np.percentile(FN_mode, 99.85)
    # -3 sigma value of the observed number of FN background events (float):
    FN_minus_3sigma = FN_mean - 3*FN_std
    # +3 sigma value of the observed number of FN background events (float):
    FN_plus_3sigma = FN_mean + 3*FN_std

    """ Display the results in histograms: """

    """ Display S_mean in histogram: """
    h1 = plt.figure(1, figsize=(15, 8))
    # Bins1 = 'auto'
    Bins1 = np.arange(0, np.max(S_mode)+0.1, 0.1)
    n_S, bins1, patches1 = plt.hist(S_mode, bins=Bins1, histtype='step', color='b',
                                    label='number of virt. experiments = {0:.0f}'.format(number_of_entries))
    plt.axvline(signal_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
                .format(signal_expected), color='r')
    plt.axvline(S_50_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(S_50_mean), color='b')
    plt.xticks(np.arange(0, np.max(S_mode)+0.5, 0.5))
    plt.xlabel("total number of observed signal events")
    plt.ylabel("counts")
    plt.title("Distribution of the observed number of signal events from DM with mass={2:.1f}MeV in virtual experiments "
              "\n(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
              .format(lower_energy_bound, upper_energy_bound, DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_signal.png")
    plt.close()


    """ Display S_mean with 68% and 95% C.L. limits: """
    h6 = plt.figure(6, figsize=(15, 8))
    # Bins6 = 'auto'
    Bins6 = np.arange(0, np.max(S_mode)+0.1, 0.1)
    n_S6, bins6, patches6 = plt.hist(S_mode, bins=Bins6, histtype='step', color='b',
                                     label='number of virt. experiments = {0:.0f}'.format(number_of_entries))
    plt.axvline(signal_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
                .format(signal_expected), color='r')
    plt.axvline(S_50_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(S_50_mean), color='b')
    # define interpolation function:
    f_S6 = interp1d(bins6[:-1], n_S6, kind='previous')
    bins6_small = np.arange(bins6[0], bins6[-2], 0.01)
    n_S6_small = f_S6(bins6_small)

    # add 95 % interval to the figure:
    plt.axvline(S_50_2_5, linestyle='-.', label='2.5% probability limit = {0:.4f}'.format(S_50_2_5), color='b')
    plt.axvline(S_50_97_5, linestyle='-.', label='97.5% probability limit = {0:.4f}'.format(S_50_97_5), color='b')
    # define a boolean array, which is True between S_50_2_5 and S_50_97_5 (array of bool):
    int_S6_95percent = (bins6_small >= S_50_2_5) * (bins6_small <= S_50_97_5)
    plt.fill_between(bins6_small, 0, n_S6_small, where=int_S6_95percent, facecolor='yellow', alpha=1, step='post')

    # add 68 % interval to the figure:
    plt.axvline(S_50_16, linestyle=':', label='16% probability limit = {0:.4f}'.format(S_50_16), color='b')
    plt.axvline(S_50_84, linestyle=':', label='84% probability limit = {0:.4f}'.format(S_50_84), color='b')
    # define a boolean array, which is True between S_50_16 and S_50_84 (array of bool):
    int_S6_68percent = (bins6_small >= S_50_16) * (bins6_small <= S_50_84)
    plt.fill_between(bins6_small, 0, n_S6_small, where=int_S6_68percent, facecolor='green', alpha=1, step='post')

    plt.xticks(np.arange(0, np.max(S_mode)+0.5, 0.5))
    plt.xlabel("total number of observed signal events")
    plt.ylabel("counts")
    plt.title("Distribution of the observed number of signal events from DM with mass={2:.1f}MeV in virtual experiments "
              "\n(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
              .format(lower_energy_bound, upper_energy_bound, DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_signal_CL.png")
    plt.close()


    """ Display S_90_limit in histogram: """
    h2 = plt.figure(2, figsize=(15, 8))
    # Bins2 = 'auto'
    Bins2 = np.arange(2, np.max(S_90_limit)+0.1, 0.1)
    n_limit_S, bins2, patches2 = plt.hist(S_90_limit, bins=Bins2, histtype='step', color='b',
                                          label='number of virt. experiments = {0:.0f}'.format(number_of_entries))
    plt.axvline(S_90_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(S_90_mean), color='b')
    plt.xticks(np.arange(2, np.max(S_90_limit)+0.5, 0.5))
    plt.xlabel("90 percent limit of number of observed signal events S")
    plt.ylabel("counts")
    plt.title("Distribution of the 90 % upper limit of the signal contribution for DM with mass = {0:.1f} MeV"
              .format(DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_S90.png")
    plt.close()


    """ Display S_90_limit with 68 % and 95% limit in histogram: """
    h7 = plt.figure(7, figsize=(15, 8))
    # Bins7 = 'auto'
    Bins7 = np.arange(2, np.max(S_90_limit)+0.1, 0.1)
    n_limit_S7, bins7, patches7 = plt.hist(S_90_limit, bins=Bins7, histtype='step', color='b',
                                           label='number of virt. experiments = {0:.0f}'.format(number_of_entries))
    plt.axvline(S_90_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(S_90_mean), color='b')
    # define interpolation function:
    f_limit_S = interp1d(bins7[:-1], n_limit_S7, kind='previous')
    bins7_small = np.arange(bins7[0], bins7[-2], 0.01)
    n_limit_S7_small = f_limit_S(bins7_small)

    # add 95 % interval to the figure:
    plt.axvline(S_90_2_5, linestyle='-.', label='2.5% probability limit = {0:.4f}'.format(S_90_2_5), color='b')
    plt.axvline(S_90_97_5, linestyle='-.', label='97.5% probability limit = {0:.4f}'.format(S_90_97_5), color='b')
    # define a boolean array, which is True between S_90_2_5 and S_90_97_5 (array of bool):
    int_limit_S_95percent = (bins7_small >= S_90_2_5) * (bins7_small <= S_90_97_5)
    plt.fill_between(bins7_small, 0, n_limit_S7_small, where=int_limit_S_95percent, facecolor='yellow', alpha=1,
                     step='post')

    # add 68 % interval to the figure:
    plt.axvline(S_90_16, linestyle=':', label='16% probability limit = {0:.4f}'.format(S_90_16), color='b')
    plt.axvline(S_90_84, linestyle=':', label='84% probability limit = {0:.4f}'.format(S_90_84), color='b')
    # define a boolean array, which is True between S_90_16 and S_90_84 (array of bool):
    int_limit_S_68percent = (bins7_small >= S_90_16) * (bins7_small <= S_90_84)
    plt.fill_between(bins7_small, 0, n_limit_S7_small, where=int_limit_S_68percent, facecolor='green', alpha=1, step='post')

    plt.xticks(np.arange(2, np.max(S_90_limit)+0.5, 0.5))
    plt.xlabel("90 percent limit of number of observed signal events S")
    plt.ylabel("counts")
    plt.title("Distribution of the 90 % upper limit of the signal contribution for DM with mass = {0:.1f} MeV"
              .format(DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_S90_CL.png")
    plt.close()


    """ Display DSNB_mean in histogram: """
    h3 = plt.figure(3, figsize=(15, 8))
    Bins3 = 'auto'
    # Bins3 = np.arange(0, 25, 1)
    n_DSNB, bins3, patches3 = plt.hist(DSNB_mode, bins=Bins3, histtype='step', color='b',
                                       label='number of virt. experiments = {0:.0f}'
                                       .format(number_of_entries))
    plt.axvline(DSNB_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
                .format(DSNB_expected), color='r')
    plt.axvline(DSNB_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(DSNB_mean),
                color='b')
    # define interpolation function:
    f_DSNB = interp1d(bins3[:-1], n_DSNB, kind='previous')
    bins3_small = np.arange(bins3[0], bins3[-2], 0.01)
    n_DSNB_small = f_DSNB(bins3_small)

    # add 95 % interval to the figure:
    plt.axvline(DSNB_2_5, linestyle='-.', label='2.5% probability limit (-2$\sigma$) = {0:.4f}'.format(DSNB_2_5), color='b')
    plt.axvline(DSNB_97_5, linestyle='-.', label='97.5% probability limit (+2$\sigma$) = {0:.4f}'.format(DSNB_97_5),
                color='b')
    # define a boolean array, which is True between DSNB_2_5 and DSNB_97_5 (array of bool):
    int_DSNB_95percent = (bins3_small >= DSNB_2_5) * (bins3_small <= DSNB_97_5)
    plt.fill_between(bins3_small, 0, n_DSNB_small, where=int_DSNB_95percent, facecolor='yellow', alpha=1, step='post')

    # add 68 % interval to the figure:
    plt.axvline(DSNB_16, linestyle=':', label='16% probability limit (-1$\sigma$) = {0:.4f}'.format(DSNB_16), color='b')
    plt.axvline(DSNB_84, linestyle=':', label='84% probability limit (+1$\sigma$) = {0:.4f}'.format(DSNB_84), color='b')
    # define a boolean array, which is True between DSNB_16 and DSNB_84 (array of bool):
    int_DSNB_68percent = (bins3_small >= DSNB_16) * (bins3_small <= DSNB_84)
    plt.fill_between(bins3_small, 0, n_DSNB_small, where=int_DSNB_68percent, facecolor='green', alpha=1, step='post')

    plt.xticks(np.arange(0, np.max(DSNB_mode)+2.5, 2.5))
    plt.xlabel("total number of observed DSNB background events")
    plt.ylabel("counts")
    plt.title("Distribution of the observed number of DSNB background events in virtual experiments "
              "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
              .format(lower_energy_bound, upper_energy_bound, DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_DSNB.png")
    plt.close()


    """ Display CCatmo_mean in histogram: """
    h4 = plt.figure(4, figsize=(15, 8))
    Bins4 = 'auto'
    # Bins4 = np.arange(0, 3, 0.1)
    n_CCatmo, bins4, patches4 = plt.hist(CCatmo_mode, bins=Bins4, histtype='step', color='b',
                                         label='number of virt. experiments = {0:.0f}'
                                         .format(number_of_entries))
    plt.axvline(CCatmo_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
                .format(CCatmo_expected), color='r')
    plt.axvline(CCatmo_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(CCatmo_mean),
                color='b')
    # define interpolation function:
    f_CCatmo = interp1d(bins4[:-1], n_CCatmo, kind='previous')
    bins4_small = np.arange(bins4[0], bins4[-2], 0.01)
    n_CCatmo_small = f_CCatmo(bins4_small)

    # add 95 % interval to the figure:
    plt.axvline(CCatmo_2_5, linestyle='-.', label='2.5% probability limit (-2$\sigma$) = {0:.4f}'.format(CCatmo_2_5),
                color='b')
    plt.axvline(CCatmo_97_5, linestyle='-.', label='97.5% probability limit (+2$\sigma$) = {0:.4f}'.format(CCatmo_97_5),
                color='b')
    # define a boolean array, which is True between CCatmo_2_5 and CCatmo_97_5 (array of bool):
    int_CCatmo_95percent = (bins4_small >= CCatmo_2_5) * (bins4_small <= CCatmo_97_5)
    plt.fill_between(bins4_small, 0, n_CCatmo_small, where=int_CCatmo_95percent, facecolor='yellow', alpha=1, step='post')

    # add 68 % interval t the figure:
    plt.axvline(CCatmo_16, linestyle=':', label='16% probability limit (-1$\sigma$) = {0:.4f}'.format(CCatmo_16), color='b')
    plt.axvline(CCatmo_84, linestyle=':', label='84% probability limit (+1$\sigma$) = {0:.4f}'.format(CCatmo_84), color='b')
    # define a boolean array, which is True between CCatmo_16 and CCatmo_84 (array of bool):
    int_CCatmo_68percent = (bins4_small >= CCatmo_16) * (bins4_small <= CCatmo_84)
    plt.fill_between(bins4_small, 0, n_CCatmo_small, where=int_CCatmo_68percent, facecolor='green', alpha=1, step='post')

    plt.xticks(np.arange(15, np.max(CCatmo_mode)+2.5, 2.5))
    plt.xlabel("total number of observed atmo. CC background events")
    plt.ylabel("counts")
    plt.title("Distribution of the observed number of atmospheric CC background events in virtual experiments "
              "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
              .format(lower_energy_bound, upper_energy_bound, DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_CCatmo.png")
    plt.close()


    """ Display Reactor_mean in histogram: """
    h5 = plt.figure(5, figsize=(15, 8))
    Bins5 = 'auto'
    # Bins5 = np.arange(0, 0.5, 0.01)
    n_Reactor, bins5, patches5 = plt.hist(Reactor_mode, bins=Bins5, histtype='step', color='b',
                                          label='number of virt. experiments = {0:.0f}'
                                          .format(number_of_entries))
    plt.axvline(Reactor_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
                .format(Reactor_expected), color='r')
    plt.axvline(Reactor_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(Reactor_mean),
                color='b')
    # define interpolation function:
    f_Reactor = interp1d(bins5[:-1], n_Reactor, kind='previous')
    bins5_small = np.arange(bins5[0], bins5[-2], 0.01)
    n_Reactor_small = f_Reactor(bins5_small)

    # add 95 % interval to the figure:
    plt.axvline(Reactor_2_5, linestyle='-.', label='2.5% probability limit (-2$\sigma$) = {0:.4f}'.format(Reactor_2_5),
                color='b')
    plt.axvline(Reactor_97_5, linestyle='-.', label='97.5% probability limit (+2$\sigma$) = {0:.4f}'.format(Reactor_97_5),
                color='b')
    # define a boolean array, which is True between Reactor_2_5 and Reactor_97_5 (array of bool):
    int_Reactor_95percent = (bins5_small >= Reactor_2_5) * (bins5_small <= Reactor_97_5)
    plt.fill_between(bins5_small, 0, n_Reactor_small, where=int_Reactor_95percent, facecolor='yellow', alpha=1, step='post')

    # add 68 % interval to the figure:
    plt.axvline(Reactor_16, linestyle=':', label='16% probability limit (-1$\sigma$) = {0:.4f}'.format(Reactor_16),
                color='b')
    plt.axvline(Reactor_84, linestyle=':', label='84% probability limit (+1$\sigma$) = {0:.4f}'.format(Reactor_84),
                color='b')
    # define a boolean array, which is True between Reactor_16 and Reactor_84 (array of bool):
    int_Reactor_68percent = (bins5_small >= Reactor_16) * (bins5_small <= Reactor_84)
    plt.fill_between(bins5_small, 0, n_Reactor_small, where=int_Reactor_68percent, facecolor='green', alpha=1, step='post')

    plt.xticks(np.arange(20, np.max(Reactor_mode)+2.5, 2.5))
    plt.xlabel("total number of observed reactor background events")
    plt.ylabel("counts")
    plt.title("Distribution of the observed number of reactor background events in virtual experiments "
              "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
              .format(lower_energy_bound, upper_energy_bound, DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_reactor.png")
    plt.close()

    """ Display NCatmo_mean in histogram: """
    h8 = plt.figure(8, figsize=(15, 8))
    Bins8 = 'auto'
    # Bins8 = np.arange(0, 3, 0.1)
    n_NCatmo, bins8, patches8 = plt.hist(NCatmo_mode, bins=Bins8, histtype='step', color='b',
                                         label='number of virt. experiments = {0:.0f}'
                                         .format(number_of_entries))
    plt.axvline(NCatmo_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
                .format(NCatmo_expected), color='r')
    plt.axvline(NCatmo_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(NCatmo_mean),
                color='b')
    # define interpolation function:
    f_NCatmo = interp1d(bins8[:-1], n_NCatmo, kind='previous')
    bins8_small = np.arange(bins8[0], bins8[-2], 0.01)
    n_NCatmo_small = f_NCatmo(bins8_small)

    # add 95 % interval to the figure:
    plt.axvline(NCatmo_2_5, linestyle='-.', label='2.5% probability limit (-2$\sigma$) = {0:.4f}'.format(NCatmo_2_5),
                color='b')
    plt.axvline(NCatmo_97_5, linestyle='-.', label='97.5% probability limit (+2$\sigma$) = {0:.4f}'.format(NCatmo_97_5),
                color='b')
    # define a boolean array, which is True between NCatmo_2_5 and NCatmo_97_5 (array of bool):
    int_NCatmo_95percent = (bins8_small >= NCatmo_2_5) * (bins8_small <= NCatmo_97_5)
    plt.fill_between(bins8_small, 0, n_NCatmo_small, where=int_NCatmo_95percent, facecolor='yellow', alpha=1, step='post')

    # add 68 % interval t the figure:
    plt.axvline(NCatmo_16, linestyle=':', label='16% probability limit (-1$\sigma$) = {0:.4f}'.format(NCatmo_16), color='b')
    plt.axvline(NCatmo_84, linestyle=':', label='84% probability limit (+1$\sigma$) = {0:.4f}'.format(NCatmo_84), color='b')
    # define a boolean array, which is True between NCatmo_16 and NCatmo_84 (array of bool):
    int_NCatmo_68percent = (bins8_small >= NCatmo_16) * (bins8_small <= NCatmo_84)
    plt.fill_between(bins8_small, 0, n_NCatmo_small, where=int_NCatmo_68percent, facecolor='green', alpha=1, step='post')

    plt.xticks(np.arange(0.0, np.max(NCatmo_mode)+0.5, 0.5))
    plt.xlabel("total number of observed atmo. NC background events")
    plt.ylabel("counts")
    plt.title("Distribution of the observed number of atmospheric NC background events in virtual experiments "
              "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
              .format(lower_energy_bound, upper_energy_bound, DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_NCatmo.png")
    plt.close()

    """ Display FN_mean in histogram: """
    h9 = plt.figure(9, figsize=(15, 8))
    Bins9 = 'auto'
    # Bins9 = np.arange(0, 3, 0.1)
    n_FN, bins9, patches9 = plt.hist(FN_mode, bins=Bins9, histtype='step', color='b',
                                     label='number of virt. experiments = {0:.0f}'.format(number_of_entries))
    plt.axvline(FN_expected, linestyle='dashed', label='expected number of events from simulation = {0:.3f}'
                .format(FN_expected), color='r')
    plt.axvline(FN_mean, linestyle='dashed', label='mean of the distribution = {0:.4f}'.format(FN_mean),
                color='b')
    # define interpolation function:
    f_FN = interp1d(bins9[:-1], n_FN, kind='previous')
    bins9_small = np.arange(bins9[0], bins9[-2], 0.00001)
    n_FN_small = f_FN(bins9_small)

    # add 95 % interval to the figure:
    plt.axvline(FN_2_5, linestyle='-.', label='2.5% probability limit (-2$\sigma$) = {0:.4f}'.format(FN_2_5),
                color='b')
    plt.axvline(FN_97_5, linestyle='-.', label='97.5% probability limit (+2$\sigma$) = {0:.4f}'.format(FN_97_5),
                color='b')
    # define a boolean array, which is True between FN_2_5 and FN_97_5 (array of bool):
    int_FN_95percent = (bins9_small >= FN_2_5) * (bins9_small <= FN_97_5)
    plt.fill_between(bins9_small, 0, n_FN_small, where=int_FN_95percent, facecolor='yellow', alpha=1, step='post')

    # add 68 % interval t the figure:
    plt.axvline(FN_16, linestyle=':', label='16% probability limit (-1$\sigma$) = {0:.4f}'.format(FN_16), color='b')
    plt.axvline(FN_84, linestyle=':', label='84% probability limit (+1$\sigma$) = {0:.4f}'.format(FN_84), color='b')
    # define a boolean array, which is True between FN_16 and FN_84 (array of bool):
    int_FN_68percent = (bins9_small >= FN_16) * (bins9_small <= FN_84)
    plt.fill_between(bins9_small, 0, n_FN_small, where=int_FN_68percent, facecolor='green', alpha=1, step='post')

    # plt.xticks(np.arange(0.0, np.max(CCatmo_mode)+0.01, 0.01))
    plt.xlabel("total number of observed fast neutron background events")
    plt.ylabel("counts")
    plt.title("Distribution of the observed number of fast neutron background events in virtual experiments "
              "\n(for DM mass of {2:.1f} MeV and in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
              .format(lower_energy_bound, upper_energy_bound, DM_mass))
    plt.legend()
    plt.savefig(path_result + "/result_FN.png")
    plt.close()

    # plt.show()
