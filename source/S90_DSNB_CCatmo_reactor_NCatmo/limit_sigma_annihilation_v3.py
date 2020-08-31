""" Script "limit_sigma_annihilation_v3.py" from 12.05.2020.

    It was used for simulation and analysis of "S90_DSNB_CCatmo_reactor_NCatmo".

    Script to calculate the 90 percent dark matter self-annihilation cross-section for different dark matter masses
    and display the results.

    Results of the simulation and analysis saved in folder "S90_DSNB_CCatmo_reactor_NCatmo"

    Information about the expected signal and background spectra generated with gen_spectrum_v4.py:

    Background: - reactor electron-antineutrinos background
                - DSNB
                - atmospheric Charged Current electron-antineutrino background on protons
                - atmospheric Neutral Current background (on C12)
                - atmospheric Charged Current background on C12

    Diffuse Supernova Neutrino Background:
    - expected spectrum of DSNB is saved in file: DSNB_bin500keV_PSD.txt
    - information about the expected spectrum is saved in file: DSNB_info_500keV_PSD.txt

    Reactor electron-antineutrino Background:
    - expected spectrum of reactor background is saved in file: Reactor_NH_power36_bin500keV_PSD.txt
    - information about the reactor background spectrum is saved in file: Reactor_info_NH_power36_bin500keV_PSD.txt

    Atmospheric Charged Current electron-antineutrino Background on protons:
    - expected spectrum of atmospheric CC background is saved in file: CCatmo_onlyP_Osc1_bin500keV_PSD.txt
    - information about the atmospheric CC background spectrum is saved in file:
        CCatmo_onlyP_info_Osc1_bin500keV_PSD.txt

    Atmospheric Neutral Current Background:
    - expected spectrum of atmospheric NC background is saved in file: NCatmo_onlyC12_wPSD99_bin500keV.txt
    - information about the atmo. NC background spectrum is saved in file: NCatmo_info_onlyC12_wPSD99_bin500keV.txt

    Atmospheric Charged Current Background on C12:
    - expected spectrum of atmospheric CC background is saved in file: CCatmo_onlyC12_Osc1_bin500keV_PSD.txt
    - information about the atmospheric CC background spectrum is saved in file:
        CCatmo_onlyC12_info_Osc1_bin500keV_PSD.txt


"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from gen_spectrum_functions import limit_annihilation_crosssection_v2
from gen_spectrum_functions import limit_neutrino_flux_v2

# TODO-me: Interpretation of the limit on the annihilation cross-section according to the 'natural scale'
# INFO-me: the self-annihilation cross-section (times the relative velocity) necessary to explain the observed
# INFO-me: abundance of DM in the Universe is ~ 3e-26 cm^3/s

# TODO: make sure, that mean_limit_S90 and std_limit_s90 in the result_dataset_output_{}.txt files are
# TODO: at the correct position! (index [7] and [8], OR index [8] and [9])

# Define variable JAVG. If JAVG is True, also the results of the limit of annihilation cross-section as function of the
# DM mass depending on different angular-averaged DM intensities J_avg over the whole Milky Way:
JAVG = True

# path of the folder, where the simulated spectra are saved (string):
path_expected_spectrum = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN"
path_expected_spectrum_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
                            "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm" \
                            "_PSD99/test_10to20_20to30_30to40_40to100_final"

# path of the directory, where the results of the analysis are saved (string):
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo"

# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536

# set the DM mass in MeV (float):
DM_mass = np.arange(15, 105, 5)

""" Preallocation of mean of 90% distribution: """
# Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section is saved:
limit_sigma_anni = np.array([])
# Preallocate the array, where the mean of the 90% upper limit of the number of signal events is saved:
limit_S_90 = np.array([])
# Preallocate the array, where the 90 % upper limit of the electron-antineutrino flux is saved:
limit_flux = np.array([])

""" Preallocation of mean of 90% distribution for different values of J_avg: """
if JAVG:
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_canonical is
    # saved:
    limit_sigma_anni_Javg_canonical = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_NFW_min is
    # saved:
    limit_sigma_anni_Javg_NFW_min = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_NFW is
    # saved:
    limit_sigma_anni_Javg_NFW = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_NFW_max is
    # saved:
    limit_sigma_anni_Javg_NFW_max = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_MQGSL_min is
    # saved:
    limit_sigma_anni_Javg_MQGSL_min = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_MQGSL is
    # saved:
    limit_sigma_anni_Javg_MQGSL = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_MQGSL_max is
    # saved:
    limit_sigma_anni_Javg_MQGSL_max = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_KKBP_min is
    # saved:
    limit_sigma_anni_Javg_KKBP_min = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_KKBP is
    # saved:
    limit_sigma_anni_Javg_KKBP = np.array([])
    # Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section for J_avg_KKBP_max is
    # saved:
    limit_sigma_anni_Javg_KKBP_max = np.array([])

""" Preallocation of 16% probability value of 90 % distribution: """
# Preallocate the array, where 16% prob. value of the 90 % upper limit of the DM self-annihilation cross-section is
# saved:
limit_sigma_anni_16 = np.array([])
# Preallocate the array, where 16% prob. value of the 90% upper limit of the number of signal events is saved:
limit_S_90_16 = np.array([])
# Preallocate the array, where 16% prob. value of the 90 % upper limit of the electron-antineutrino flux is saved:
limit_flux_16 = np.array([])

""" Preallocation of 84% probability value of 90% distribution: """
# Preallocate the array, where the 84% prob. value of the 90 % upper limit of the DM self-annihilation cross-section
# is saved:
limit_sigma_anni_84 = np.array([])
# Preallocate the array, where the 84% prob. value of the 90% upper limit of the number of signal events is saved:
limit_S_90_84 = np.array([])
# Preallocate the array, where the 84% prob. value of the 90 % upper limit of the electron-antineutrino flux is saved:
limit_flux_84 = np.array([])

""" Preallocation of 2.5% probability value of 90% distribution: """
# Preallocate the array, where the 2.5% prob. value of the 90 % upper limit of the DM self-annihilation cross-section
# is saved:
limit_sigma_anni_2_5 = np.array([])
# Preallocate the array, where the 2.5% prob. value of the 90% upper limit of the number of signal events is saved:
limit_S_90_2_5 = np.array([])
# Preallocate the array, where the 2.5% prob. value of the 90 % upper limit of the electron-antineutrino flux is saved:
limit_flux_2_5 = np.array([])

""" Preallocation of 97.5% probability value of 90% distribution: """
# Preallocate the array, where the 97.5% prob. value of the 90 % upper limit of the DM self-annihilation cross-section
# is saved:
limit_sigma_anni_97_5 = np.array([])
# Preallocate the array, where the 97.5% prob value of the 90% upper limit of the number of signal events is saved:
limit_S_90_97_5 = np.array([])
# Preallocate the array, where the 97.5% prob. value of the 90 % upper limit of the electron-antineutrino flux is saved:
limit_flux_97_5 = np.array([])

# Loop over the different DM masses:
for mass in DM_mass:

    """ Information about the work saved in dataset_output_{}: 
        Dark matter with mass = "mass" MeV

        Neutrino signal from Dark Matter annihilation in the Milky Way:
        - expected spectrum of the signal is saved in file: signal_DMmass{}_bin500keV_PSD.txt
        - information about the signal spectrum is saved in file: signal_info_DMmass{}_bin500keV_PSD.txt
    """
    # load information about the signal spectrum (np.array of float):
    info_signal = np.loadtxt(path_expected_spectrum + "/signal_info_DMmass{0}_bin500keV_PSD.txt".format(mass))
    # exposure time in years:
    time_year = info_signal[6]
    # exposure time in seconds:
    time_s = time_year * 3.156 * 10 ** 7
    # number of targets in JUNO (float):
    N_target = info_signal[7]
    # IBD detection efficiency in JUNO (without muon veto cut) (float):
    epsilon_IBD = info_signal[8]
    # exposure ratio of muon veto cut (float):
    exposure_ratio_muon_veto = info_signal[17]

    # number of signal events before PSD:
    N_signal_wo_PSD = np.sum(np.loadtxt(path_expected_spectrum + "/signal_DMmass{0}_bin500keV.txt".format(mass)))
    # number of signal events after PSD:
    N_signal_w_PSD = np.sum(np.loadtxt(path_expected_spectrum + "/signal_DMmass{0}_bin500keV_PSD.txt".format(mass)))
    # PSD efficiency of this DM mass:
    PSD_efficiency = float(N_signal_w_PSD) / float(N_signal_wo_PSD)

    # For different values of angular-averaged DM intensity over whole Milky Way (float):
    if JAVG:
        # canonical value of J_avg (float)
        J_avg_canonical = 5.0

        # minimum of J_avg corresponding to NFW profile and to minimum density allowed for this profile by
        # observational constraints (float):
        J_avg_NFW_min = 1.3
        # average of J corresponding to NFW profile (float):
        J_avg_NFW = 3.0
        # maximum of J_avg corresponding to NFW profile and to minimum density allowed for this profile by
        # observational constraints (float):
        J_avg_NFW_max = 41.0

        # minimum of J_avg corresponding to MQGSL profile and to maximum density allowed for this profile by
        # observational constraints (float):
        J_avg_MQGSL_min = 5.2
        # average of J corresponding to MQGSL profile (float):
        J_avg_MQGSL = 8
        # maximum of J_avg corresponding to MQGSL profile and to maximum density allowed for this profile by
        # observational constraints (float):
        J_avg_MQGSL_max = 104

        # minimum of J_avg corresponding to KKBP profile and to minimum density allowed for this profile by
        # observational constraints (float):
        J_avg_KKBP_min = 1.9
        # average of J corresponding to KKBP profile (float):
        J_avg_KKBP = 2.6
        # maximum of J_avg corresponding to KKBP profile and to minimum density allowed for this profile by
        # observational constraints (float):
        J_avg_KKBP_max = 8.5

    # angular-averaged DM intensity over whole Milky Way (float):
    J_avg = info_signal[15]

    # natural scale of the DM self-annihilation cross-section (times the relative velocity) necessary to explain
    # the observed abundance of DM in the Universe, in cm**3/s (float):
    sigma_anni_natural = info_signal[14]

    # path of the dataset output folder (string):
    path_dataset_output = path_folder + "/dataset_output_{0}".format(mass)
    # path of the file, where the result of the analysis is saved (string):
    path_result = path_dataset_output + "/result_mcmc/result_dataset_output_{0}.txt".format(mass)
    # load the result file (np.array of float):
    result = np.loadtxt(path_result)

    # mean of the 90% probability limit of the number of signal events (float):
    mean_limit_S90 = result[8]
    # standard deviation of the 90% probability limit of the number of signal events (float):
    std_limit_S90 = result[9]
    # 16 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_16 = result[10]
    # 84 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_84 = result[11]
    # 2.5 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_2_5 = result[41]
    # 97.5 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_97_5 = result[42]
    # 0.15 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_0_15 = result[43]
    # 99.85 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_99_85 = result[44]

    """ Mean of 90 % distribution: """
    # append the value of mean_limit_S90 of one DM mass to the array (np.array of float):
    limit_S_90 = np.append(limit_S_90, mean_limit_S90)

    # Calculate the 90 % probability limit of the electron-antineutrino flux from DM self-annihilation in the entire
    # Milky Way for DM with mass of "mass" MeV in electron-neutrinos/(cm**2 * s) (float):
    flux_limit = limit_neutrino_flux_v2(mean_limit_S90, mass, N_target, time_s, epsilon_IBD, exposure_ratio_muon_veto,
                                        PSD_efficiency, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the value of flux_limit of one DM mass to the array (np.array of float):
    limit_flux = np.append(limit_flux, flux_limit)

    # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
    # mass of "mass" MeV in cm**2 (float):
    limit = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg, N_target, time_s, epsilon_IBD,
                                               exposure_ratio_muon_veto, PSD_efficiency, MASS_NEUTRON, MASS_PROTON,
                                               MASS_POSITRON)

    # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
    limit_sigma_anni = np.append(limit_sigma_anni, limit)

    if JAVG:
        """ canonical value """
        # Calculate the 90 % probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_canonical (float):
        limit_Javg_canonical = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_canonical, N_target,
                                                                  time_s, epsilon_IBD, exposure_ratio_muon_veto,
                                                                  PSD_efficiency, MASS_NEUTRON, MASS_PROTON,
                                                                  MASS_POSITRON)
        # append the value of limit_Javg_canonical of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_canonical = np.append(limit_sigma_anni_Javg_canonical, limit_Javg_canonical)

        """ NFW profile """
        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_NFW_min (float):
        limit_Javg_NFW_min = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_NFW_min, N_target, time_s,
                                                                epsilon_IBD, exposure_ratio_muon_veto, PSD_efficiency,
                                                                MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_NFW_min = np.append(limit_sigma_anni_Javg_NFW_min, limit_Javg_NFW_min)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_NFW (float):
        limit_Javg_NFW = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_NFW, N_target, time_s,
                                                            epsilon_IBD, exposure_ratio_muon_veto, PSD_efficiency,
                                                            MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_NFW = np.append(limit_sigma_anni_Javg_NFW, limit_Javg_NFW)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_NFW_max (float):
        limit_Javg_NFW_max = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_NFW_max, N_target, time_s,
                                                                epsilon_IBD, exposure_ratio_muon_veto, PSD_efficiency,
                                                                MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_NFW_max = np.append(limit_sigma_anni_Javg_NFW_max, limit_Javg_NFW_max)

        """ MQGSL profile """
        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_MQGSL_min (float):
        limit_Javg_MQGSL_min = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_MQGSL_min, N_target,
                                                                  time_s, epsilon_IBD, exposure_ratio_muon_veto,
                                                                  PSD_efficiency, MASS_NEUTRON, MASS_PROTON,
                                                                  MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_MQGSL_min = np.append(limit_sigma_anni_Javg_MQGSL_min, limit_Javg_MQGSL_min)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_MQGSL (float):
        limit_Javg_MQGSL = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_MQGSL, N_target, time_s,
                                                              epsilon_IBD, exposure_ratio_muon_veto, PSD_efficiency,
                                                              MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_MQGSL = np.append(limit_sigma_anni_Javg_MQGSL, limit_Javg_MQGSL)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_MQGSL_max (float):
        limit_Javg_MQGSL_max = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_MQGSL_max, N_target,
                                                                  time_s, epsilon_IBD, exposure_ratio_muon_veto,
                                                                  PSD_efficiency, MASS_NEUTRON, MASS_PROTON,
                                                                  MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_MQGSL_max = np.append(limit_sigma_anni_Javg_MQGSL_max, limit_Javg_MQGSL_max)

        """ KKBP profile """
        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_KKBP_min (float):
        limit_Javg_KKBP_min = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_KKBP_min, N_target, time_s,
                                                                 epsilon_IBD, exposure_ratio_muon_veto, PSD_efficiency,
                                                                 MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_KKBP_min = np.append(limit_sigma_anni_Javg_KKBP_min, limit_Javg_KKBP_min)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_KKBP (float):
        limit_Javg_KKBP = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_KKBP, N_target, time_s,
                                                             epsilon_IBD, exposure_ratio_muon_veto, PSD_efficiency,
                                                             MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_KKBP = np.append(limit_sigma_anni_Javg_KKBP, limit_Javg_KKBP)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_KKBP_max (float):
        limit_Javg_KKBP_max = limit_annihilation_crosssection_v2(mean_limit_S90, mass, J_avg_KKBP_max, N_target, time_s,
                                                                 epsilon_IBD, exposure_ratio_muon_veto, PSD_efficiency,
                                                                 MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_KKBP_max = np.append(limit_sigma_anni_Javg_KKBP_max, limit_Javg_KKBP_max)

    """ 16 % and 84 % probability interval of 90% distribution: """
    # append the values of limit_S90_16 and limit_S90_84 of one DM mass to the arrays (np.array of float):
    limit_S_90_16 = np.append(limit_S_90_16, limit_S90_16)
    limit_S_90_84 = np.append(limit_S_90_84, limit_S90_84)

    # Calculate the 16% and 84% probability interval of the 90% probability limit of the electron-antineutrino flux
    # from DM self-annihilation in the entire Milky Way for DM with mass of "mass" MeV
    # in electron-neutrinos/(cm**2 * s) (float):
    flux_limit_16 = limit_neutrino_flux_v2(limit_S90_16, mass, N_target, time_s, epsilon_IBD, exposure_ratio_muon_veto,
                                           PSD_efficiency, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    flux_limit_84 = limit_neutrino_flux_v2(limit_S90_84, mass, N_target, time_s, epsilon_IBD, exposure_ratio_muon_veto,
                                           PSD_efficiency, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the values of flux_limit_16 and flux_limit_84 of one Dm mass to the arrays (np.array of float):
    limit_flux_16 = np.append(limit_flux_16, flux_limit_16)
    limit_flux_84 = np.append(limit_flux_84, flux_limit_84)

    # Calculate the 16% and 84% probability limit of the 90% probability limit of the averaged DM self-annihilation
    # cross-section for DM with mass of "mass" MeV in cm**2 (float):
    limit_16 = limit_annihilation_crosssection_v2(limit_S90_16, mass, J_avg, N_target, time_s, epsilon_IBD,
                                                  exposure_ratio_muon_veto, PSD_efficiency,
                                                  MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_84 = limit_annihilation_crosssection_v2(limit_S90_84, mass, J_avg, N_target, time_s, epsilon_IBD,
                                                  exposure_ratio_muon_veto, PSD_efficiency,
                                                  MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the values of limit_16 and limit_84 of one DM mass to the arrays (np.array of float):
    limit_sigma_anni_16 = np.append(limit_sigma_anni_16, limit_16)
    limit_sigma_anni_84 = np.append(limit_sigma_anni_84, limit_84)

    """ 2.5 % and 97.5 % probability interval of 90% distribution: """
    # append the values of limit_S90_2_5 and limit_S90_2_5 of one DM mass to the arrays (np.array of float):
    limit_S_90_2_5 = np.append(limit_S_90_2_5, limit_S90_2_5)
    limit_S_90_97_5 = np.append(limit_S_90_97_5, limit_S90_97_5)

    # Calculate the 2.5% and 97.5% probability interval of the 90% probability limit of the electron-antineutrino flux
    # from DM self-annihilation in the entire Milky Way for DM with mass of "mass" MeV
    # in electron-neutrinos/(cm**2 * s) (float):
    flux_limit_2_5 = limit_neutrino_flux_v2(limit_S90_2_5, mass, N_target, time_s, epsilon_IBD,
                                            exposure_ratio_muon_veto, PSD_efficiency,
                                            MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    flux_limit_97_5 = limit_neutrino_flux_v2(limit_S90_97_5, mass, N_target, time_s, epsilon_IBD,
                                             exposure_ratio_muon_veto, PSD_efficiency,
                                             MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the values of flux_limit_2_5 and flux_limit_97_5 of one Dm mass to the arrays (np.array of float):
    limit_flux_2_5 = np.append(limit_flux_2_5, flux_limit_2_5)
    limit_flux_97_5 = np.append(limit_flux_97_5, flux_limit_97_5)

    # Calculate the 2.5% and 97.5% probability limit of the 90% probability limit of the averaged DM self-annihilation
    # cross-section for DM with mass of "mass" MeV in cm**2 (float):
    limit_2_5 = limit_annihilation_crosssection_v2(limit_S90_2_5, mass, J_avg, N_target, time_s, epsilon_IBD,
                                                   exposure_ratio_muon_veto, PSD_efficiency,
                                                   MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_97_5 = limit_annihilation_crosssection_v2(limit_S90_97_5, mass, J_avg, N_target, time_s, epsilon_IBD,
                                                    exposure_ratio_muon_veto, PSD_efficiency,
                                                    MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the values of limit_2_5 and limit_97_5 of one DM mass to the arrays (np.array of float):
    limit_sigma_anni_2_5 = np.append(limit_sigma_anni_2_5, limit_2_5)
    limit_sigma_anni_97_5 = np.append(limit_sigma_anni_97_5, limit_97_5)

""" Save values of limit_sigma_anni to txt file: """
np.savetxt(path_folder + "/limit_annihilation_JUNO.txt", limit_sigma_anni, fmt='%.5e',
           header="90 % upper limit of the DM annihilation cross-section from JUNO experiment in cm^3/2:\n"
                  "Values calculated with script limit_sigma_annihilation_v3.py;\n"
                  "Analysis done with script analyze_spectra_v7_server2.py and auto_output_analysis_v3.py;\n"
                  "1000 datasets have been analyzed;\n"
                  "Output in folder {0};\n"
                  "Spectra used for the analysis from "
                  "'/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/'\n"
                  "and /home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/\n"
                  "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/\n"
                  "test_10to20_20to30_30to40_40to100_final/:\n"
                  "signal_DMmass()_bin500keV_PSD.txt,\n"
                  "DSNB_bin500keV_PSD.txt,\n"
                  "Reactor_NH_power36_bin500keV_PSD.txt,\n"
                  "CCatmo_onlyP_Osc1_bin500keV_PSD.txt,\n"
                  "CCatmo_onlyC12_Osc1_bin500keV_PSD.txt,\n"
                  "NCatmo_onlyC12_wPSD99_bin500keV.txt;\n"
                  "DM masses in MeV = {1};\n"
                  "{2} years of data taking, number of free protons = {3};\n"
                  "IBD efficiency = {4}, muon veto efficiency = {5};\n"
                  "canonical value J_avg = {6}:"
           .format(path_folder, DM_mass, time_year, N_target, epsilon_IBD, exposure_ratio_muon_veto, J_avg))

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way, obtained from Super-K Data.
    New Super-K limit of 2853 days = 7.82 years of data (the old limit from 0710.5420 was for only 1496 days of data).
    (they have used the canonical value J_avg = 5, the results are digitized from figure 1, on page 4 of the paper 
    'Dark matter-neutrino interactions through the lens of their cosmological implications', PhysRevD.97.075039)
    The digitized data is saved in "/home/astro/blum/PhD/paper/MeV_DM/SuperK_limit_new.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_SuperK = np.array([10.5384, 10.9966, 10.9966, 11.2257, 11.9129, 12.8293, 14.433, 15.1203, 16.2658, 16.953,
                           17.4112, 18.0985, 18.5567, 19.244, 20.3895, 21.5349, 22.9095, 23.8259, 24.9714, 25.8877,
                           27.2623, 28.1787, 28.866, 30.2405, 32.5315, 33.9061, 35.7388, 37.5716, 39.4044, 41.4662,
                           43.5281, 45.3608, 47.1936, 49.2554, 50.4009, 51.7755, 53.3792, 54.9828, 57.9611, 60.9393,
                           62.543, 64.3757, 66.8958, 69.874, 72.6231, 75.3723, 78.5796, 81.7869, 85.2234, 88.6598,
                           91.4089, 94.1581, 98.5109, 100.802])
# 90% limit of the self-annihilation cross-section in cm^3/s (np.array of float):
sigma_anni_SuperK = np.array([2.50761E-22, 2.20703E-22, 1.90157E-22, 1.38191E-22, 4.19682E-23, 8.15221E-24, 1.5502E-24,
                              8.18698E-25, 4.32374E-25, 3.3493E-25, 3.01122E-25, 2.70727E-25, 2.59447E-25, 2.59447E-25,
                              2.76551E-25, 2.94782E-25, 3.01122E-25, 2.88575E-25, 2.70727E-25, 2.53984E-25, 2.48636E-25,
                              2.53984E-25, 2.70727E-25, 3.14215E-25, 4.51174E-25, 5.23647E-25, 5.82438E-25, 6.07762E-25,
                              5.94965E-25, 5.94965E-25, 6.34188E-25, 7.51892E-25, 9.30201E-25, 1.10284E-24, 1.17555E-24,
                              1.17555E-24, 1.1508E-24, 1.07962E-24, 9.10614E-25, 7.51892E-25, 6.90537E-25, 6.61763E-25,
                              6.75997E-25, 7.3606E-25, 8.36308E-25, 9.10614E-25, 9.30201E-25, 8.91441E-25, 8.54296E-25,
                              8.7267E-25, 9.70646E-25, 1.1508E-24, 1.72424E-24, 2.22589E-24])

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way expected for 
    Hyper-Kamiokande after 10 years.
    (they have used the canonical value J_avg = 5, the results are digitized from figure 1, on page 2 of the paper 
    'Implications of a Dark Matter-Neutrino Coupling at Hyper–Kamiokande', Arxiv:1805.09830)
    The digitized data is saved in "/home/astro/blum/PhD/paper/MeV_DM/HyperK_limit_no_Gd.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_HyperK = np.array([11.4202, 11.4898, 11.5711, 11.6313, 11.6313, 11.7241, 11.7859, 11.9793, 12.0118, 12.1336,
                           12.2554, 12.4642, 12.4642, 12.6034, 12.7426, 12.7774, 12.8795, 13.0142, 13.0906, 13.2298,
                           13.4386, 13.6473, 13.7865, 13.9953, 14.1323, 14.2737, 14.4129, 14.5544, 14.6936, 14.7771,
                           14.9719, 15.1785, 15.4337, 15.6657, 16.5727, 17.1665, 17.8641, 19.0992, 20.8513, 22.2884,
                           23.9136, 25.4049, 26.8199, 28.4739, 30.005, 31.5361, 33.0673, 34.5984, 36.1922, 37.6607,
                           39.1919, 40.723, 42.3168, 43.8201, 45.3164, 46.8476, 48.4251, 49.7794, 52.0071, 54.2249,
                           55.7561, 57.1908, 59.418, 60.9394, 62.4374, 63.9686, 65.4997, 67.0309, 68.562, 70.0931,
                           71.6243, 73.1554, 74.7534, 76.281, 77.782, 79.3132, 80.8443, 82.3423, 83.8734, 85.4046,
                           86.9357, 88.4234, 89.998, 91.5292, 93.123, 94.5915, 96.0961, 97.6537, 99.1849, 100.716])

# 90 % limit of the self-annihilation cross-section in cm^3/s (array of float):
sigma_anni_HyperK = np.array([2.81969E-23, 2.3984E-23, 2.07369E-23, 1.79792E-23, 1.5389E-23, 1.34158E-23, 1.08796E-23,
                              9.07639E-24, 7.67308E-24, 6.07612E-24, 5.10954E-24, 4.29904E-24, 3.72832E-24, 3.1529E-24,
                              2.75079E-24, 2.42671E-24, 2.07014E-24, 1.76387E-24, 1.53958E-24, 1.34482E-24,
                              1.07845E-24, 8.76268E-25, 7.47593E-25, 6.49867E-25, 5.60194E-25, 4.82895E-25,
                              4.18958E-25, 3.64793E-25, 3.08801E-25, 2.6732E-25, 2.2808E-25, 2.00699E-25, 1.5469E-25,
                              1.32772E-25, 6.85055E-26, 5.56656E-26, 4.55841E-26, 3.74336E-26, 3.91269E-26,
                              4.31199E-26, 4.78795E-26, 5.27168E-26, 5.75659E-26, 6.33309E-26, 6.96262E-26,
                              7.63859E-26, 8.27672E-26, 8.99002E-26, 9.75563E-26, 1.05156E-25, 1.12134E-25, 1.207E-25,
                              1.28126E-25, 1.34863E-25, 1.40667E-25, 1.42365E-25, 1.41514E-25, 1.35923E-25,
                              1.23903E-25, 1.07922E-25, 9.82287E-26, 8.90453E-26, 8.01206E-26, 7.52633E-26,
                              7.20222E-26, 7.08082E-26, 7.08747E-26, 7.22607E-26, 7.50706E-26, 7.95367E-26,
                              8.48912E-26, 9.1121E-26, 1.00294E-25, 1.12329E-25, 1.24761E-25, 1.38262E-25,
                              1.54204E-25, 1.7043E-25, 1.80407E-25, 1.87869E-25, 1.90304E-25, 1.90679E-25, 1.92784E-25,
                              1.96469E-25, 2.02367E-25, 2.13703E-25, 2.27027E-25, 2.43049E-25, 2.62985E-25,
                              2.81874E-25])

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way obtained from KamLAND data 
    for 2343 days = 6.42 years of data taking.
    (they have used the canonical value J_avg = 5, the results are digitized from figure 5, on page 19 of the paper 
    'Search for extraterrestrial antineutrino sources with the KamLAND detector', Arxiv:1105.3516)
    The digitized data is saved in "/home/astro/blum/PhD/paper/KamLand/limit_KamLAND.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_Kamland = np.array([10.321, 11.8765, 12.4691, 13.284, 13.8025, 14.2469, 14.7654, 15.358, 15.358, 15.6543,
                            15.9506, 16.6173, 17.8025, 18.4691, 19.2099, 19.9506, 20.6173, 21.358, 22.0988, 22.3951,
                            22.6173, 23.0617, 23.4321, 23.9506, 24.6914, 25.4321, 26.0988, 26.8395, 27.5802, 28.321,
                            29.8765])

# 90 % limit of the self-annihilation cross-section in cm^3/s (array of float):
sigma_anni_Kamland = np.array([1.18275E-24, 9.19504E-25, 1.34141E-24, 1.95691E-24, 2.21943E-24, 2.36361E-24,
                               2.29039E-24, 1.89629E-24, 2.01948E-24, 1.89629E-24, 1.72545E-24, 1.42856E-24,
                               1.04285E-24, 9.48901E-25, 8.91017E-25, 9.79238E-25, 1.22056E-24, 1.52136E-24,
                               1.72545E-24, 1.67199E-24, 1.57E-24, 1.29986E-24, 1.04285E-24, 9.19504E-25, 8.91017E-25,
                               9.79238E-25, 1.1106E-24, 1.18275E-24, 1.14611E-24, 9.79238E-25, 3.55838E-24])

# maximum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_max = 10 ** (-22)
# minimum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_min = 10 ** (-26)

# increase distance between plot and title:
rcParams["axes.titlepad"] = 20

""" Semi-logarithmic plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO with 
    probability intervals: """
h1 = plt.figure(1, figsize=(10, 6))
plt.semilogy(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black', linewidth=2.0,
             label="90% C.L. limit from Super-K data (7.82 years)")
plt.semilogy(DM_mass_HyperK, sigma_anni_HyperK, linestyle=":", color='black', linewidth=2.0,
             label="90% C.L. limit simulated for Hyper-K (10 years)")
plt.semilogy(DM_mass_Kamland, sigma_anni_Kamland, linestyle="-", color='black', linewidth=2.0,
             label="90% C.L. limit from KamLAND data (6.42 years)")
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
             label='90% upper limit simulated for JUNO')
# plt.semilogy(DM_mass, limit_sigma_anni_16, linestyle=':', color='red')
plt.fill_between(DM_mass, limit_sigma_anni_16, limit_sigma_anni, facecolor="green", alpha=1)
# plt.semilogy(DM_mass, limit_sigma_anni_84, linestyle=':', color='red')
plt.fill_between(DM_mass, limit_sigma_anni, limit_sigma_anni_84, facecolor='green', alpha=1,
                 label="68 % probability interval")
# plt.semilogy(DM_mass, limit_sigma_anni_2_5, linestyle='-.', color='red')
plt.fill_between(DM_mass, limit_sigma_anni_2_5, limit_sigma_anni_16, facecolor='yellow', alpha=1)
# plt.semilogy(DM_mass, limit_sigma_anni_97_5, linestyle='-.', color='red')
plt.fill_between(DM_mass, limit_sigma_anni_84, limit_sigma_anni_97_5, facecolor='yellow', alpha=1,
                 label='95 % probability interval')
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='$<\\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$')
plt.fill_between(DM_mass_SuperK, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(10, np.max(DM_mass))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$<\\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_annihilation_JUNO_ProbInt_SuperK_HyperK_Kamland.png")
plt.close()

""" Semi-logarithmic plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO: """
h2 = plt.figure(2, figsize=(10, 6))
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
             label='90% upper limit on $<\\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, limit_sigma_anni, y_max, facecolor='red', alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section\n($<\\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass_SuperK, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(10, np.max(DM_mass))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$<\\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_annihilation_JUNO.png")
plt.close()

""" Semi-log. plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO, Super-K, Hyper-K and 
KamLAND: """
h3 = plt.figure(3, figsize=(10, 6))
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
             label='90% C.L. limit simulated for JUNO (10 years)')
plt.fill_between(DM_mass, limit_sigma_anni, y_max, facecolor='red', alpha=0.4)
plt.semilogy(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black', linewidth=2.0,
             label="90% C.L. limit from Super-K data (7.82 years)")
plt.semilogy(DM_mass_HyperK, sigma_anni_HyperK, linestyle=":", color='black', linewidth=2.0,
             label="90% C.L. limit simulated for Hyper-K (10 years)")
plt.semilogy(DM_mass_Kamland, sigma_anni_Kamland, linestyle="-", color='black', linewidth=2.0,
             label="90% C.L. limit from KamLAND data (6.42 years)")
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='$<\\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$')
plt.fill_between(DM_mass_SuperK, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(10, np.max(DM_mass))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$<\\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_annihilation_JUNO_SuperK_HyperK_Kamland.png")
plt.close()

""" plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO, Super-K, Hyper-K and KamLAND: """
# h4 = plt.figure(4, figsize=(15, 8))
# plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
#          label='90% upper limit simulated for JUNO')
# plt.fill_between(DM_mass, limit_sigma_anni, 4*10**(-24), facecolor='red', alpha=0.4)
# plt.plot(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black', linewidth=2.0,
#          label="90% C.L. limit from Super-K data (PhysRevD.97.075039)")
# plt.plot(DM_mass_HyperK, sigma_anni_HyperK, linestyle=":", color='black', linewidth=2.0,
#          label="90% C.L. limit simulated for Hyper-K (arXiv:1805.09830)")
# plt.plot(DM_mass_Kamland, sigma_anni_Kamland, linestyle="-", color='black', linewidth=2.0,
#          label="90% C.L. limit from KamLAND data (arXiv:1105.3516)")
# plt.axhline(sigma_anni_natural, linestyle=':', color='black',
#             label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
# plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
# plt.xlim(np.min(DM_mass), np.max(DM_mass))
# plt.ylim(y_min, 4*10**(-24))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("Dark Matter mass in MeV", fontsize=15)
# plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
# plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=20)
# plt.legend(fontsize=13)
# plt.grid()

""" Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO with probability intervals: """
h5 = plt.figure(5, figsize=(10, 6))
plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit simulated for JUNO')
# plt.plot(DM_mass, limit_sigma_anni_16, linestyle=':', color='red')
plt.fill_between(DM_mass, limit_sigma_anni_16, limit_sigma_anni, facecolor="green", alpha=1)
# plt.plot(DM_mass, limit_sigma_anni_84, linestyle=':', color='red')
plt.fill_between(DM_mass, limit_sigma_anni, limit_sigma_anni_84, facecolor='green', alpha=1,
                 label="68 % probability interval")
# plt.plot(DM_mass, limit_sigma_anni_2_5, linestyle='-.', color='red')
plt.fill_between(DM_mass, limit_sigma_anni_2_5, limit_sigma_anni_16, facecolor='yellow', alpha=1)
# plt.plot(DM_mass, limit_sigma_anni_97_5, linestyle='-.', color='red')
plt.fill_between(DM_mass, limit_sigma_anni_84, limit_sigma_anni_97_5, facecolor='yellow', alpha=1,
                 label='95 % probability interval')
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='$<\\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, np.max(limit_sigma_anni_97_5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$<\\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_annihilation_JUNO_ProbInt_noLog.png")
plt.close()

""" Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO: """
h6 = plt.figure(6, figsize=(10, 6))
plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit of $<\\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, 3 * 10 ** (-25), limit_sigma_anni, facecolor="red", alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section\n($<\\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, 3 * 10 ** (-25))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$<\\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_annihilation_JUNO_noLog.png")
plt.close()

""" Plot of the mean of the 90% probability limit of the number of signal events from JUNO 
    with probability intervals: """
h7 = plt.figure(7, figsize=(10, 6))
plt.plot(DM_mass, limit_S_90, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit on number of signal events ($\bar{S}_{90}$), simulated for JUNO')
# plt.plot(DM_mass, limit_S_90_16, linestyle=':', color='red')
plt.fill_between(DM_mass, limit_S_90_16, limit_S_90, facecolor="green", alpha=1)
# plt.plot(DM_mass, limit_S_90_84, linestyle=':', color='red')
plt.fill_between(DM_mass, limit_S_90, limit_S_90_84, facecolor='green', alpha=1,
                 label="68 % probability interval")
# plt.plot(DM_mass, limit_S_90_2_5, linestyle='-.', color='red')
plt.fill_between(DM_mass, limit_S_90_2_5, limit_S_90_16, facecolor='yellow', alpha=1)
# plt.plot(DM_mass, limit_S_90_97_5, linestyle='-.', color='red')
plt.fill_between(DM_mass, limit_S_90_84, limit_S_90_97_5, facecolor='yellow', alpha=1,
                 label='95 % probability interval')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, np.max(limit_S90_97_5)+0.5*np.max(limit_S90_97_5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$\bar{S}_{90}$", fontsize=13)
plt.title("90% upper probability limit on the number of signal events from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_S90_ProbInt.png")
plt.close()

""" Plot of the mean of the 90% probability limit of the number of signal events from JUNO: """
h8 = plt.figure(8, figsize=(10, 6))
plt.plot(DM_mass, limit_S_90, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit on number of signal events ($\bar{S}_{90}$), simulated for JUNO')
plt.fill_between(DM_mass, 10, limit_S_90, facecolor="red", alpha=0.4)
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, 10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$\bar{S}_{90}$", fontsize=13)
plt.title("90% upper probability limit on the number of signal events from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_S90.png")
plt.close()

""" Plot of the 90% upper limit of the electron-antineutrino flux from DM annihilation in the Milky Way with 
    probability intervals: """
h9 = plt.figure(9, figsize=(10, 6))
plt.semilogy(DM_mass, limit_flux, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit on $\\bar{\\nu}_{e}$-flux')
# plt.semilogy(DM_mass, limit_flux_16, linestyle=':', color='red')
plt.fill_between(DM_mass, limit_flux_16, limit_flux, facecolor="green", alpha=1)
# plt.semilogy(DM_mass, limit_flux_84, linestyle=':', color='red')
plt.fill_between(DM_mass, limit_flux, limit_flux_84, facecolor='green', alpha=1,
                 label="68 % probability interval")
# plt.semilogy(DM_mass, limit_flux_2_5, linestyle='-.', color='red')
plt.fill_between(DM_mass, limit_flux_2_5, limit_flux_16, facecolor='yellow', alpha=1)
# plt.semilogy(DM_mass, limit_flux_97_5, linestyle='-.', color='red')
plt.fill_between(DM_mass, limit_flux_84, limit_flux_97_5, facecolor='yellow', alpha=1,
                 label='95 % probability interval')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, np.max(limit_flux_97_5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("90% upper limit of $\\phi_{\\bar{\\nu}_{e}}$ in $\\bar{\\nu}_{e}/(cm^{2}s)$", fontsize=13)
plt.title("90% upper probability limit on electron-antineutrino flux from DM self-annihilation\n"
          "in the Milky Way",
          fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_flux_ProbInt.png")
plt.close()

""" Plot of the 90% upper limit of the electron-antineutrino flux from DM annihilation in the Milky Way: """
h10 = plt.figure(10, figsize=(10, 6))
plt.semilogy(DM_mass, limit_flux, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit on $\\bar{\\nu}_{e}$-flux')
plt.fill_between(DM_mass, np.max(limit_flux), limit_flux, facecolor="red", alpha=0.4)
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, np.max(limit_flux))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("90% upper limit of $\\phi_{\\bar{\\nu}_{e}}$ in $\\bar{\\nu}_{e}/(cm^{2}s)$", fontsize=13)
plt.title("90% upper probability limit on electron-antineutrino flux from DM self-annihilation\nin the Milky Way",
          fontsize=15)
plt.legend(fontsize=12)
plt.grid()
plt.savefig(path_folder + "/limit_flux.png")
plt.close()

if JAVG:
    # minimum of y axis:
    y_min_2 = 0

    """ Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO for different values of J_avg 
    from the NFW profile: """
    h14 = plt.figure(14, figsize=(10, 6))
    # plot canonical:
    plt.plot(DM_mass, limit_sigma_anni_Javg_canonical, marker='x', markersize='6.0', linestyle='-', color='black',
             linewidth=2.0, label='canonical value $J_{avg}$' + '={0:.1f}'.format(J_avg_canonical))
    # plot NFW:
    plt.plot(DM_mass, limit_sigma_anni_Javg_NFW_min, marker='x', markersize='6.0', linestyle='--', color='blue',
             linewidth=2.0, label='minimum of $J_{avg, NFW}$' + '={0:.1f}'.format(J_avg_NFW_min))
    plt.plot(DM_mass, limit_sigma_anni_Javg_NFW, marker='x', markersize='6.0', linestyle='-', color='blue',
             linewidth=2.0, label='average of $J_{avg, NFW}$' + '={0:.1f}'.format(J_avg_NFW))
    plt.plot(DM_mass, limit_sigma_anni_Javg_NFW_max, marker='x', markersize='6.0', linestyle=':', color='blue',
             linewidth=2.0, label='maximum of $J_{avg, NFW}$' + '={0:.1f}'.format(J_avg_NFW_max))
    plt.fill_between(DM_mass, limit_sigma_anni_Javg_NFW_min, limit_sigma_anni_Javg_NFW_max, facecolor='blue', alpha=0.5)
    # natural scale:
    plt.axhline(sigma_anni_natural, linestyle=':', color='black',
                label='$<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$')
    plt.fill_between(DM_mass_SuperK, y_min_2, sigma_anni_natural, facecolor="grey", alpha=0.5, hatch='/')
    plt.xlim(10, np.max(DM_mass))
    plt.ylim(y_min_2, np.max(limit_sigma_anni_Javg_NFW_min))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=13)
    plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
    plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment\n"
              "for the NFW halo profile", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(path_folder + "/limit_annihilation_Javg_NFW.png")
    plt.close()

    """ Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO for different values of J_avg 
    from the MQGSL profile: """
    h15 = plt.figure(15, figsize=(10, 6))
    # plot canonical:
    plt.plot(DM_mass, limit_sigma_anni_Javg_canonical, marker='x', markersize='6.0', linestyle='-', color='black',
             linewidth=2.0, label='canonical value $J_{avg}$' + '={0:.1f}'.format(J_avg_canonical))
    # plot MQGSL:
    plt.plot(DM_mass, limit_sigma_anni_Javg_MQGSL_min, marker='x', markersize='6.0', linestyle='--', color='red',
             linewidth=2.0, label='minimum of $J_{avg, MQGSL}$' + '={0:.1f}'.format(J_avg_MQGSL_min))
    plt.plot(DM_mass, limit_sigma_anni_Javg_MQGSL, marker='x', markersize='6.0', linestyle='-', color='red',
             linewidth=2.0, label='average of $J_{avg, MQGSL}$' + '={0:.1f}'.format(J_avg_MQGSL))
    plt.plot(DM_mass, limit_sigma_anni_Javg_MQGSL_max, marker='x', markersize='6.0', linestyle=':', color='red',
             linewidth=2.0, label='maximum of $J_{avg, MQGSL}$' + '={0:.1f}'.format(J_avg_MQGSL_max))
    plt.fill_between(DM_mass, limit_sigma_anni_Javg_MQGSL_min, limit_sigma_anni_Javg_MQGSL_max, facecolor='red',
                     alpha=0.5)
    # natural scale:
    plt.axhline(sigma_anni_natural, linestyle=':', color='black',
                label='$<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$')
    plt.fill_between(DM_mass_SuperK, y_min_2, sigma_anni_natural, facecolor="grey", alpha=0.5, hatch='/')
    plt.xlim(10, np.max(DM_mass))
    plt.ylim(y_min_2, np.max(limit_sigma_anni_Javg_canonical))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=13)
    plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
    plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment\n"
              "for the MQGSL halo profile", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(path_folder + "/limit_annihilation_Javg_MQGSL.png")
    plt.close()

    """ Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO for different values of J_avg 
    from the KKBP profile: """
    h16 = plt.figure(16, figsize=(10, 6))
    # plot canonical:
    plt.plot(DM_mass, limit_sigma_anni_Javg_canonical, marker='x', markersize='6.0', linestyle='-', color='black',
             linewidth=2.0, label='canonical value $J_{avg}$' + '={0:.1f}'.format(J_avg_canonical))
    # plot KKBP:
    plt.plot(DM_mass, limit_sigma_anni_Javg_KKBP_min, marker='x', markersize='6.0', linestyle='--', color='green',
             linewidth=2.0, label='minimum of $J_{avg, KKBP}$' + '={0:.1f}'.format(J_avg_KKBP_min))
    plt.plot(DM_mass, limit_sigma_anni_Javg_KKBP, marker='x', markersize='6.0', linestyle='-', color='green',
             linewidth=2.0, label='average of $J_{avg, KKBP}$' + '={0:.1f}'.format(J_avg_KKBP))
    plt.plot(DM_mass, limit_sigma_anni_Javg_KKBP_max, marker='x', markersize='6.0', linestyle=':', color='green',
             linewidth=2.0, label='maximum of $J_{avg, KKBP}$' + '={0:.1f}'.format(J_avg_KKBP_max))
    plt.fill_between(DM_mass, limit_sigma_anni_Javg_KKBP_min, limit_sigma_anni_Javg_KKBP_max, facecolor='green',
                     alpha=0.5)
    # natural scale:
    plt.axhline(sigma_anni_natural, linestyle=':', color='black',
                label='$<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$')
    plt.fill_between(DM_mass_SuperK, y_min_2, sigma_anni_natural, facecolor="grey", alpha=0.5, hatch='/')
    plt.xlim(10, np.max(DM_mass))
    plt.ylim(y_min_2, np.max(limit_sigma_anni_Javg_KKBP_min))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=13)
    plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
    plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment\n"
              "for the KKBP halo profile", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(path_folder + "/limit_annihilation_Javg_KKBP.png")
    plt.close()

    """ Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO for the minimum and maximum
        values of J_avg (minimum of NFW, maximum of MQGSL): """
    h17 = plt.figure(17, figsize=(10, 6))
    # plot canonical:
    plt.plot(DM_mass, limit_sigma_anni_Javg_canonical, marker='x', markersize='6.0', linestyle='-', color='red',
             linewidth=2.0, label='canonical value $J_{avg}$' + '={0:.1f}'.format(J_avg_canonical))
    # plot maximum:
    plt.plot(DM_mass, limit_sigma_anni_Javg_MQGSL_max, marker='x', markersize='6.0', linestyle=':', color='red',
             linewidth=2.0, label='maximum of $J_{avg}$' + '={0:.1f} (MQGSL model)'.format(J_avg_MQGSL_max))
    # plot minimum:
    plt.plot(DM_mass, limit_sigma_anni_Javg_NFW_min, marker='x', markersize='6.0', linestyle='--', color='red',
             linewidth=2.0, label='minimum of $J_{avg}$' + '={0:.1f} (NFW model)'.format(J_avg_NFW_min))
    plt.fill_between(DM_mass, limit_sigma_anni_Javg_NFW_min, limit_sigma_anni_Javg_MQGSL_max, facecolor='red',
                     alpha=0.5)
    # natural scale:
    plt.axhline(sigma_anni_natural, linestyle='-', color='black',
                label='$<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$')
    plt.fill_between(DM_mass_SuperK, y_min_2, sigma_anni_natural, facecolor="grey", alpha=0.5, hatch='/')
    plt.xlim(10, np.max(DM_mass))
    plt.ylim(y_min_2, np.max(limit_sigma_anni_Javg_KKBP_min))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=13)
    plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
    plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment\n"
              "for minimum and maximum values of $J_{avg}$", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(path_folder + "/limit_annihilation_Javg_min_max.png")
    plt.close()

# plt.show()
