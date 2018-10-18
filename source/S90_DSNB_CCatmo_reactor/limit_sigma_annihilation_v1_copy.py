""" This script is a copy of the script "limit_sigma_annihilation_v1.py" from 04.10.2018.
    It was used for simulation and analysis of "S90_DSNB_CCatmo_reactor".


    Script to calculate the 90 percent dark matter self-annihilation cross-section for different dark matter masses
    and display the results.

    Results of the simulation and analysis saved in folder "S90_DSNB_CCatmo_reactor"

    Information about the expected signal and background spectra generated with gen_spectrum_v2.py:

    Background: - reactor electron-antineutrinos background
                - DSNB
                - atmospheric Charged Current electron-antineutrino background

    Diffuse Supernova Neutrino Background:
    - expected spectrum of DSNB is saved in file: DSNB_EmeanNuXbar22_bin100keV.txt
    - information about the expected spectrum is saved in file: DSNB_info_EmeanNuXbar22_100keV.txt

    Reactor electron-antineutrino Background:
    - expected spectrum of reactor background is saved in file: Reactor_NH_power36_bin100keV.txt
    - information about the reactor background spectrum is saved in file: Reactor_info_NH_power36_bin100keV.txt

    Atmospheric Charged Current electron-antineutrino Background:
    - expected spectrum of atmospheric CC background is saved in file: CCatmo_Osc1_bin100keV.txt
    - information about the atmospheric CC background spectrum is saved in file: CCatmo_info_Osc1_bin100keV.txt
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from work.MeVDM_JUNO.source.gen_spectrum_functions import limit_annihilation_crosssection
from work.MeVDM_JUNO.source.gen_spectrum_functions import limit_neutrino_flux

# TODO-me: Interpretation of the limit on the annihilation cross-section according to the 'natural scale'
# INFO-me: the self-annihilation cross-section (times the relative velocity) necessary to explain the observed
# INFO-me: abundance of DM in the Universe is ~ 3e-26 cm^3/s

# TODO: make sure, that mean_limit_S90 and std_limit_s90 in the result_dataset_output_{}.txt files are
# TODO: at the correct position! (index [7] and [8], OR index [8] and [9])

# Define variable ASIMOV. If ASIMOV is True, also the results of the Asimov-approach is displayed:
ASIMOV = False

# Define variable JAVG. If JAVG is True, also the results of the limit of annihilation cross-section as function of the
# DM mass depending on different angular-averaged DM intensities J_avg over the whole Milky Way:
JAVG = True

# path of the folder, where the simulated spectra are saved (string):
path_expected_spectrum = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2"

# path of the directory, where the results of the analysis are saved (string):
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor"

# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536

# set the DM mass in MeV (float):
DM_mass = np.arange(20, 105, 5)

""" Preallocation of mean of 90% distribution: """
# Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section is saved:
limit_sigma_anni = np.array([])
# Preallocate the array, where the mean of the 90% upper limit of the number of signal events is saved:
limit_S_90 = np.array([])
# Preallocate the array, where the 90 % upper limit of the electron-antineutrino flux is saved:
limit_flux = np.array([])

""" Preallocation of mean of 90% distribution fro different values of J_avg: """
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

""" Preallocation for the Asimov approach: """
# Preallocate the array, where the mean of the 90 % upper limit of the number of signal events from the Asimov-approach
# is saved:
limit_S_90_asimov = np.array([])
# Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section from the Asimov-approach
# is saved:
limit_sigma_anni_asimov = np.array([])
# Preallocate the array, where the 90 % upper limit of the electron_antineutrino flux from the Asimov-approach
# is saved:
limit_flux_asimov = np.array([])

# Loop over the different DM masses:
for mass in DM_mass:

    """ Information about the work saved in dataset_output_{}: 
        Dark matter with mass = "mass" MeV

        Neutrino signal from Dark Matter annihilation in the Milky Way:
        - expected spectrum of the signal is saved in file: signal_DMmass{}_bin100keV.txt
        - information about the signal spectrum is saved in file: signal_info_DMmass{}_bin100keV.txt
    """
    # load information about the signal spectrum (np.array of float):
    info_signal = np.loadtxt(path_expected_spectrum + "/signal_info_DMmass{0}_bin100keV.txt".format(mass))
    # exposure time in years:
    time_year = info_signal[6]
    # exposure time in seconds:
    time_s = time_year * 3.156 * 10 ** 7
    # number of targets in JUNO (float):
    N_target = info_signal[7]
    # IBD detection efficiency in JUNO (float):
    epsilon_IBD = info_signal[8]

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
    J_avg = info_signal[13]

    # natural scale of the DM self-annihilation cross-section (times the relative velocity) necessary to explain
    # the observed abundance of DM in the Universe, in cm**3/s (float):
    sigma_anni_natural = info_signal[12]

    # path of the dataset output folder (string):
    path_dataset_output = path_folder + "/dataset_output_{0}".format(mass)
    # path of the file, where the result of the analysis is saved (string):
    path_result = path_dataset_output + "/result_mcmc/result_dataset_output_{0}.txt".format(mass)
    # load the result file (np.array of float):
    result = np.loadtxt(path_result)

    """ the structure of the result_dataset_output_{}.txt file differs for masses divisible by ten (20,30,40,50,60,
        70,80,90,100) to masses NOT divisible by ten (25,35,45,55,65,75,85,95).
        For masses divisible by ten, mean_limit_S90 = result[7] and std_limit_S90 = result[8].
        For masses NOT divisible by ten, mean_limit_S90 = result[8] and std_limit_S90 = result[9].
    """
    """
    # if mass is divisible by ten:
    if (mass % 10) == 0:
        # mean of the 90% probability limit of the number of signal events (float):
        mean_limit_S90 = result[7]
        # standard deviation of the 90% probability limit of the number of signal events (float):
        std_limit_S90 = result[8]
    else:
        # mean of the 90% probability limit of the number of signal events (float):
        mean_limit_S90 = result[8]
        # standard deviation of the 90% probability limit of the number of signal events (float):
        std_limit_S90 = result[9]
    """

    # mean of the 90% probability limit of the number of signal events (float):
    mean_limit_S90 = result[8]
    # standard deviation of the 90% probability limit of the number of signal events (float):
    std_limit_S90 = result[9]
    # 16 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_16 = result[10]
    # 84 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_84 = result[11]
    # 2.5 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_2_5 = result[31]
    # 97.5 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_97_5 = result[32]
    # 0.15 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_0_15 = result[33]
    # 99.85 % probability value of the 90% probability limit of the number of signal events (float):
    limit_S90_99_85 = result[34]

    """ Mean of 90 % distribution: """
    # append the value of mean_limit_S90 of one DM mass to the array (np.array of float):
    limit_S_90 = np.append(limit_S_90, mean_limit_S90)

    # Calculate the 90 % probability limit of the electron-antineutrino flux from DM self-annihilation in the entire
    # Milky Way for DM with mass of "mass" MeV in electron-neutrinos/(cm**2 * s) (float):
    flux_limit = limit_neutrino_flux(mean_limit_S90, mass, N_target, time_s, epsilon_IBD,
                                     MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the value of flux_limit of one DM mass to the array (np.array of float):
    limit_flux = np.append(limit_flux, flux_limit)

    # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
    # mass of "mass" MeV in cm**2 (float):
    limit = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg, N_target, time_s,
                                            epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
    limit_sigma_anni = np.append(limit_sigma_anni, limit)

    if JAVG:
        """ canonical value """
        # Calculate the 90 % probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_canonical (float):
        limit_Javg_canonical = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_canonical, N_target, time_s,
                                                               epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_Javg_canonical of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_canonical = np.append(limit_sigma_anni_Javg_canonical, limit_Javg_canonical)

        """ NFW profile """
        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_NFW_min (float):
        limit_Javg_NFW_min = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_NFW_min, N_target, time_s,
                                                             epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_NFW_min = np.append(limit_sigma_anni_Javg_NFW_min, limit_Javg_NFW_min)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_NFW (float):
        limit_Javg_NFW = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_NFW, N_target, time_s,
                                                         epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_NFW = np.append(limit_sigma_anni_Javg_NFW, limit_Javg_NFW)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_NFW_max (float):
        limit_Javg_NFW_max = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_NFW_max, N_target, time_s,
                                                             epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_NFW_max = np.append(limit_sigma_anni_Javg_NFW_max, limit_Javg_NFW_max)

        """ MQGSL profile """
        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_MQGSL_min (float):
        limit_Javg_MQGSL_min = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_MQGSL_min, N_target, time_s,
                                                               epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_MQGSL_min = np.append(limit_sigma_anni_Javg_MQGSL_min, limit_Javg_MQGSL_min)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_MQGSL (float):
        limit_Javg_MQGSL = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_MQGSL, N_target, time_s,
                                                           epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_MQGSL = np.append(limit_sigma_anni_Javg_MQGSL, limit_Javg_MQGSL)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_MQGSL_max (float):
        limit_Javg_MQGSL_max = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_MQGSL_max, N_target, time_s,
                                                               epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_MQGSL_max = np.append(limit_sigma_anni_Javg_MQGSL_max, limit_Javg_MQGSL_max)

        """ KKBP profile """
        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_KKBP_min (float):
        limit_Javg_KKBP_min = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_KKBP_min, N_target, time_s,
                                                              epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_KKBP_min = np.append(limit_sigma_anni_Javg_KKBP_min, limit_Javg_KKBP_min)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_KKBP (float):
        limit_Javg_KKBP = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_KKBP, N_target, time_s,
                                                          epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_KKBP = np.append(limit_sigma_anni_Javg_KKBP, limit_Javg_KKBP)

        # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
        # mass of "mass" MeV in cm**2 and for J_avg_KKBP_max (float):
        limit_Javg_KKBP_max = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg_KKBP_max, N_target, time_s,
                                                              epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
        # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
        limit_sigma_anni_Javg_KKBP_max = np.append(limit_sigma_anni_Javg_KKBP_max, limit_Javg_KKBP_max)

    """ 16 % and 84 % probability interval of 90% distribution: """
    # append the values of limit_S90_16 and limit_S90_84 of one DM mass to the arrays (np.array of float):
    limit_S_90_16 = np.append(limit_S_90_16, limit_S90_16)
    limit_S_90_84 = np.append(limit_S_90_84, limit_S90_84)

    # Calculate the 16% and 84% probability interval of the 90% probability limit of the electron-antineutrino flux
    # from DM self-annihilation in the entire Milky Way for DM with mass of "mass" MeV
    # in electron-neutrinos/(cm**2 * s) (float):
    flux_limit_16 = limit_neutrino_flux(limit_S90_16, mass, N_target, time_s, epsilon_IBD,
                                        MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    flux_limit_84 = limit_neutrino_flux(limit_S90_84, mass, N_target, time_s, epsilon_IBD,
                                        MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the values of flux_limit_16 and flux_limit_84 of one Dm mass to the arrays (np.array of float):
    limit_flux_16 = np.append(limit_flux_16, flux_limit_16)
    limit_flux_84 = np.append(limit_flux_84, flux_limit_84)

    # Calculate the 16% and 84% probability limit of the 90% probability limit of the averaged DM self-annihilation
    # cross-section for DM with mass of "mass" MeV in cm**2 (float):
    limit_16 = limit_annihilation_crosssection(limit_S90_16, mass, J_avg, N_target, time_s, epsilon_IBD,
                                               MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_84 = limit_annihilation_crosssection(limit_S90_84, mass, J_avg, N_target, time_s, epsilon_IBD,
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
    flux_limit_2_5 = limit_neutrino_flux(limit_S90_2_5, mass, N_target, time_s, epsilon_IBD,
                                         MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    flux_limit_97_5 = limit_neutrino_flux(limit_S90_97_5, mass, N_target, time_s, epsilon_IBD,
                                          MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the values of flux_limit_2_5 and flux_limit_97_5 of one Dm mass to the arrays (np.array of float):
    limit_flux_2_5 = np.append(limit_flux_2_5, flux_limit_2_5)
    limit_flux_97_5 = np.append(limit_flux_97_5, flux_limit_97_5)

    # Calculate the 2.5% and 97.5% probability limit of the 90% probability limit of the averaged DM self-annihilation
    # cross-section for DM with mass of "mass" MeV in cm**2 (float):
    limit_2_5 = limit_annihilation_crosssection(limit_S90_2_5, mass, J_avg, N_target, time_s, epsilon_IBD,
                                                MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_97_5 = limit_annihilation_crosssection(limit_S90_97_5, mass, J_avg, N_target, time_s, epsilon_IBD,
                                                 MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

    # append the values of limit_2_5 and limit_97_5 of one DM mass to the arrays (np.array of float):
    limit_sigma_anni_2_5 = np.append(limit_sigma_anni_2_5, limit_2_5)
    limit_sigma_anni_97_5 = np.append(limit_sigma_anni_97_5, limit_97_5)

if ASIMOV:

    DM_mass_2 = np.arange(20, 110, 10)

    for mass_2 in DM_mass_2:
        result_2 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor/Asimov_dataset_output/"
                              "analysis_mcmc_{0}MeV/Dataset1_mcmc_analysis.txt".format(mass_2))

        # get the 90 % limit of the number of signal events
        limit_S90_asimov = result_2[1]

        # calculate the 90 % probability limit of the electron-antineutrino flux from DM self-annihilation in the entire
        # Milky Way from DM with mass of "DM_mass_2" MeV in electron-antineutrinos/(cm**2 * s) (float):
        flux_limit_asimov = limit_neutrino_flux(limit_S90_asimov, mass_2, N_target, time_s, epsilon_IBD,
                                                MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

        # Calculate the 90 % probability limit of the averaged DM self-annihilation cross-section for DM with mass
        # of "DM_mass_2" MeV in cm**2 (float):
        limit_asimov = limit_annihilation_crosssection(limit_S90_asimov, mass_2, J_avg, N_target, time_s,
                                                       epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)

        # Append the values to the preallocated arrays:
        limit_S_90_asimov = np.append(limit_S_90_asimov, limit_S90_asimov)
        limit_flux_asimov = np.append(limit_flux_asimov, flux_limit_asimov)
        limit_sigma_anni_asimov = np.append(limit_sigma_anni_asimov, limit_asimov)

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way, obtained from Super-K Data.
    (th have used the canonical value J_avg = 5, the results are digitized from figure 1, on page 7 of the paper 
    'Testing MeV Dark Matter with neutrino detectors', arXiv: 0710.5420v1)
    The digitized data is saved in "/home/astro/blum/PhD/paper/SuperKamiokande/limit_SuperK_digitized.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_SuperK = np.array([13.3846, 13.5385, 13.6923, 13.6923, 14.1538, 15.2308, 15.8462, 16.7692, 17.6923, 19.0769,
                           20, 20.6154, 21.5385, 22.4615, 23.5385, 24.4615, 25.5385, 27.3846, 29.0769, 35.2308,
                           37.3846, 39.8462, 42.3077, 43.8462, 45.5385, 47.2308, 48.9231, 51.8462, 54.6154, 57.2308,
                           59.8462, 62.6154, 65.5385, 67.2308, 69.0769, 72.4615, 75.6923, 78.9231, 82, 84.9231,
                           87.6923, 90.1538, 92.6154, 95.6923, 98.7692, 104.154, 108.923, 113.385, 119.385, 120.769,
                           121.538, 122])
# 90% limit of the self-annihilation cross-section (np.array of float):
sigma_anni_SuperK = np.array([4.74756E-23, 4.21697E-23, 2.7617E-23, 3.68279E-23, 1.08834E-23, 1.83953E-24, 7.62699E-25,
                              3.10919E-25, 1.63394E-25, 8.58665E-26, 6.43908E-26, 5.62341E-26, 4.99493E-26,
                              4.66786E-26, 4.51244E-26, 4.58949E-26, 4.82863E-26, 5.62341E-26, 6.54902E-26,
                              1.18448E-25, 1.50131E-25, 1.83953E-25, 2.17889E-25, 2.29242E-25, 2.33156E-25,
                              2.33156E-25, 2.25393E-25, 2.00203E-25, 1.63394E-25, 1.28912E-25, 1.03444E-25,
                              8.44249E-26, 7.49894E-26, 7.24927E-26, 7.24927E-26, 7.62699E-26, 8.44249E-26,
                              9.66705E-26, 1.14505E-25, 1.42696E-25, 1.77828E-25, 2.29242E-25, 3.00567E-25,
                              4.58949E-25, 7.12756E-25, 1.77828E-24, 4.66786E-24, 1.22528E-23, 4.99493E-23,
                              7.00791E-23, 8.30076E-23, 9.83212E-23])

# maximum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_max = 10 ** (-22)
# minimum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_min = 10 ** (-26)

# increase distance between plot and title:
rcParams["axes.titlepad"] = 20

""" Semi-logarithmic plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO with 
    probability intervals: """
h1 = plt.figure(1, figsize=(15, 8))
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='blue', linewidth=2.0,
             label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.semilogy(DM_mass, limit_sigma_anni_16, linestyle=':', color='blue')
plt.fill_between(DM_mass, limit_sigma_anni_16, limit_sigma_anni, facecolor="green", alpha=1)
plt.semilogy(DM_mass, limit_sigma_anni_84, linestyle=':', color='blue')
plt.fill_between(DM_mass, limit_sigma_anni, limit_sigma_anni_84, facecolor='green', alpha=1,
                 label="68 % probability interval")
plt.semilogy(DM_mass, limit_sigma_anni_2_5, linestyle='-.', color='blue')
plt.fill_between(DM_mass, limit_sigma_anni_2_5, limit_sigma_anni_16, facecolor='yellow', alpha=1)
plt.semilogy(DM_mass, limit_sigma_anni_97_5, linestyle='-.', color='blue')
plt.fill_between(DM_mass, limit_sigma_anni_84, limit_sigma_anni_97_5, facecolor='yellow', alpha=1,
                 label='95 % probability interval')
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" Semi-logarithmic plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO: """
h2 = plt.figure(2, figsize=(15, 8))
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
             label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, limit_sigma_anni, y_max, facecolor='red', alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" Semi-log. plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO AND Super-K: """
h3 = plt.figure(3, figsize=(15, 8))
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
             label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, y_max, limit_sigma_anni, facecolor="red", alpha=0.4)
plt.semilogy(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black', linewidth=2.0,
             label="90% C.L. limit on $<\sigma_A v>$, obtained from Super-K data (arXiv:0710.5420)")
plt.fill_between(DM_mass_SuperK, y_max, sigma_anni_SuperK, facecolor="gray", alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass_SuperK, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass_SuperK), np.max(DM_mass_SuperK))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO AND Super-K: """
h4 = plt.figure(4, figsize=(15, 8))
plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, y_max, limit_sigma_anni, facecolor="red", alpha=0.4)
plt.plot(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black', linewidth=2.0,
         label="90% C.L. limit on $<\sigma_A v>$, obtained from Super-K data (arXiv:0710.5420)")
plt.fill_between(DM_mass_SuperK, y_max, sigma_anni_SuperK, facecolor="gray", alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, 3 * 10 ** (-25))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO with probability intervals: """
h5 = plt.figure(5, figsize=(15, 8))
plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='blue', linewidth=2.0,
         label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.plot(DM_mass, limit_sigma_anni_16, linestyle=':', color='blue')
plt.fill_between(DM_mass, limit_sigma_anni_16, limit_sigma_anni, facecolor="green", alpha=1)
plt.plot(DM_mass, limit_sigma_anni_84, linestyle=':', color='blue')
plt.fill_between(DM_mass, limit_sigma_anni, limit_sigma_anni_84, facecolor='green', alpha=1,
                 label="68 % probability interval")
plt.plot(DM_mass, limit_sigma_anni_2_5, linestyle='-.', color='blue')
plt.fill_between(DM_mass, limit_sigma_anni_2_5, limit_sigma_anni_16, facecolor='yellow', alpha=1)
plt.plot(DM_mass, limit_sigma_anni_97_5, linestyle='-.', color='blue')
plt.fill_between(DM_mass, limit_sigma_anni_84, limit_sigma_anni_97_5, facecolor='yellow', alpha=1,
                 label='95 % probability interval')
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, 3 * 10 ** (-25))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO: """
h6 = plt.figure(6, figsize=(15, 8))
plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, 3 * 10 ** (-25), limit_sigma_anni, facecolor="red", alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, 3 * 10 ** (-25))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" Plot of the mean of the 90% probability limit of the number of signal events from JUNO 
    with probability intervals: """
h7 = plt.figure(7, figsize=(15, 8))
plt.plot(DM_mass, limit_S_90, marker='x', markersize='6.0', linestyle='-', color='blue', linewidth=2.0,
         label='90% upper limit on number of signal events ($S_{90}$), simulated for JUNO')
plt.plot(DM_mass, limit_S_90_16, linestyle=':', color='blue')
plt.fill_between(DM_mass, limit_S_90_16, limit_S_90, facecolor="green", alpha=1)
plt.plot(DM_mass, limit_S_90_84, linestyle=':', color='blue')
plt.fill_between(DM_mass, limit_S_90, limit_S_90_84, facecolor='green', alpha=1,
                 label="68 % probability interval")
plt.plot(DM_mass, limit_S_90_2_5, linestyle='-.', color='blue')
plt.fill_between(DM_mass, limit_S_90_2_5, limit_S_90_16, facecolor='yellow', alpha=1)
plt.plot(DM_mass, limit_S_90_97_5, linestyle='-.', color='blue')
plt.fill_between(DM_mass, limit_S_90_84, limit_S_90_97_5, facecolor='yellow', alpha=1,
                 label='95 % probability interval')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, 10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$S_{90}$ in events", fontsize=15)
plt.title("90% upper probability limit on the number of signal events from the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" Plot of the mean of the 90% probability limit of the number of signal events from JUNO: """
h8 = plt.figure(8, figsize=(15, 8))
plt.plot(DM_mass, limit_S_90, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit on number of signal events ($S_{90}$), simulated for JUNO')
plt.fill_between(DM_mass, 10, limit_S_90, facecolor="red", alpha=0.4)
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, 10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$S_{90}$ in events", fontsize=15)
plt.title("90% upper probability limit on the number of signal events from the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" Plot of the 90% upper limit of the electron-antineutrino flux from DM annihilation in the Milky Way with 
    probability intervals: """
h9 = plt.figure(9, figsize=(15, 8))
plt.plot(DM_mass, limit_flux, marker='x', markersize='6.0', linestyle='-', color='blue', linewidth=2.0,
         label='90% upper limit on $\\bar{\\nu}_{e}$-flux')
plt.plot(DM_mass, limit_flux_16, linestyle=':', color='blue')
plt.fill_between(DM_mass, limit_flux_16, limit_flux, facecolor="green", alpha=1)
plt.plot(DM_mass, limit_flux_84, linestyle=':', color='blue')
plt.fill_between(DM_mass, limit_flux, limit_flux_84, facecolor='green', alpha=1,
                 label="68 % probability interval")
plt.plot(DM_mass, limit_flux_2_5, linestyle='-.', color='blue')
plt.fill_between(DM_mass, limit_flux_2_5, limit_flux_16, facecolor='yellow', alpha=1)
plt.plot(DM_mass, limit_flux_97_5, linestyle='-.', color='blue')
plt.fill_between(DM_mass, limit_flux_84, limit_flux_97_5, facecolor='yellow', alpha=1,
                 label='95 % probability interval')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, 0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("90% upper limit of $\phi_{\\bar{\\nu}_{e}}$ in $\\bar{\\nu}_{e}/(cm^{2}s)$", fontsize=15)
plt.title("90% upper probability limit on electron-antineutrino flux from DM self-annihilation in the Milky Way",
          fontsize=20)
plt.legend(fontsize=13)
plt.grid()

""" Plot of the 90% upper limit of the electron-antineutrino flux from DM annihilation in the Milky Way: """
h10 = plt.figure(10, figsize=(15, 8))
plt.plot(DM_mass, limit_flux, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit on $\\bar{\\nu}_{e}$-flux')
plt.fill_between(DM_mass, 0.6, limit_flux, facecolor="red", alpha=0.4)
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, 0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("90% upper limit of $\phi_{\\bar{\\nu}_{e}}$ in $\\bar{\\nu}_{e}/(cm^{2}s)$", fontsize=15)
plt.title("90% upper probability limit on electron-antineutrino flux from DM self-annihilation in the Milky Way",
          fontsize=20)
plt.legend(fontsize=13)
plt.grid()


if ASIMOV:
    # Plot of the mean of the 90% probability limit of the number of signal events from JUNO:
    h11 = plt.figure(11, figsize=(15, 8))
    plt.plot(DM_mass, limit_S_90, marker='x', markersize='6.0', linestyle='-', color='blue',
             label='90% upper limit on number of signal events ($S_{90}$), simulated for JUNO')
    plt.plot(DM_mass_2, limit_S_90_asimov, marker="x", markersize="6.0", linestyle="-", color="red",
             label="90% upper limit from Asimov approach test")
    plt.xlim(np.min(DM_mass), np.max(DM_mass))
    plt.ylim(0, 10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=15)
    plt.ylabel("$S_{90}$ in events", fontsize=15)
    plt.title(
        "90% upper probability limit on the number of signal events from the JUNO experiment\n (with Asimov test)",
        fontsize=20)
    plt.legend()
    plt.grid()

    # Plot of the 90% upper limit of the electron-antineutrino flux from DM annihilation in the Milky Way:
    h12 = plt.figure(12, figsize=(15, 8))
    plt.plot(DM_mass, limit_flux, marker='x', markersize='6.0', linestyle='-', color='blue',
             label='90% upper limit on $\\bar{\\nu}_{e}$-flux')
    plt.plot(DM_mass_2, limit_flux_asimov, marker='x', markersize='6.0', linestyle='-', color='red',
             label='90% upper limit on $\\bar{\\nu}_{e}$-flux from Asimov test')
    plt.xlim(np.min(DM_mass), np.max(DM_mass))
    plt.ylim(0, 0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=15)
    plt.ylabel("90% upper limit of $\phi_{\\bar{\\nu}_{e}}$ in $\\bar{\\nu}_{e}/(cm^{2}s)$", fontsize=15)
    plt.title("90% upper probability limit on electron-antineutrino flux from DM self-annihilation in the Milky Way\n "
              "(with Asimov test)", fontsize=20)
    plt.legend()
    plt.grid()

    # Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO:
    h13 = plt.figure(13, figsize=(15, 8))
    plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='blue',
             label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
    plt.plot(DM_mass_2, limit_sigma_anni_asimov, marker='x', markersize='6.0', linestyle='-', color='red',
             label='90% upper limit on $<\sigma_A v>$, simulated for JUNO (from Asimov test)')
    plt.axhline(sigma_anni_natural, linestyle=':', color='black',
                label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
    plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
    plt.xlim(np.min(DM_mass), np.max(DM_mass))
    plt.ylim(y_min, 3 * 10 ** (-25))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=15)
    plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
    plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment\n (with Asimov "
              "test)", fontsize=20)
    plt.legend()
    plt.grid()

if JAVG:
    # minimum of y axis:
    y_min_2 = 0
    # maximum of y axis:
    y_max_2 = 4.5 * 10 ** (-25)

    """ Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO for different values of J_avg 
    from the NFW profile: """
    h14 = plt.figure(14, figsize=(15, 8))
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
                label='natural scale of the annihilation cross-section\n($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
    plt.fill_between(DM_mass, y_min_2, sigma_anni_natural, facecolor="grey", alpha=0.5, hatch='/')
    plt.xlim(np.min(DM_mass), np.max(DM_mass))
    plt.ylim(y_min_2, y_max_2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=15)
    plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
    plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment\n"
              "for the NFW halo profile", fontsize=20)
    plt.legend(fontsize=13)
    plt.grid()

    """ Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO for different values of J_avg 
    from the MQGSL profile: """
    h15 = plt.figure(15, figsize=(15, 8))
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
                label='natural scale of the annihilation cross-section\n($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
    plt.fill_between(DM_mass, y_min_2, sigma_anni_natural, facecolor="grey", alpha=0.5, hatch='/')
    plt.xlim(np.min(DM_mass), np.max(DM_mass))
    plt.ylim(y_min_2, y_max_2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=15)
    plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
    plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment\n"
              "for the MQGSL halo profile", fontsize=20)
    plt.legend(fontsize=13)
    plt.grid()

    """ Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO for different values of J_avg 
    from the KKBP profile: """
    h16 = plt.figure(16, figsize=(15, 8))
    # plot canonical:
    plt.plot(DM_mass, limit_sigma_anni_Javg_canonical, marker='x', markersize='6.0', linestyle='-', color='black',
             linewidth=2.0, label='canonical value $J_{avg}$' + '={0:.1f}'.format(J_avg_canonical))
    # plot MQGSL:
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
                label='natural scale of the annihilation cross-section\n($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
    plt.fill_between(DM_mass, y_min_2, sigma_anni_natural, facecolor="grey", alpha=0.5, hatch='/')
    plt.xlim(np.min(DM_mass), np.max(DM_mass))
    plt.ylim(y_min_2, y_max_2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Dark Matter mass in MeV", fontsize=15)
    plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
    plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment\n"
              "for the KKBP halo profile", fontsize=20)
    plt.legend(fontsize=13)
    plt.grid()

plt.show()
