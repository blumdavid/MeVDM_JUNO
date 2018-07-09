""" Script to calculate the 90 percent dark matter self-annihilation cross-section for different dark matter masses
    and display the results.

    Results of the simulation and analysis saved in folder "signal_DSNB_CCatmo_reactor"

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
from work.MeVDM_JUNO.source.gen_spectrum_functions import limit_annihilation_crosssection

# TODO-me: Interpretation of the limit on the annihilation cross-section according to the 'natural scale'
# INFO-me: the self-annihilation cross-section (times the relative velocity) necessary to explain the observed
# INFO-me: abundance of DM in the Universe is ~ 3e-26 cm^3/s

# TODO: make sure, that mean_limit_S90 and std_limit_s90 in the result_dataset_output_{}.txt files are
# TODO: at the correct position! (index [7] and [8], OR index [8] and [9])

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

# Preallocate the array, where the 90 % upper limit of the DM self-annihilation cross-section is saved:
limit_sigma_anni = np.array([])
# Preallocate the array, where the mean of the 90% upper limit of the number of signal events is saved:
limit_S_90 = np.array([])

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

    # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
    # mass of "mass" MeV in cm**2 (float):
    limit = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg, N_target, time_s,
                                            epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    print(limit)

    # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
    limit_sigma_anni = np.append(limit_sigma_anni, limit)

    # append the value of mean_limit_S90 of one DM mass to the array (np.array of float):
    limit_S_90 = np.append(limit_S_90, mean_limit_S90)


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
y_max = 10**(-22)
# minimum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_min = 10**(-26)

# Semi-logarithmic plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO:
h1 = plt.figure(1)
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red',
             label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, y_max, limit_sigma_anni, facecolor="red", alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, y_max)
plt.xlabel("Dark Matter mass in MeV")
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$")
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment")
plt.legend()
plt.grid()

# Semi-log. plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO AND Super-K:
h2 = plt.figure(2)
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red',
             label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, y_max, limit_sigma_anni, facecolor="red", alpha=0.4)
plt.semilogy(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black',
             label="90% C.L. limit on $<\sigma_A v>$, obtained from Super-K data")
plt.fill_between(DM_mass_SuperK, y_max, sigma_anni_SuperK, facecolor="gray", alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass_SuperK, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass_SuperK), np.max(DM_mass_SuperK))
plt.ylim(y_min, y_max)
plt.xlabel("Dark Matter mass in MeV")
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$")
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment")
plt.legend()
plt.grid()

# Semi-log. plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO AND Super-K:
h5 = plt.figure(5)
plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red',
         label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, y_max, limit_sigma_anni, facecolor="red", alpha=0.4)
plt.plot(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black',
         label="90% C.L. limit on $<\sigma_A v>$, obtained from Super-K data")
plt.fill_between(DM_mass_SuperK, y_max, sigma_anni_SuperK, facecolor="gray", alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, 3*10**(-25))
plt.xlabel("Dark Matter mass in MeV")
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$")
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment")
plt.legend()
plt.grid()

# Plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO:
h3 = plt.figure(3)
plt.plot(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='red',
         label='90% upper limit on $<\sigma_A v>$, simulated for JUNO')
plt.fill_between(DM_mass, 3*10**(-25), limit_sigma_anni, facecolor="red", alpha=0.4)
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, 3*10**(-25))
plt.xlabel("Dark Matter mass in MeV")
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$")
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment")
plt.legend()
plt.grid()

# Plot of the mean of the 90% probability limit of the number of signal events from JUNO:
h4 = plt.figure(4)
plt.plot(DM_mass, limit_S_90, marker='x', markersize='6.0', linestyle='-', color='blue',
         label='90% upper limit on number of signal events ($S_{90}$), simulated for JUNO')
plt.fill_between(DM_mass, 10, limit_S_90, facecolor="red", alpha=0.4)
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(0, 10)
plt.xlabel("Dark Matter mass in MeV")
plt.ylabel("$S_{90}$ in events")
plt.title("90% upper probability limit on the number of signal events from the JUNO experiment")
plt.legend()
plt.grid()

plt.show()
