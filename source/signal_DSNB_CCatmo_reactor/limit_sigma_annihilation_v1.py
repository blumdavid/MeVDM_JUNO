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

# path of the folder, where the simulated spectra are saved (string):
path_expected_spectrum = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2"

# path of the directory, where the results of the analysis are saved (string):
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor"

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

    """ the structure of the result_dataset_output_{}.txt file is differs for masses divisible by ten (20,30,40,50,60,
        70,80,90,100) to masses NOT divisible by ten (25,35,45,55,65,75,85,95).
        For masses divisible by ten, mean_limit_S90 = result[7] and std_limit_S90 = result[8].
        For masses NOT divisible by ten, mean_limit_S90 = result[8] and std_limit_S90 = result[9].
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

    # Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
    # mass of "mass" MeV in cm**2 (float):
    limit = limit_annihilation_crosssection(mean_limit_S90, mass, J_avg, N_target, time_s,
                                            epsilon_IBD, MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    print(limit)

    # append the value of limit_sigma_anni_90 of one DM mass to the array (np.array of float):
    limit_sigma_anni = np.append(limit_sigma_anni, limit)


# maximum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_max = 10**(-23)
# minimum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_min = 10**(-26)

h1 = plt.figure(1)
plt.semilogy(DM_mass, limit_sigma_anni, marker='x', markersize='6.0', linestyle='-', color='black',
             label='90% upper limit on $<\sigma_A v>$\n')
plt.fill_between(DM_mass, y_max, limit_sigma_anni, facecolor="grey", alpha=0.8)
plt.axhline(sigma_anni_natural, linestyle='--', color='black',
            label='natural scale of the annihilation cross-section\n($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.4, hatch='/')
plt.ylim(y_min, y_max)
plt.xlabel("Dark Matter mass in MeV")
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$")
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment")
plt.legend()
plt.grid()
plt.show()
