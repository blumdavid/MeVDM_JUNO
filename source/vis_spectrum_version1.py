""" old script to simulate the visible electron-antineutrino spectrum
    -> do not use this script
    -> instead use the script vis_spectrum.py (gives same results but faster and better programmed
    """


# Script to calculate the visible electron-antineutrino spectrum in JUNO from electron-antineutrinos produced through
# annihilation of MeV dark matter in the Milky Way
#

# Import the necessary packages:
import numpy as np
from math import gamma
from matplotlib import pyplot

""" Variables that can be changed: """
# Dark Matter mass in MeV (float):
MASS_DM = 20.0
# NUMBER defines, how many "pseudo"-neutrinos are generated per real neutrino in the Script
# to get higher statistics (integer):
NUMBER = 1
# Number_visible defines, how many random numbers E_visible_".." are generated from the gaussian distribution
# "faltung" around E_neutrino_".." (integer):
NUMBER_VISIBLE = 1

""" List of variables used in the script: """
# energy corresponding to the electron-antineutrino energy in MeV (array of float64):
E1 = np.arange(3, 130, 0.01)
# energy corresponding to the visible energy in MeV, E2 defines the bins in pyplot.hist() (array of float64):
E2 = np.arange(1, 130, 1)

""" List of constants used in the script: """
# velocity of light in vacuum, in cm/s (float constant):
C_LIGHT = 2.998*10**10
# mass of positron in MeV (float constant):
MASS_POSITRON = 0.511
# mass of proton in MeV (float constant):
MASS_PROTON = 938.272
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.565
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
delta = MASS_NEUTRON - MASS_PROTON


""" List of assumptions or estimations made for the calculation of the signal spectrum:
1. neutrino flux (phi_signal) is calculated as described in paper 0710.5420

2. theoretical signal spectrum (N_neutrino) is calculated as described in paper 0710.5420:
2.1. only IBD on free protons is considered 
2.2. detection efficiency of IBD on free protons is considered to be 73 %
2.3. exposure time is set to 10 years
2.4. IBD cross-section is calculated with (naive +) approximation from paper 0302055, good approximation for 
     neutrino energies up to 300 MeV

3. Delta-function is approximated as very thin gaussian distribution 
3.1. Number of neutrinos with energy E_neutrino_signal is defined by round(float(N_neutrino_signal)) and values of 
     E_neutrino_signal are generated by a random number of a gaussian distribution with mean = MASS_DM 
     and sigma = epsilon -> is equal to generating from TheoSpectrum_signal
3.2. Loop from 1 to length(E_neutrino_signal) and convolve each entry in E_neutrino_signal with faltung
3.3. faltung is defined by simple low energy approximation E_visible = E_neutrino - 1.8 MeV and by
     energy resolution of the detector to be 3%/sqrt(E_visible)
"""

""" Theoretical spectrum of the electron-antineutrino signal, monoenergetic spectrum at mass_DM: """
# Calculate the electron-antineutrino flux from DM annihilation in the Milky Way at Earth from paper 0710.5420:
# DM annihilation cross-section necessary to explain the observed abundance of DM in the Universe, in cm**3/s (float):
sigma_anni = 3*10**(-26)
# canonical value of the angular-averaged intensity over the whole Milky Way (integer):
J_avg = 5
# solar radius circle in cm, 8.5 kiloparsec, 1kpc = 3.086*10**21 cm (float):
R_solar = 8.5*3.086*10**21
# normalizing DM density, in MeV/cm**3 (float):
rho_0 = 0.3*1000
# electron-antineutrino flux at Earth in 1/(MeV * s *cm**2) (float):
phi_signal = sigma_anni / 6 * J_avg * R_solar * rho_0**2 / MASS_DM**2

""" Calculate theoretical signal-spectrum (number of events per energy) with equations from paper 0710.5420: """
# total time-exposure in seconds, 10 years (float):
time = 10*3.156*10**7
# total exposure time in years:
t_years = time / (3.156*10**7)
# Number of free protons (target particles) for IBD in JUNO (float):
N_target = 1.45*10**33
# detection efficiency of IBD in JUNO (integer):
detection_eff = 0.73   # TODO: detection efficiency is set to 73 % from physics_report p. 40 table 2.1
# IBD cross-section in cm**2 for neutrinos with energy E = MASS_DM, equation (25) from paper 0302005_IBDcrosssection:
# simple approximation which agrees with full result of paper within few per-mille for E_neutrino <= 300 MeV:
# positron energy defined as E_neutrino - delta in MeV (float):
energy_positron = MASS_DM - delta
# positron momentum defined as sqrt(energy_positron**2-mass_positron**2) in MeV (float64):
momentum_positron = np.sqrt(energy_positron**2 - MASS_POSITRON**2)
# IBD cross-section for neutrino energy = MASS_DM (float64):
sigma_IBD = 10**(-43) * momentum_positron * energy_positron * \
            MASS_DM**(-0.07056 + 0.02018*np.log(MASS_DM) - 0.001953*np.log(MASS_DM)**3)
# Number of electron-antineutrino events in JUNO after "time" years in 1/MeV (float64):
# TODO: only IBD on free protons is considered
N_neutrino_signal = sigma_IBD * phi_signal * N_target * time * detection_eff

# delta function from theoretical spectrum is approximated as very thin gaussian distribution:
# epsilon defines sigma in the gaussian distribution (float):
epsilon = 10**(-5)
delta_function = 1 / (np.sqrt(2 * np.pi * epsilon**2)) * np.exp(-0.5 * (E1 - MASS_DM) ** 2 / epsilon**2)
# normalize delta_function (integral(delta_function) = 1) (array of float64):
delta_function = delta_function / sum(delta_function)
# Theoretical spectrum of the electron-antineutrino signal in 1/MeV (number of events as function of the
# electron-antineutrino energy) (array of float64):
TheoSpectrum_signal = N_neutrino_signal * delta_function

# Generate (NUMBER*N_neutrino_signal)-many neutrinos of energy E_neutrino_signal from the theoretical spectrum
# TheoSpectrum_signal, TheoSpectrum_signal is described as gaussian distribution with mean=MASS_DM and sigma=epsilon
# (array of float64):
# TODO: number of neutrinos is rounded to an integer number
E_neutrino_signal = np.random.normal(MASS_DM, epsilon, round(float(NUMBER * N_neutrino_signal)))

""" Visible spectrum of the electron-antineutrino signal, theoretical spectrum is convolved 
    with Faltung (gaussian distribution): """
# Convolve the generated energies E_neutrino_signal with the faltung defined below for every entry in E_neutrino_signal:
# Preallocate E_visible_signal (empty array):
E_visible_signal = np.array([])
# loop over entries in E_neutrino_signal, for each energy generate NUMBER_VISIBLE random numbers:
for index1 in np.arange(len(E_neutrino_signal)):
    # simple estimation of dependency between E_visible and E_neutrino_signal from Inverse Beta Decay,
    # E_visible = E_neutrino_signal - delta + MASS_POSITRON = E_neutrino_signal - 0.782 MeV,
    # only for low energies over threshold (float64):
    # TODO: use a more general correlation between E_vis and E_neutrino!
    corr_TheoVis = E_neutrino_signal[index1] - delta + MASS_POSITRON
    # energy resolution, assumption 3% * sqrt(E_visible in MeV) (float64):
    # TODO: use a more general term for the energy resolution of the JUNO detector
    energy_resolution = 0.03 * np.sqrt(corr_TheoVis)
    # generate (NUMBER_VISIBLE)-many random numbers around E_neutrino_signal[index1] from convolution, which is
    # described by a gaussian distribution with mean = corr_TheoVis and sigma = energy_resolution (array of float64):
    E_visible_signal = np.append(E_visible_signal, np.random.normal(corr_TheoVis, energy_resolution, NUMBER_VISIBLE))


"""
List of assumptions or estimations made for the calculation of the background spectrum from DSNB electron-antineutrinos:

1. DSNB calculation based on the calculation in the PhD-thesis of Michi Wurm (Cosmic background discrimination for the
   rare neutrino event search in Borexino and Lena) and on the calculation in paper 0410061 of Ando:
2. only electron-antineutrinos interacting with free protons are considered (Inverse Beta Decay), dominant up to ~80 MeV
3. In the calculation of the number flux the factor used here is "3.9*10^(-3)" like in 0410061 and 
   not "3.9*10^(-4)" like in Michi Wurm's dissertation
4. 70 percent of the electron-antineutrinos survive and 30 percent of the muon-/tau-antineutrinos appear 
   as electron-antineutrinos at the Earth
"""

""" Theoretical spectrum of DSNB electron-antineutrino background: """
# Calculate the electron-antineutrino flux from DSNB from Michi Wurm's dissertation and paper 0410061 of Ando:
# differential number flux is divided into electron-antineutrinos and non-electron antineutrinos,
# then the oscillation of the flavors are considered and a final electron-antineutrino flux at Earth is calculated

# redshift, where the gravitational collapse begun (integer):
z_max = 5
# redshift is variable to calculate the flux (array of float64):
z = np.arange(0, z_max, 0.01)
# Hubble constant in 1/s (70 km/(s*Mpc) = 70 * 1000 m / (s*10**6*3.086*10**16 m ) (float):
H_0 = 70*1000/(10**6*3.086*10**16)
# factor, that describes the overall normalization of R_SN (redshift-dependent Supernova rate) (float):
f_SN = 2.5
# parameter: is 1 for a value of H_0 = 70 km/(s*Mpc) (integer):
h_70 = 1

""" differential number flux for electron-antineutrinos: """
# mean energy of electron-antineutrinos in MeV (float):
E_mean_NuEBar = 15.4
# pinching parameters (float):
beta_NuEBar = 3.8
# neutrino luminosity in MeV (1 erg = 624151 MeV) (value taken from 0710.5420) (float):
L_NuEBar = 4.9*10**52*624151
# part of electron-antineutrino spectrum independent of redshift z in 1/MeV (without oscillation) (array of float64):
TheoSpectrum_NuEBar_constant = (1+beta_NuEBar)**(1+beta_NuEBar) * L_NuEBar / (gamma(1+beta_NuEBar) * E_mean_NuEBar**2)\
                               * (E1 / E_mean_NuEBar)**beta_NuEBar
# Calculation of the integral from 0 to z_max:
# The integral have to be calculated for every entry in E1, because the integrand depends on E1 and z:
# preallocate the result-vector of the integral (empty array):
TheoSpectrum_NuEBar_integral = np.array([])
# Therefore loop over entries in E1 and calculate the numerical integral:
for index2 in np.arange(len(E1)):
    # calculate the integrand for one entry in E1 (array of float64):
    integrand_NuEBar = (1+z)**beta_NuEBar * np.exp(-(1+beta_NuEBar) * (1+z) * E1[index2]/E_mean_NuEBar)\
                       * np.exp(3.4*z) / (np.exp(3.8*z)+45) / ((1+z)**(3/2))
    # integrate integrand_NuEBar over z with the trapezoidal method and append the value to
    # TheoSpectrum_NuEBar_integral (array of float64):
    integral_NuEBar = np.trapz(integrand_NuEBar, z)
    TheoSpectrum_NuEBar_integral = np.append(TheoSpectrum_NuEBar_integral, np.array([integral_NuEBar]))

# Convert TheoSpectrum_NuEBar_integral from 1/(year * Mpc**3) into  1/(second * cm**3) (array of float64):
TheoSpectrum_NuEBar_integral = TheoSpectrum_NuEBar_integral / (3.154 * 10**7) / (10**6 * 3.086 * 10**18)**3
# differential number flux of electron-antineutrinos in 1/(MeV*s*cm**2) (array of float64):
# TODO: which is the correct value? 3.9*10**(-3) or 3.9*10**(-4)
NumberFlux_NuEBar = C_LIGHT/H_0 * 3.9 * 10**(-3) * f_SN * h_70 \
                    * TheoSpectrum_NuEBar_constant * TheoSpectrum_NuEBar_integral
# differential number flux of electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
# about 70 percent of electron-antineutrinos survive (array of float64):
NumberFlux_NuEBar = 0.7 * NumberFlux_NuEBar

""" differential number flux for NON-electron-antineutrinos (muon- and tau-antineutrinos): """
# mean energy of NON-electron-antineutrinos in MeV (float):
E_mean_NuBar = 21.6
# pinching parameters (float):
beta_NuBar = 1.8
# neutrino luminosity in MeV (1 erg = 624151 MeV) (value taken from 0710.5420) (float):
L_NuBar = 5.0*10**52*624151
# part of NON-electron-antineutrino spectrum independent of redshift z in 1/MeV (without oscillation)
# (array of float64):
TheoSpectrum_NuBar_constant = (1+beta_NuBar)**(1+beta_NuBar) * L_NuBar / (gamma(1+beta_NuBar) * E_mean_NuBar**2)\
                              * (E1 / E_mean_NuBar)**beta_NuBar
# Calculation of the integral from 0 to z_max:
# The integral have to be calculated for every entry in E1, because the integrand depends on E1 and z:
# Preallocate the result-vector of the integral (empty array):
TheoSpectrum_NuBar_integral = np.array([])
# Therefore loop over entries in E1 and calculate the numerical integral:
for index3 in np.arange(len(E1)):
    # calculate the integrand for one entry in E1 (array of float64):
    integrand_NuBar = (1 + z) ** beta_NuBar * np.exp(-(1 + beta_NuBar) * (1 + z) * E1[index3] / E_mean_NuBar) \
                       * np.exp(3.4 * z) / (np.exp(3.8 * z) + 45) / ((1 + z) ** (3 / 2))
    # integrate integrand_NuEBar over z with the trapezoidal method and append the value to
    # TheoSpectrum_NuEBar_integral (array of float64):
    integral_NuBar = np.trapz(integrand_NuBar, z)
    TheoSpectrum_NuBar_integral = np.append(TheoSpectrum_NuBar_integral, np.array([integral_NuBar]))

# Convert TheoSpectrum_NuBar_integral from 1/(year * Mpc**3) into 1/(second * cm**3) (array of float64):
TheoSpectrum_NuBar_integral = TheoSpectrum_NuBar_integral/(3.154*10**7)/(10**6*3.086*10**18)**3
# differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) (array of float64):
# TODO: which is the correct value? 3.9*10**(-3) or 3.9*10**(-4)
NumberFlux_NuBar = C_LIGHT/H_0 * 3.9*10**(-3) * f_SN * h_70 * TheoSpectrum_NuBar_constant * TheoSpectrum_NuBar_integral
# differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
# about 30 percent of the emitted muon- and tau-antineutrinos will appear as electron-antineutrinos
# at the Earth (array of float64):
NumberFlux_NuBar = 0.3 * NumberFlux_NuBar

# Total number flux of electron anti-neutrinos from DSNB at Earth/JUNO in 1/(MeV * s * cm**2) (array of float64):
NumberFlux_DSNB = NumberFlux_NuEBar + NumberFlux_NuBar

# Calculation of the theoretical spectrum (number of events per energy) of electron-antineutrinos from DSNB
# (equ. 9.1 in Michi Wurm's dissertation is equal to equ. 4 in 0710.5420, except to the detection efficiency
# but detection_efficiency = 1 and has therefore no effect):
# IBD cross-section in cm**2 for neutrinos with energy E = E1, equation (25) from paper 0302005_IBDcrosssection:
# simple approximation which agrees with full result of paper within few per-mille for E_neutrino <= 300 MeV:
# positron energy defined as E_neutrino - Delta in MeV (array of float64):
energy_positron1 = E1 - delta
# positron momentum defined as sqrt(energy_positron1**2-mass_positron**2) in MeV (array of float64):
momentum_positron1 = np.sqrt(energy_positron1**2 - MASS_POSITRON**2)
# IBD cross-section for neutrino energy = MASS_DM in cm**2 (array of float64):
sigma_IBD1 = 10**(-43) * momentum_positron1 * energy_positron1 \
             * E1**(-0.07056 + 0.02018*np.log(E1) - 0.001953*np.log(E1)**3)
# Theoretical spectrum of DSNB neutrino events in JUNO after "time" years in 1/MeV (array of float64):
# TODO: detection efficiency is set to 0.73 AND only IBD on free protons is considered
TheoSpectrum_DSNB = sigma_IBD1 * NumberFlux_DSNB * N_target * time * detection_eff


""" Visible spectrum of the DSNB electron-antineutrino background, theoretical spectrum is convolved 
    with Faltung (gaussian distribution): """
# first integrate the theoretical spectrum over E1 to get the number of DSNB neutrinos (float64):
N_neutrino_DSNB = np.trapz(TheoSpectrum_DSNB, E1)
# Then normalize the theoretical spectrum TheoSpectrum_DSNB, so that sum(TheoSpectrum_DSNB) = 1.
# Then pdf_TheoSpectrum_DSNB can be used as probability in np.random.choice (array of float64):
pdf_TheoSpectrum_DSNB = TheoSpectrum_DSNB / sum(TheoSpectrum_DSNB)

# Then generate (NUMBER * N_neutrino_DSNB)-many neutrinos with energies E_neutrino_DSNB from the theoretical spectrum
# (array of float64):
# TODO: number of neutrinos is rounded to an integer number
E_neutrino_DSNB = np.random.choice(E1, round(float(NUMBER * N_neutrino_DSNB)), p=pdf_TheoSpectrum_DSNB)

# Convolve the generated energies E_neutrino_DSNB with the faltung defined below for every entry in E_neutrino_DSNB:
# Preallocate E_visible_DSNB (empty array):
E_visible_DSNB = np.array([])
# loop over entries in E_neutrino_DSNB, for each energy generate NUMBER_VISIBLE random numbers:
for index4 in np.arange(len(E_neutrino_DSNB)):
    # simple estimation of dependency between E_visible and E_neutrino_DSNB from Inverse Beta Decay,
    # E_visible = E_neutrino_DSNB - delta + MASS_POSITRON = E_neutrino_DSNB - 0.782 MeV,
    # only for low energies over threshold (float64):
    # TODO: use a more general correlation between E_vis and E_neutrino!
    corr_TheoVis = E_neutrino_DSNB[index4] - delta + MASS_POSITRON
    # energy resolution, assumption 3% * sqrt(E_visible in MeV) (float64):
    # TODO: use a more general term for the energy resolution of the JUNO detector
    energy_resolution = 0.03 * np.sqrt(corr_TheoVis)
    # generate (NUMBER_VISIBLE)-many random numbers around E_neutrino_DSNB[index4] from convolution, which is
    # described by a gaussian distribution with mean = corr_TheoVis and sigma = energy_resolution (array of float64):
    E_visible_DSNB = np.append(E_visible_DSNB, np.random.normal(corr_TheoVis, energy_resolution, NUMBER_VISIBLE))


"""
List of assumptions or estimations made for the calculation of the atmospheric Charged Current Background:

1. the differential flux is taken for no oscillation, for only electron-antineutrinos, for solar average and at the 
   site of Super-Kamiokande
2. simulated data taken from table 2 and 3 in paper: 1-s2.0-S0927650505000526-main from Battistoni et al. in 2005
3. this data is linear interpolated to get the differential flux corresponding to the binning in E1, it is estimated 
   that for E1_atmo = 0 MeV, flux_atmo is also 0.
"""

""" Theoretical spectrum of atmospheric charged-current background: """
# Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main:
E1_atmo = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133, 150,
                    168, 188])

# differential flux in energy for no oscillation for electron-antineutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)):
flux_atmo_NuEbar = 10**(-4) * np.array([0., 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                        83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9, 21.3, 18.3])
# linear interpolation of the simulated data above to get the differential neutrino flux corresponding to E1,
# differential flux of electron-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)):
Flux_atmo_NuEbar = np.interp(E1, E1_atmo, flux_atmo_NuEbar)

# differential flux in energy for no oscillation for muon-antineutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)):
flux_atmo_NuMubar = 10**(-4) * np.array([0., 116., 128., 136., 150., 158., 162., 170., 196., 177., 182., 183., 181.,
                                         155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0, 55.6, 47.6, 40.8])
# linear interpolation of the simulated data above to get the differential neutrino flux corresponding to E1,
# differential flux of muon-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)):
Flux_atmo_NuMubar = np.interp(E1, E1_atmo, flux_atmo_NuMubar)

# differential flux in energy for no oscillation for electron-neutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)):
flux_atmo_NuE = 10**(-4) * np.array([0., 69.9, 74.6, 79.7, 87.4, 94.2, 101., 103., 109., 108., 107., 101., 88.5,
                                     69.6, 64.4, 59.3, 54.3, 49.7, 45.1, 40.6, 35.8, 31.7, 27.3, 23.9, 20.4])
# linear interpolation of the simulated data above to get the differential neutrino flux corresponding to E1,
# differential flux of electron-neutrinos in (MeV**(-1) * cm**(-2) * s**(-1)):
Flux_atmo_NuE = np.interp(E1, E1_atmo, flux_atmo_NuE)

# differential flux in energy for no oscillation for muon-neutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)):
flux_atmo_NuMu = 10**(-4) * np.array([0., 114., 124., 138., 146., 155., 159., 164., 181., 174., 179., 178., 176., 153.,
                                      131., 123., 114., 107., 96.3, 84.2, 72.7, 63.5, 55.2, 47.7, 41.2])
# linear interpolation of the simulated data above to get the differential neutrino flux corresponding to E1,
# differential flux of electron-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)):
Flux_atmo_NuMu = np.interp(E1, E1_atmo, flux_atmo_NuMu)


# flux of electron-antineutrinos at the detector WITHOUT oscillation:
Flux_atmospheric_NuEbar = Flux_atmo_NuEbar

# Theoretical spectrum (in events per MeV) of electron-antineutrinos ("number of positron-events") from
# inverse beta decay on free protons, oscillation is considered (from paper 0903.5323.pdf, equ. 64):
# TODO: consider the flux with oscillation
# TODO: only IBD of electron-antineutrinos on free protons is considered,
# TODO: -> NO CC-interaction on bound nuclei is considered, because these interactions do not mimic IBD
# TODO: for solar average
# TODO: at site of Super-Kamiokande
# TODO: detection efficiency of IBD is set to 0.73
# TODO: "dispersion" is not considered
TheoSpectrum_atmo_IBD = Flux_atmospheric_NuEbar * sigma_IBD1 * detection_eff * N_target * time

""" Visible spectrum of the atmospheric charged-current background, theoretical spectrum is convolved 
    with Faltung (gaussian distribution): """
""" First calculate the visible spectrum from IBD on free protons of electron-antineutrinos"""
# first integrate the theoretical spectrum over E1 to get the number of atmospheric electron-antineutrinos (float64):
N_neutrino_atmo_IBD = np.trapz(TheoSpectrum_atmo_IBD, E1)
# Then normalize the theoretical spectrum TheoSpectrum_atmo_IBD, so that sum(TheoSpectrum_atmo_IBD) = 1.
# Then pdf_TheoSpectrum_atmo_IBD can be used as probability in np.random.choice (array of float64):
pdf_TheoSpectrum_atmo_IBD = TheoSpectrum_atmo_IBD / sum(TheoSpectrum_atmo_IBD)
# Then generate (NUMBER * N_neutrino_atmo_IBD)-many neutrinos with energies E_neutrino_atmo_IBD from the
# theoretical spectrum (array of float64):
# TODO: number of neutrinos is rounded to an integer number
E_neutrino_atmo_IBD = np.random.choice(E1, round(float(NUMBER * N_neutrino_atmo_IBD)), p=pdf_TheoSpectrum_atmo_IBD)
# Convolve the generated energies E_neutrino_atmo_IBD with the faltung defined below for every entry in
# E_neutrino_atmo_IBD:
# Preallocate E_visible_atmo_IBD (empty array):
E_visible_atmo_IBD = np.array([])
# loop over entries in E_neutrino_atmo_IBD, for each energy generate NUMBER_VISIBLE random numbers:
for index5 in np.arange(len(E_neutrino_atmo_IBD)):
    # simple estimation of dependency between E_visible and E_neutrino_atmo_IBD from Inverse Beta Decay,
    # E_visible = E_neutrino_atmo_IBD - delta + MASS_POSITRON = E_neutrino_IBD - 0.782 MeV,
    # only for low energies over threshold (float64):
    # TODO: use a more general correlation between E_vis and E_neutrino!
    corr_TheoVis = E_neutrino_atmo_IBD[index5] - delta + MASS_POSITRON
    # energy resolution, assumption 3% * sqrt(E_visible in MeV) (float64):
    # TODO: use a more general term for the energy resolution of the JUNO detector
    energy_resolution = 0.03 * np.sqrt(corr_TheoVis)
    # generate (NUMBER_VISIBLE)-many random numbers around E_neutrino_atmo_IBD[index5] from convolution, which is
    # described by a gaussian distribution with mean = corr_TheoVis and sigma = energy_resolution (array of float64):
    E_visible_atmo_IBD = np.append(E_visible_atmo_IBD,
                                   np.random.normal(corr_TheoVis, energy_resolution, NUMBER_VISIBLE))

""" TOTAL visible energy from atmospheric charged-current background in MeV: """
E_visible_atmospheric = E_visible_atmo_IBD


"""
List of assumptions or estimations made for the calculation of the reactor-antineutrino Background:

1. the reactor electron-antineutrino flux is calculated according to the offline Inverse-Beta generator
   KRLReactorFlux.cc and KRLReactorFlux.hh, which is based on the paper of Vogel and Engel 1989 (PhysRevD.39.3378)
2. neutrino-oscillation is taken into account. same calculation like in NuOscillation.cc and NuOscillation.hh. 
   Normal Hierarchy is considered
"""

""" Theoretical spectrum of reactor electron-antineutrinos: """

""" Flux of reactor electron-antineutrinos calculated with the code described 
in KRLReactorFlux.cc and KRLReactorFlux.hh"""
# Total thermal power of the Yangjiang and Taishan nuclear power plants (from PhysicsReport), in GW:
power_th = 35.73
# Coefficients from KRLReactorFlux.cc (data taken from Vogel and Engel 1989 (PhysRevD.39.3378), table 1):
Coeff235U = np.array([0.870, -0.160, -0.0910])
Coeff239Pu = np.array([0.896, -0.239, -0.0981])
Coeff238U = np.array([0.976, -0.162, -0.0790])
Coeff241Pu = np.array([0.793, -0.080, -0.1085])
# Fractions (data taken from KRLReactorFLux.cc):
Fraction235U = 0.6
Fraction239Pu = 0.3
Fraction238U = 0.05
Fraction241Pu = 0.05

# fit to the electron-antineutrino spectrum, in units of electron-antineutrinos/(MeV * fission)
# (from KRLReactorFlux.cc and Vogel/Engel 1989 (PhysRevD.39.3378), equation 4):
U235 = np.exp(Coeff235U[0] + Coeff235U[1]*E1 + Coeff235U[2]*E1**2)
Pu239 = np.exp(Coeff239Pu[0] + Coeff239Pu[1]*E1 + Coeff239Pu[2]*E1**2)
U238 = np.exp(Coeff238U[0] + Coeff238U[1]*E1 + Coeff238U[2]*E1**2)
Pu241 = np.exp(Coeff241Pu[0] + Coeff241Pu[1]*E1 + Coeff241Pu[2]*E1**2)

# add the weighted sum of the terms, electron-antineutrino spectrum in units of electron-antineutrinos/(MeV * fission)
# (data taken from KRLReactorFLux.cc):
spec1_reactor = U235*Fraction235U + Pu239*Fraction239Pu + U238*Fraction238U + Pu241*Fraction241Pu
# There are 3.125*10**19 fissions/GW/second, spectrum in units of electron-antineutrino/(MeV * GW * s):
spec2_reactor = spec1_reactor * 3.125*10**19
# There are about 3.156*10**7 seconds in a year, spectrum in units of electron-antineutrino/(MeV * GW * year):
spec3_reactor = spec2_reactor * 3.156*10**7
# electron-antineutrino flux in units of electron-antineutrino/(MeV * year):
flux_reactor = spec3_reactor * power_th


""" Consider Neutrino oscillation for NORMAL HIERARCHY from NuOscillation.cc: """
# Oscillation parameters:
# distance reactor to detector in meter:
L_m = 5.25*10**4
# distance reactor to detector in centimeter:
L_cm = L_m * 100
# mixing angles from PDG 2016 (same in NuOscillation.cc):
sin2_th12 = 0.297
sin2_th13 = 0.0214
# mass squared differences in eV**2 from PDG 2016 (same in NuOscillation.cc):
Dm2_21 = 7.37*10**(-5)
Dm2_31 = 2.50*10**(-3) + Dm2_21/2
Dm2_32 = 2.50*10**(-3) - Dm2_21/2
# calculate the other parameters:
cos2_th12 = 1. - sin2_th12
sin2_2th12 = 4.*sin2_th12*cos2_th12
cos2_th13 = 1. - sin2_th13
sin2_2th13 = 4.*sin2_th13*cos2_th13
cos4_th13 = cos2_th13**2
# With these parameters calculate survival probability of electron-antineutrinos:
P21 = sin2_2th12 * cos4_th13 * np.sin(1.267 * Dm2_21 * L_m / E1)**2
P31 = sin2_2th13 * cos2_th12 * np.sin(1.267 * Dm2_31 * L_m / E1)**2
P32 = sin2_2th13 * sin2_th12 * np.sin(1.267 * Dm2_32 * L_m / E1)**2
# Survival probability of electron-antineutrinos for Normal Hierarchy:
Prob_oscillation_NH = 1. - P21 - P31 - P32

""" Theoretical reactor electron-antineutrino spectrum in JUNO with oscillation """
# Theoretical spectrum in JUNO for normal hierarchy with oscillation in time years:
TheoSpectrum_reactor = 1 / (4*np.pi*L_cm**2) * flux_reactor * sigma_IBD1 * detection_eff * N_target * t_years\
                       * Prob_oscillation_NH


""" Visible spectrum of the reactor electron-antineutrino background, theoretical spectrum is convolved 
    with Faltung (gaussian distribution): """
# first integrate the theoretical spectrum over E1 to get the number of reactor neutrinos (float64):
N_neutrino_reactor = np.trapz(TheoSpectrum_reactor, E1)
print("Number of reactor electron-antineutrinos after 10 years = %d" % N_neutrino_reactor)
# Then normalize the theoretical spectrum TheoSpectrum_reactor, so that sum(TheoSpectrum_reactor) = 1.
# Then pdf_TheoSpectrum_reactor can be used as probability in np.random.choice (array of float64):
pdf_TheoSpectrum_reactor = TheoSpectrum_reactor / sum(TheoSpectrum_reactor)
# Then generate (NUMBER * N_neutrino_reactor)-many neutrinos with energies E_neutrino_reactor from the
# theoretical spectrum (array of float64):
# TODO: number of neutrinos is rounded to an integer number
E_neutrino_reactor = np.random.choice(E1, round(float(NUMBER * N_neutrino_reactor)), p=pdf_TheoSpectrum_reactor)

# Convolve the generated energies E_neutrino_reactor with the faltung defined below
# for every entry in E_neutrino_reactor:
# Preallocate E_visible_reactor (empty array):
E_visible_reactor = np.array([])
# loop over entries in E_neutrino_reactor, for each energy generate NUMBER_VISIBLE random numbers:
for index5 in np.arange(len(E_neutrino_reactor)):
    # simple estimation of dependency between E_visible and E_neutrino_reactor from Inverse Beta Decay,
    # E_visible = E_neutrino_reactor - delta + MASS_POSITRON = E_neutrino_reactor - 0.782 MeV,
    # only for low energies over threshold (float64):
    # TODO: use a more general correlation between E_vis and E_neutrino!
    corr_TheoVis = E_neutrino_reactor[index5] - delta + MASS_POSITRON
    # energy resolution, assumption 3% * sqrt(E_visible in MeV) (float64):
    # TODO: use a more general term for the energy resolution of the JUNO detector
    energy_resolution = 0.03 * np.sqrt(corr_TheoVis)
    # generate (NUMBER_VISIBLE)-many random numbers around E_neutrino_reactor[index5] from convolution, which is
    # described by a gaussian distribution with mean = corr_TheoVis and sigma = energy_resolution (array of float64):
    E_visible_reactor = np.append(E_visible_reactor, np.random.normal(corr_TheoVis, energy_resolution, NUMBER_VISIBLE))


# Total theoretical spectrum in 1/(MeV*t_year):
TheoSpectrum_total = TheoSpectrum_signal + TheoSpectrum_DSNB + TheoSpectrum_atmo_IBD + TheoSpectrum_reactor
# Total visible energy in MeV:
E_visible_total = np.append(E_visible_reactor, np.append(E_visible_atmospheric,
                                                         np.append(E_visible_signal, E_visible_DSNB)))


# Display the theoretical spectra with the settings below:
h1 = pyplot.figure(1)
pyplot.plot(E1, TheoSpectrum_total, label='total spectrum')
pyplot.plot(E1, TheoSpectrum_signal, '--', label='signal from DM annihilation')
pyplot.plot(E1, TheoSpectrum_DSNB, '--', label='DSNB background')
pyplot.plot(E1, TheoSpectrum_atmo_IBD, '--', label='atmospheric IBD events without oscillation')
pyplot.plot(E1, TheoSpectrum_reactor, '--', label='reactor electron-antineutrino background')
pyplot.xlim(E1[0], E1[-1])
pyplot.ylim(ymin=0, ymax=25)
# pyplot.xticks(np.arange(4.0, E1[-1]), 2.0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Theoretical spectrum dN/dE in 1/MeV")
pyplot.title("Theoretical electron-antineutrino spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
             .format(t_years, MASS_DM))
pyplot.legend()


# Display the expected spectra with the settings below:
h2 = pyplot.figure(2)
n_total, bins3, patches3 = pyplot.hist(E_visible_total, bins=E2, histtype='step', label='total spectrum')
n_signal, bins1, patches1 = pyplot.hist(E_visible_signal, bins=E2, histtype='step', label='signal from DM annihilation')
n_DSNB, bins2, patches2 = pyplot.hist(E_visible_DSNB, bins=E2, histtype='step', label='DSNB background')
n_atmospheric, bins4, patches4 = pyplot.hist(E_visible_atmospheric, bins=E2, histtype='step',
                                             label='atmospheric IBD events without oscillation')
n_reactor, bins5, patches5 = pyplot.hist(E_visible_reactor, bins=E2, histtype='step',
                                         label='reactor electron-antineutrino background')
pyplot.xlim(E2[0], E2[-1])
pyplot.ylim(ymin=0, ymax=25)
# pyplot.xticks(np.arange(2.0, E2[-1]), 2.0)
pyplot.xlabel("Visible energy in MeV")
pyplot.ylabel("Expected spectrum dN/dE in 1/MeV*{0:d})".format(NUMBER*NUMBER_VISIBLE))
pyplot.title("Expected spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
             .format(t_years, MASS_DM))
pyplot.legend()

pyplot.show()
