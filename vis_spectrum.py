""" Script to calculate the theoretical neutrino spectrum and to simulate the expected spectrum from DM annihilation
    with background in the JUNO detector in the energy range from few MeV to hundred MeV
    -> old version of the simulation
    -> do NOT use it
    -> instead use gen_simu_spectrum_v1.py
"""

# import of the necessary packages:
import numpy as np
from math import gamma
from matplotlib import pyplot

print('Simulation started...')

""" Variables: """
# Dark Matter mass in MeV (float):
mass_DM = 20.0
# number defines, how many "pseudo"-neutrinos are generated per real neutrino in the Script
# to get higher statistics (integer):
number = 1
# number_vis defines, how many random numbers E_visible_".." are generated from the gaussian distribution
# "faltung" around E_neutrino_".." (integer):
number_vis = 1

""" energy-arrays: """
# energy corresponding to the electron-antineutrino energy in MeV (np.array of float64):
interval = 0.01
E1 = np.arange(10, 105, interval)
# energy corresponding to the visible energy in MeV, E2 defines the bins in pyplot.hist() (np.array of float64):
E2 = np.arange(7, 130, 0.1)

""" Natural constants: """
# velocity of light in vacuum, in cm/s (float constant):
C_LIGHT = 2.998 * 10 ** 10
# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON

""" Constants depending on JUNO: """
# total time-exposure in seconds, 10 years (float):
time = 10 * 3.156 * 10 ** 7
# total exposure time in years (float):
t_years = time / (3.156 * 10 ** 7)
# Number of free protons (target particles) for IBD in JUNO (float):
N_target = 1.45 * 10 ** 33
# detection efficiency of IBD in JUNO, from physics_report.pdf, page 40, table 2.1
# (combined efficiency of energy cut, time cut, vertex cut, Muon veto, fiducial volume) (float):
detection_eff = 0.73  # TODO: detection efficiency is set to 73  from physics_report p. 40 table 2.1


""" Define several functions used in the script: """


def sigma_ibd(energy, delta, mass_positron):
    """ IBD cross-section in cm**2 for neutrinos with E=energy, equation (25) from paper 0302005_IBDcrosssection:
        simple approximation which agrees with full result of paper within few per-mille
        for neutrino energies <= 300 MeV:
        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float OR float)
        delta: difference mass_neutron minus mass_proton in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        """
    # positron energy defined as energy - delta in MeV (np.array of float64 or float):
    energy_positron = energy - delta
    # positron momentum defined as sqrt(energy_positron**2-mass_positron**2) in MeV (np.array of float64 or float):
    momentum_positron = np.sqrt(energy_positron ** 2 - mass_positron ** 2)
    # IBD cross-section in cm**2 (array of float64 or float):
    sigma = (10 ** (-43) * momentum_positron * energy_positron *
             energy ** (-0.07056 + 0.02018 * np.log(energy) - 0.001953 * np.log(energy) ** 3))
    return sigma


def energy_neutrino(energy, theospectrum, num):
    """ neutrino energies (np.array of floats) in MeV generated randomly from the calculated theoretical
        spectrum theospectrum:
        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        theospectrum: theoretical neutrino spectrum in 1/MeV (np.array of float)
        num: defines, how many "pseudo"-neutrinos are generated per real neutrino to get higher statistics (integer)
        """
    # first integrate the theoretical spectrum over energy to get the number neutrinos (float64):
    n_neutrino = np.trapz(theospectrum, energy)
    # Then normalize the theoretical spectrum theospectrum, so that sum(theospectrum) = 1.
    # Then pdf_theospectrum can be used as probability in np.random.choice (np.array of float64):
    pdf_theospectrum = theospectrum / sum(theospectrum)
    # Then generate (num * n_neutrino)-many neutrinos with energies e_neutrino from the theoretical spectrum
    # (np.array of float64):
    # TODO: (number * number of neutrinos) is rounded to an integer number
    e_neutrino = np.random.choice(energy, round(float(num * n_neutrino)), p=pdf_theospectrum)
    return e_neutrino


def correlation_vis_neutrino(e_neutrino_entry, delta, mass_positron):
    """ simple estimation of correlation between E_visible and e_neutrino from Inverse Beta Decay:
        e_neutrino_entry: ONE entry of the e_neutrino np.array in MeV, the e_neutrino np.array contains the
                          neutrino energies generated from the theoretical spectrum (float)
        delta: difference mass_neutron minus mass_proton in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        """
    # TODO: use a more general correlation between E_vis and E_neutrino!
    # Correlation between visible and neutrino energy in MeV:
    # E_visible = E_neutrino - delta + mass_positron = E_neutrino - 0.782 MeV,
    # only for low energies over threshold (float64):
    corr_vis_neutrino = e_neutrino_entry - delta + mass_positron
    return corr_vis_neutrino


def energy_resolution(corr_vis_neutrino):
    """ 'energy resolution' of the JUNO detector, in detail: energy_resolution returns the width sigma of the
        gaussian distribution. The real energy of the neutrino is smeared by a gaussian distribution characterized
        by sigma:
        corr_vis_neutrino: correlation between visible and neutrino energy,
                           characterizes the visible energy in MeV (float)
        """
    # parameters to describe the energy resolution in percent (maximum values of table 13-4, page 196, PhysicsReport):
    p0 = 2.8
    p1 = 0.26
    p2 = 0.9
    # energy resolution defined as sigma/E_visible in percent, 3-parameter function (page 195, PhysicsReport) (float):
    energy_res = np.sqrt((p0 / np.sqrt(corr_vis_neutrino))**2 + p1**2 + (p2 / corr_vis_neutrino)**2)
    # sigma/width of the gaussian distribution in percent (float):
    sigma_resolution = energy_res * corr_vis_neutrino
    # sigma converted from percent to 'no unit' (float):
    sigma_resolution = sigma_resolution / 100
    return sigma_resolution


def visible_spectrum(e_neutrino, delta, mass_positron, num):
    """ generate an array of the visible energies in the detector from the real neutrino energies, which are generated
        from the theoretical spectrum (theoretical spectrum is convolved with gaussian distribution):
        e_neutrino: e_neutrino np. array contains the neutrino energies generated from the theoretical spectrum
                    (np.array of float)
        delta: difference mass_neutron minus mass_proton in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        num: defines, how many random numbers E_visible are generated from the gaussian
             distribution around E_neutrino (integer)
        """
    # Convolve the generated energies e_neutrino with the faltung defined below for every entry in e_neutrino:
    # Preallocate e_visible (empty array):
    e_visible = np.array([])
    # loop over entries in e_neutrino, for each energy generate num random numbers:
    for index in np.arange(len(e_neutrino)):
        # correlation between E_visible and E_neutrino (float):
        corr_vis_neutrino = correlation_vis_neutrino(e_neutrino[index], delta, mass_positron)
        # energy resolution of the detector (float):
        sigma_resolution = energy_resolution(corr_vis_neutrino)
        # generate (num)-many random numbers around e_neutrino[index] from convolution, which is described
        # by a gaussian distribution with mean = corr_vis_neutrino and sigma = sigma_resolution (np.array of float64):
        e_visible = np.append(e_visible, np.random.normal(corr_vis_neutrino, sigma_resolution, num))
    return e_visible


def numberflux_dsnb(energy, e_mean, beta, luminosity, redshift, c, hubble_0, f_sn, h_param):
    """ function calculates the number-flux of electron-antineutrinos (Nu_E_bar OR muon- and tau-antineutrinos
        (Nu_x_bar) from diffuse supernova neutrino background (DSNB). The calculation is based on the PhD thesis
        of Michael Wurm:
        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        e_mean: mean energy of DSNB antineutrinos in MeV (float)
        beta: pinching parameters (float)
        luminosity: neutrino luminosity in MeV (1 erg = 624151 MeV) (float)
        redshift: redshift z is variable to calculate the flux (np.array of float64)
        c: speed of light in vacuum (float)
        hubble_0: Hubble constant in 1/s (float)
        f_sn: factor, that describes the overall normalization of R_SN (redshift-dependent Supernova rate) (float)
        h_param: parameter h_70 is 1 for a value of H_0 = 70 km/(s*Mpc) (integer)
        """
    # part of the antineutrino spectrum independent of redshift z in 1/MeV (without oscillation) (np.array of float64):
    flux_constant = (1 + beta) ** (1 + beta) * luminosity / (gamma(1 + beta) * e_mean ** 2) * (energy / e_mean) ** beta
    # Calculation of the integral from 0 to z_max: The integral have to be calculated for every entry in energy,
    # because the integrand depends on energy and redshift:
    # preallocate the result-vector of the integral (empty array):
    flux_integral = np.array([])
    # Therefore loop over entries in energy and calculate the numerical integral:
    for index1 in np.arange(len(energy)):
        # calculate the integrand for one entry in energy (np.array of float64):
        integrand = (1 + redshift) ** beta * np.exp(-(1 + beta) * (1 + redshift) * energy[index1] / e_mean) * \
                    np.exp(3.4 * redshift) / (np.exp(3.8 * redshift) + 45) / ((1 + redshift) ** (3 / 2))
        # integrate integrand over redshift with the trapezoidal method and append the value to
        # flux_integral (np.array of float64):
        integral = np.trapz(integrand, redshift)
        flux_integral = np.append(flux_integral, np.array([integral]))
    # Convert flux_integral from 1/(year * Mpc**3) into  1/(second * cm**3) (np.array of float64):
    flux_integral = flux_integral / (3.156 * 10 ** 7 * (10 ** 6 * 3.086 * 10 ** 18) ** 3)
    # differential number flux of DSNB antineutrinos in 1/(MeV*s*cm**2) (np.array of float64):
    # TODO: which is the correct value? 3.9*10**(-3) or 3.9*10**(-4)
    numberflux = c / hubble_0 * 3.9 * 10 ** (-3) * f_sn * h_param * flux_constant * flux_integral

    return numberflux


""" Often used values of functions: """
# IBD cross-section for the DM signal in cm**2, must be calculated only for energy = mass_DM (float):
sigma_IBD_signal = sigma_ibd(mass_DM, DELTA, MASS_POSITRON)
# IBD cross-section for the backgrounds in cm**2, must be calculated for the whole energy range E1 (np.array of floats):
sigma_IBD = sigma_ibd(E1, DELTA, MASS_POSITRON)


# TODO: the number of neutrinos from the theoretical spectrum can be calculated separately, when needed:
# N_neutrino_DSNB = np.trapz(TheoSpectrum_DSNB, E1)

print("... simulation of DM annihilation signal...")

""" SIMULATE THE SIGNAL FROM NEUTRINOS FROM DM ANNIHILATION IN THE MILKY WAY: """

""" List of assumptions or estimations made for the calculation of the signal spectrum:
1. neutrino flux (phi_signal) is calculated as described in paper 0710.5420

2. theoretical signal spectrum (N_neutrino) is calculated as described in paper 0710.5420:
2.1. only IBD on free protons is considered 

3. Delta-function is approximated as very thin gaussian distribution 
3.1. Number of neutrinos with energy E_neutrino_signal is defined by round(float(N_neutrino_signal)) and values of 
     E_neutrino_signal are generated by a random number of a gaussian distribution with mean = MASS_DM 
     and sigma = epsilon -> is equal to generating from TheoSpectrum_signal
3.2. Loop from 1 to length(E_neutrino_signal) and convolve each entry in E_neutrino_signal with faltung
3.3. faltung is defined by correlation_vis_neutrino and energy_resolution
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
phi_signal = sigma_anni / 6 * J_avg * R_solar * rho_0**2 / mass_DM**2
# Number of electron-antineutrino events in JUNO after "time" years in 1/MeV (float64):
N_neutrino_signal = sigma_IBD_signal * phi_signal * N_target * time * detection_eff

# delta function from theoretical spectrum is approximated as very thin gaussian distribution
# (ONLY NEEDED TO DISPLAY THE THEORETICAL SPECTRUM):
# epsilon defines sigma in the gaussian distribution (float):
epsilon = 10**(-5)
delta_function = 1 / (np.sqrt(2 * np.pi * epsilon**2)) * np.exp(-0.5 * (E1 - mass_DM) ** 2 / epsilon**2)
# normalize delta_function (integral(delta_function) = 1) (array of float64):
delta_function = delta_function / sum(delta_function)
# Theoretical spectrum of the electron-antineutrino signal in 1/MeV (number of events as function of the
# electron-antineutrino energy) (array of float64):
TheoSpectrum_signal = N_neutrino_signal * delta_function

# Generate (number*N_neutrino_signal)-many neutrinos of energy E_neutrino_signal from the theoretical spectrum
# TheoSpectrum_signal (E_neutrino_signal is generated from gaussian distribution with mean=mass_DM and sigma=epsilon
# and not from TheoSpectrum_signal, because there are problems of np.random.choice() with this very thin distribution)
# array of neutrino energies generated from theoretical spectrum in MeV (np.array of float):
# TODO: (number * number of neutrinos) is rounded to an integer number
E_neutrino_signal = np.random.normal(mass_DM, epsilon, round(float(number * N_neutrino_signal)))

""" Visible spectrum of the electron-antineutrino signal, theoretical spectrum is convolved 
    with Faltung (gaussian distribution): """
# array of visible energies in MeV (np.array of float):
E_visible_signal = visible_spectrum(E_neutrino_signal, DELTA, MASS_POSITRON, number_vis)


print("... simulation of DSNB background...")

""" SIMULATE THE DSNB ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: """

"""
List of assumptions or estimations made for the calculation of the background spectrum from DSNB electron-antineutrinos:

1. DSNB calculation based on the calculation in the PhD-thesis of Michi Wurm (Cosmic background discrimination for the
   rare neutrino event search in Borexino and Lena) and on the calculation in paper 0410061 of Ando:
2. only electron-antineutrinos interacting with free protons are considered (Inverse Beta Decay), dominant up to ~80 MeV
3. In the calculation of the number flux the factor used here is "3.9*10^(-4)" like in Michi Wurm's dissertation and 
   not "3.9*10^(-3)" like in paper from Ando
4. 70 percent of the electron-antineutrinos survive and 30 percent of the muon-/tau-antineutrinos appear 
   as electron-antineutrinos at the Earth
"""

""" Theoretical spectrum of DSNB electron-antineutrino background: """
# Calculate the electron-antineutrino flux from DSNB from Michi Wurm's dissertation and paper 0410061 of Ando:
# differential number flux is divided into electron-antineutrinos and non-electron antineutrinos,
# then the oscillation of the flavors are considered and a final electron-antineutrino flux at Earth is calculated

# redshift, where the gravitational collapse begun (integer):
z_max = 5
# redshift is variable to calculate the flux (np.array of float64):
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
# differential number flux of electron-antineutrinos in 1/(MeV*s*cm**2) (np.array of float):
NumberFlux_NuEBar = numberflux_dsnb(E1, E_mean_NuEBar, beta_NuEBar, L_NuEBar, z, C_LIGHT, H_0, f_SN, h_70)
# differential number flux of electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
# about 70 percent of electron-antineutrinos survive (np.array of float64):
NumberFlux_NuEBar = 0.7 * NumberFlux_NuEBar

""" differential number flux for NON-electron-antineutrinos (muon- and tau-antineutrinos): """
# mean energy of NON-electron-antineutrinos in MeV (float):
E_mean_NuBar = 21.6
# pinching parameters (float):
beta_NuBar = 1.8
# neutrino luminosity in MeV (1 erg = 624151 MeV) (value taken from 0710.5420) (float):
L_NuBar = 5.0*10**52*624151
# differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) (np.array of float):
NumberFlux_NuBar = numberflux_dsnb(E1, E_mean_NuBar, beta_NuBar, L_NuBar, z, C_LIGHT, H_0, f_SN, h_70)
# differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
# about 30 percent of the emitted muon- and tau-antineutrinos will appear as electron-antineutrinos
# at the Earth (np.array of float64):
NumberFlux_NuBar = 0.3 * NumberFlux_NuBar

# Total number flux of electron anti-neutrinos from DSNB at Earth/JUNO in 1/(MeV * s * cm**2) (np.array of float64):
NumberFlux_DSNB = NumberFlux_NuEBar + NumberFlux_NuBar

# Theoretical spectrum of DSNB neutrino events in JUNO after "time" years in 1/MeV (np.array of float64):
TheoSpectrum_DSNB = sigma_IBD * NumberFlux_DSNB * N_target * time * detection_eff

""" Visible spectrum of the DSNB electron-antineutrino background, theoretical spectrum is convolved 
    with Faltung (gaussian distribution): """
# generate (number * number of DSNB-antineutrinos)-many antineutrinos with energies E_neutrino_DSNB in MeV
# from the theoretical spectrum (np.array of float64):
E_neutrino_DSNB = energy_neutrino(E1, TheoSpectrum_DSNB, number)
# # array of visible energies of DSNB neutrinos in MeV (np.array of float):
E_visible_DSNB = visible_spectrum(E_neutrino_DSNB, DELTA, MASS_POSITRON, number_vis)


print("... simulation of atmospheric CC background...")

""" SIMULATE THE ATMOSPHERIC CHARGED CURRENT ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: """

"""
List of assumptions or estimations made for the calculation of the atmospheric Charged Current Background:

1. the differential flux is taken for NO oscillation
1.2. for only electron-antineutrinos -> is ok, because only IBD on free protons is considered 
                                        (CC-interaction on bound nuclei do not mimic IBD)
1.3. for solar average -> is a good approximation
1.4. at the site of Super-K -> is ok, because JUNO is located at a lower geographical latitude (22.6°N) than 
                               Super-K (36.4°N) and therefore the flux in JUNO should be lower
                               -> Flux is overestimated a bit
5. simulated data taken from table 2 and 3 in paper: 1-s2.0-S0927650505000526-main from Battistoni et al. in 2005
6. this data is linear interpolated to get the differential flux corresponding to the binning in E1, it is estimated 
   that for E1_atmo = 0 MeV, flux_atmo is also 0.
"""

""" Theoretical spectrum of atmospheric charged-current background: """
# Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
E1_atmo = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133, 150,
                    168, 188])

# differential flux in energy for no oscillation for electron-antineutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_CCatmo_NuEbar = 10**(-4) * np.array([0., 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                          83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9, 21.3, 18.3])
# linear interpolation of the simulated data above to get the differential neutrino flux corresponding to E1,
# differential flux of electron-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
Flux_CCatmo_NuEbar = np.interp(E1, E1_atmo, flux_CCatmo_NuEbar)

# differential flux in energy for no oscillation for muon-antineutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_CCatmo_NuMubar = 10 ** (-4) * np.array([0., 116., 128., 136., 150., 158., 162., 170., 196., 177., 182., 183., 181.,
                                             155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0, 55.6, 47.6, 40.8])
# linear interpolation of the simulated data above to get the differential neutrino flux corresponding to E1,
# differential flux of muon-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
Flux_CCatmo_NuMubar = np.interp(E1, E1_atmo, flux_CCatmo_NuMubar)

# Differential fluxes of electron- and muon-neutrinos
# (not needed because only antineutrino can contribute to the electron-antineutrino flux)
"""
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
"""

# total flux of electron-antineutrinos at the detector WITHOUT oscillation in (MeV**(-1) * cm**(-2) * s**(-1))
# (factors set to 1 and 0) (np.array of float):
# TODO: consider the flux with oscillation
Flux_CCatmospheric_NuEbar = 0.547 * Flux_CCatmo_NuEbar + 0.239 * Flux_CCatmo_NuMubar
# Theoretical spectrum (in events per MeV) of electron-antineutrinos ("number of positron-events") from
# inverse beta decay on free protons (from paper 0903.5323.pdf, equ. 64) (np.array of float):
# TODO: "dispersion" is not considered
TheoSpectrum_CCatmospheric = Flux_CCatmospheric_NuEbar * sigma_IBD * detection_eff * N_target * time

""" Visible spectrum of the atmospheric charged-current electron-antineutrino background, 
    theoretical spectrum is convolved with Faltung (gaussian distribution): """
# generate (number * number of atmospheric-antineutrinos)-many antineutrinos with energies E_neutrino_atmo in MeV
# from the theoretical spectrum (np.array of float64):
E_neutrino_CCatmospheric = energy_neutrino(E1, TheoSpectrum_CCatmospheric, number)
# array of visible energies of atmospheric CC neutrinos in MeV (np.array of float):
E_visible_CCatmospheric = visible_spectrum(E_neutrino_CCatmospheric, DELTA, MASS_POSITRON, number_vis)


print("... simulation of reactor neutrino background...")

""" SIMULATE THE REACTOR ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: """

"""
List of assumptions or estimations made for the calculation of the reactor-antineutrino background:
1. the reactor electron-antineutrino flux is calculated according to the offline Inverse-Beta generator
   KRLReactorFlux.cc and KRLReactorFlux.hh, which is based on the paper of Vogel and Engel 1989 (PhysRevD.39.3378)
2. neutrino-oscillation is taken into account. same calculation like in NuOscillation.cc and NuOscillation.hh. 
   Normal Hierarchy is considered
"""

""" Theoretical spectrum of reactor electron-antineutrinos: """
""" Flux of reactor electron-antineutrinos calculated with the code described 
    in KRLReactorFlux.cc and KRLReactorFlux.hh"""
# Total thermal power of the Yangjiang and Taishan nuclear power plants (from PhysicsReport), in GW (float):
power_th = 35.73
# Coefficients from KRLReactorFlux.cc (data taken from Vogel and Engel 1989 (PhysRevD.39.3378), table 1)
# (np.array of float):
Coeff235U = np.array([0.870, -0.160, -0.0910])
Coeff239Pu = np.array([0.896, -0.239, -0.0981])
Coeff238U = np.array([0.976, -0.162, -0.0790])
Coeff241Pu = np.array([0.793, -0.080, -0.1085])
# Fractions (data taken from KRLReactorFLux.cc) (float):
Fraction235U = 0.6
Fraction239Pu = 0.3
Fraction238U = 0.05
Fraction241Pu = 0.05
# fit to the electron-antineutrino spectrum, in units of electron-antineutrinos/(MeV * fission)
# (from KRLReactorFlux.cc and Vogel/Engel 1989 (PhysRevD.39.3378), equation 4) (np.array of float):
U235 = np.exp(Coeff235U[0] + Coeff235U[1]*E1 + Coeff235U[2]*E1**2)
Pu239 = np.exp(Coeff239Pu[0] + Coeff239Pu[1]*E1 + Coeff239Pu[2]*E1**2)
U238 = np.exp(Coeff238U[0] + Coeff238U[1]*E1 + Coeff238U[2]*E1**2)
Pu241 = np.exp(Coeff241Pu[0] + Coeff241Pu[1]*E1 + Coeff241Pu[2]*E1**2)
# add the weighted sum of the terms, electron-antineutrino spectrum in units of electron-antineutrinos/(MeV * fission)
# (data taken from KRLReactorFLux.cc) (np.array of float):
spec1_reactor = U235*Fraction235U + Pu239*Fraction239Pu + U238*Fraction238U + Pu241*Fraction241Pu
# There are 3.125*10**19 fissions/GW/second, spectrum in units of electron-antineutrino/(MeV * GW * s)
# (np.array of float):
spec2_reactor = spec1_reactor * 3.125*10**19
# There are about 3.156*10**7 seconds in a year, spectrum in units of electron-antineutrino/(MeV * GW * year)
# (np.array of float):
spec3_reactor = spec2_reactor * 3.156*10**7
# electron-antineutrino flux in units of electron-antineutrino/(MeV * year) (np.array of float):
flux_reactor = spec3_reactor * power_th

""" Consider neutrino oscillation for NORMAL HIERARCHY from NuOscillation.cc: """
# Oscillation parameters:
# distance reactor to detector in meter (float):
L_m = 5.25*10**4
# distance reactor to detector in centimeter (float):
L_cm = L_m * 100
# mixing angles from PDG 2016 (same in NuOscillation.cc) (float):
sin2_th12 = 0.297
sin2_th13 = 0.0214
# mass squared differences in eV**2 from PDG 2016 (same in NuOscillation.cc) (float):
Dm2_21 = 7.37*10**(-5)
Dm2_31 = 2.50*10**(-3) + Dm2_21/2
Dm2_32 = 2.50*10**(-3) - Dm2_21/2
# calculate the other parameters (float):
cos2_th12 = 1. - sin2_th12
sin2_2th12 = 4.*sin2_th12*cos2_th12
cos2_th13 = 1. - sin2_th13
sin2_2th13 = 4.*sin2_th13*cos2_th13
cos4_th13 = cos2_th13**2
# With these parameters calculate survival probability of electron-antineutrinos (np.array of float):
P21 = sin2_2th12 * cos4_th13 * np.sin(1.267 * Dm2_21 * L_m / E1)**2
P31 = sin2_2th13 * cos2_th12 * np.sin(1.267 * Dm2_31 * L_m / E1)**2
P32 = sin2_2th13 * sin2_th12 * np.sin(1.267 * Dm2_32 * L_m / E1)**2
# Survival probability of electron-antineutrinos for Normal Hierarchy (np.array of float):
Prob_oscillation_NH = 1. - P21 - P31 - P32

""" Theoretical reactor electron-antineutrino spectrum in JUNO with oscillation """
# Theoretical spectrum in JUNO for normal hierarchy with oscillation in units of electron-antineutrinos/MeV
# in "t_years" years (np.array of float):
TheoSpectrum_reactor = 1 / (4*np.pi*L_cm**2) * flux_reactor * sigma_IBD * detection_eff * N_target * t_years\
                       * Prob_oscillation_NH

""" Visible spectrum of the reactor electron-antineutrino background, theoretical spectrum is convolved 
    with Faltung (gaussian distribution): """
# generate (number * number of reactor-antineutrinos)-many antineutrinos with energies E_neutrino_reactor in MeV
# from the theoretical spectrum (np.array of float64):
E_neutrino_reactor = energy_neutrino(E1, TheoSpectrum_reactor, number)
# array of visible energies of reactor neutrinos in MeV (np.array of float):
E_visible_reactor = visible_spectrum(E_neutrino_reactor, DELTA, MASS_POSITRON, number_vis)


""" SIMULATE THE NEUTRAL CURRENT ATMOSPHERIC NEUTRINOS BACKGROUND IN JUNO: """
# TODO: Neutral Current atmospheric background has to be added


# Total theoretical spectrum in 1/(MeV) in "time" years:
TheoSpectrum_total = TheoSpectrum_signal + TheoSpectrum_DSNB + TheoSpectrum_CCatmospheric + TheoSpectrum_reactor
# Total visible energy in MeV:
E_visible_total = np.append(E_visible_reactor, np.append(E_visible_CCatmospheric,
                                                         np.append(E_visible_signal, E_visible_DSNB)))

""" Save the simulated data to txt-file: """
"""
print("... save data to file...")

# save TheoSpectrum to txt file:
np.savetxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}_TheoSpectrum.txt'
           .format(mass_DM, E1[0], E1[-1]), TheoSpectrum_total, fmt='%1.4e',
           header='Total theoretical spectrum in MeV in JUNO after {0:.0f} years and '
                  'for DM of mass {1:.0f} MeV (E1 from {2:.0f}MeV to {3:.0f}MeV '
                  'with interval {4:.4f}MeV):'
           .format(t_years, mass_DM, E1[0], E1[-1], interval))
# save E_visible_signal to txt file:
np.savetxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}_E_visible_signal.txt'
           .format(mass_DM, E1[0], E1[-1]), E_visible_signal, fmt='%1.4e',
           header='Visible energy of the DM annihilation signal in MeV in JUNO after {0:.0f} years and for DM of '
                  'mass {1:.0f} MeV (number = {2:d}, E1 from {3:.0f}MeV to {4:.0f}MeV with interval {5:.4f}MeV):'
           .format(t_years, mass_DM, number*number_vis, E1[0], E1[-1], interval))
# save E_visible_DSNB to txt file:
np.savetxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}_E_visible_DSNB.txt'
           .format(mass_DM, E1[0], E1[-1]), E_visible_DSNB, fmt='%1.4e',
           header='Visible energy of the DSNB background in MeV in JUNO after {0:.0f} years and for DM of '
                  'mass {1:.0f} MeV (number = {2:d}, E1 from {3:.0f}MeV to {4:.0f}MeV with interval {5:.4f}MeV):'
           .format(t_years, mass_DM, number*number_vis, E1[0], E1[-1], interval))
# save E_visible_CCatmospheric to txt file:
np.savetxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}_'
           'E_visible_CCatmospheric.txt'
           .format(mass_DM, E1[0], E1[-1]), E_visible_CCatmospheric, fmt='%1.4e',
           header='Visible energy of the CC atmospheric background in MeV in JUNO after {0:.0f} years and for DM of '
                  'mass {1:.0f} MeV (number = {2:d}, E1 from {3:.0f}MeV to {4:.0f}MeV with interval {5:.4f}MeV):'
           .format(t_years, mass_DM, number*number_vis, E1[0], E1[-1], interval))
# save E_visible_reactor to txt file:
np.savetxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}_E_visible_reactor.txt'
           .format(mass_DM, E1[0], E1[-1]), E_visible_reactor, fmt='%1.4e',
           header='Visible energy of the reactor neutrino background in MeV in JUNO after {0:.0f} years and for DM of '
                  'mass {1:.0f} MeV (number = {2:d}, E1 from {3:.0f}MeV to {4:.0f}MeV with interval {5:.4f}MeV):'
           .format(t_years, mass_DM, number*number_vis, E1[0], E1[-1], interval))
"""
h1 = pyplot.figure(1)
pyplot.plot(E1, Flux_CCatmo_NuEbar, 'b', label='electron-antineutrino flux without oscillation')
pyplot.plot(E1, Flux_CCatmo_NuMubar, 'r', label='muon-antineutrino flux without oscillation')
pyplot.plot(E1, Flux_CCatmospheric_NuEbar, 'k', label='0.547 * flux_nuEbar + 0.239 * flux_nuMubar, Peres')
pyplot.xlim(E1[0], E1[-1])
pyplot.xlabel('Neutrino energy in MeV')
pyplot.ylabel('Differential flux in 1/(MeV*s*cm**2)')
pyplot.title('Differential fluxes of atmospheric CC background for solar average at the site of '
             'Super-K')
pyplot.legend()
pyplot.grid()

"""
print("... display plots...")

# Display the theoretical spectra with the settings below:
h1 = pyplot.figure(1)
pyplot.plot(E1, TheoSpectrum_total, 'k', label='total spectrum')
pyplot.plot(E1, TheoSpectrum_signal, 'r--', label='signal from DM annihilation')
pyplot.plot(E1, TheoSpectrum_DSNB, 'b--', label='DSNB background')
pyplot.plot(E1, TheoSpectrum_CCatmospheric, 'g--', label='atmospheric CC background without oscillation')
pyplot.plot(E1, TheoSpectrum_reactor, 'c--', label='reactor electron-antineutrino background')
pyplot.xlim(E1[0], E1[-1])
pyplot.ylim(ymin=0, ymax=25)
# pyplot.xticks(np.arange(4.0, E1[-1]), 2.0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Theoretical spectrum dN/dE in 1/MeV")
pyplot.title("Theoretical electron-antineutrino spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
             .format(t_years, mass_DM))
pyplot.legend()


# Display the expected spectra with the settings below:
h2 = pyplot.figure(2)
n_total, bins3, patches3 = pyplot.hist(E_visible_total, bins=E2, histtype='step', color='k', label='total spectrum')
n_signal, bins1, patches1 = pyplot.hist(E_visible_signal, bins=E2, histtype='step', color='r',
                                        label='signal from DM annihilation')
n_DSNB, bins2, patches2 = pyplot.hist(E_visible_DSNB, bins=E2, histtype='step', color='b', label='DSNB background')
n_atmospheric, bins4, patches4 = pyplot.hist(E_visible_CCatmospheric, bins=E2, histtype='step', color='g',
                                             label='atmospheric CC background without oscillation')
n_reactor, bins5, patches5 = pyplot.hist(E_visible_reactor, bins=E2, histtype='step', color='c',
                                         label='reactor electron-antineutrino background')
pyplot.xlim(E2[0], E2[-1])
pyplot.ylim(ymin=0, ymax=25)
# pyplot.xticks(np.arange(2.0, E2[-1]), 2.0)
pyplot.xlabel("Visible energy in MeV")
pyplot.ylabel("Expected spectrum dN/dE in 1/MeV * {0:d}".format(number*number_vis))
pyplot.title("Expected spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
             .format(t_years, mass_DM))
pyplot.legend()
"""
pyplot.show()

print('... simulation finished\n'
      '#######################\n'
      'Change the name of the generated files to avoid overwriting!!\n'
      '########################')
