""" Script to generate and simulate the visible spectrum in the JUNO detector in the energy range
    from few MeV to hundred MeV:
    - for signal of DM annihilation for different DM masses
    - for DSNB background for different values of E_mean, beta and f_Star
    - for CC atmospheric neutrino background for different values of Oscillation, Prob_e_to_e, Prob_mu_to_e
    - for reactor anti-neutrino background for different values of power_thermal, Fraction_U235, Fraction_U238,
    Fraction_Pu239, Fraction_Pu241 and L_meter

    ->  the script works fine, BUT is NOT very convenient
    ->  in this script: the theoretical spectra are calculated, then the neutrino energies are randomly generated from
        the theoretical spectrum, then the neutrino energies are convolved with a gaussian defined by the energy
        resolution and the correlation of visible and neutrino energy to generate the visible energies
        ->  this is correct BUT not very convenient, because this way is slow and it is not necessary to simulate the
            MC simulation for such high statistics

    ->  therefore: use the script gen_spectrum_v1.py to generate the electron-antineutrino spectrum (here the
        theoretical spectrum is calculated, then the whole spectrum is convolved with the gaussian defined by
        energy resolution and correlation of visible and neutrino energy. This spectrum correspond to the visible
        spectrum in the detector.
    """

# import of the necessary packages:
import datetime
import numpy as np
from math import gamma
from matplotlib import pyplot

""" Set boolean values to define, what is simulated in the code, if the data is saved, if spectra are displayed, and 
    if a data-set or MC spectrum is generated: """
# generate signal from DM annihilation:
DM_SIGNAL = False
# generate DSNB background:
DSNB_BACKGROUND = False
# generate CC atmospheric background:
CCATMOSPHERIC_BACKGROUND = True
# generate reactor antineutrino background:
REACTOR_BACKGROUND = False

# save the data:
SAVE_DATA = False
# display the generated spectra:
DISPLAY_SPECTRA = True
# difference between data-set and MC spectrum is defined by "number"

""" Variables: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")
# number defines, how many "pseudo"-neutrinos are generated per real neutrino in the Script
# to get higher statistics (integer):
number = 10
# number_dataset (number_dataset_stop - number_dataset_start) defines, how many datasets are generated
# (one dataset is the visible spectrum for number = 1, MC simulation is defined for number > 1)
# number_dataset_start defines the start point (integer):
number_dataset_start = 1
# number_dataset_stop defines the end point (integer):
number_dataset_stop = 2
# number_MCsimu defines a serially numbered value to save the txt files in a serially numbered way (integer):
number_MCsimu = 100
# number_vis defines, how many random numbers E_visible_".." are generated from the gaussian distribution
# "faltung" around E_neutrino_".." (integer):
number_vis = 1

""" Dark Matter mass in MeV:"""
# Dark Matter mass in MeV (float):
mass_DM = 20.0

""" energy-array: """
# energy corresponding to the electron-antineutrino energy in MeV (np.array of float64):
interval = 1.0
E1 = np.arange(5, 105, interval)

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
detection_eff = 0.73  # TODO: detection efficiency is set to 73 from physics_report p. 40 table 2.1


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
        spectrum theospectrum.
        the number of neutrinos generated for a Dataset is defined by the poisson-distributed values
        around n_neutrino.
        the number of neutrinos generated for MC simulation is defined by the rounded value of num*n_neutrino.
        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        theospectrum: theoretical neutrino spectrum in 1/MeV (np.array of float)
        num: defines, how many "pseudo"-neutrinos are generated per real neutrino to get higher statistics (integer)

        return: e_neutrino: array of neutrino energies generated from the theoretical spectrum (np.array of float)
                n_neutrino_generated: number of neutrinos, which are randomly generated from the theoretical spectrum
                (integer)
        """
    # first integrate the theoretical spectrum over energy to get the number of neutrinos (float64):
    n_neutrino = np.trapz(theospectrum, energy)
    # Then normalize the theoretical spectrum theospectrum, so that sum(theospectrum) = 1.
    # Then pdf_theospectrum can be used as probability in np.random.choice (np.array of float64):
    pdf_theospectrum = theospectrum / sum(theospectrum)
    # differ between simulation of a Dataset and a MC simulation:
    # for Dataset (num == 1), for MC simulation (num > 1):
    if num == 1:
        # the number of neutrinos generated for a Dataset is defined by the poisson-distributed values
        # around n_neutrino (The detector measures an integer number of neutrinos, which is poisson distributed around
        # the value n_neutrino (normally n_neutrino is a float number and you can not generate a float number of
        # random numbers).
        # Generate a poisson distributed integer value around n_neutrino (np.array of integer):
        n_neutrino_generated = np.random.poisson(n_neutrino, 1)
    else:
        # the number of neutrinos generated for MC simulation is defined by the rounded value of num*n_neutrino
        # (this is valid, if num is large enough (num > 1000):
        # round num*n_neutrino to get the number of neutrino, which are generated (integer):
        n_neutrino_generated = round(float(num * n_neutrino))
    # Then generate n_neutrino_generated-many neutrinos with energies e_neutrino from the theoretical spectrum
    # (np.array of float64):
    e_neutrino = np.random.choice(energy, n_neutrino_generated, p=pdf_theospectrum)
    return e_neutrino, n_neutrino_generated


def correlation_vis_neutrino(e_neutrino_entry, mass_proton, mass_neutron, mass_positron):
    """ correlation between E_visible and e_neutrino from Inverse Beta Decay from paper of Strumia/Vissani
        'Precise quasielastic neutrino/nucleon cross section', 0302055_IBDcrosssection.pdf:
        The average lepton energy E_positron_Strumia is approximated by the equation 16 in the paper
        (at better than 1 percent below energies of around 100 MeV).
        This is a much better approximation than the usual formula E_positron_Strumia = E_neutrino - DELTA.
        It permits ('erlaubt') to relate E_positron_Strumia with E_neutrino incorporating ('unter Berücksichtigung')
        a large part of the effect due to the recoil of the nucleon.
        e_neutrino_entry: ONE entry of the e_neutrino np.array in MeV, the e_neutrino np.array contains the
                          neutrino energies generated from the theoretical spectrum (float)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        """
    # Correlation between visible and neutrino energy in MeV:
    # first define some useful term for the calculation (d=delta from page 3 equ. 12, s from page 3):
    d = (mass_neutron**2 - mass_proton**2 - mass_positron**2) / (2 * mass_proton)
    s = 2 * mass_proton * e_neutrino_entry + mass_proton**2
    # neutrino energy in center of mass (cm) frame in MeV, page 4, equ. 13:
    e_neutrino_cm = (s - mass_proton**2) / (2 * np.sqrt(s))
    # positron energy in CM frame in MeV, page 4, equ. 13:
    e_positron_cm = (s - mass_neutron**2 + mass_positron**2) / (2 * np.sqrt(s))
    # Average lepton energy in MeV, which can be approximated (at better than 1 percent below ~ 100 MeV) by
    # (page 5, equ. 16):
    e_positron = e_neutrino_entry - d - e_neutrino_cm * e_positron_cm / mass_proton
    # prompt visible energy in the detector in MeV:
    corr_vis_neutrino = e_positron + mass_positron
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


def visible_spectrum(e_neutrino, mass_proton, mass_neutron, mass_positron, num):
    """ generate an array of the visible energies in the detector from the real neutrino energies, which are generated
        from the theoretical spectrum (theoretical spectrum is convolved with gaussian distribution):
        e_neutrino: e_neutrino np. array contains the neutrino energies generated from the theoretical spectrum
                    (np.array of float)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        num: defines, how many random numbers E_visible are generated from the gaussian
             distribution around E_neutrino (integer)
        """
    # Convolve the generated energies e_neutrino with the faltung defined below for every entry in e_neutrino:
    # Preallocate e_visible (empty array):
    e_visible = np.array([])
    # loop over entries in e_neutrino, for each energy generate num random numbers:
    for index1 in np.arange(len(e_neutrino)):
        # correlation between E_visible and E_neutrino (float):
        corr_vis_neutrino = correlation_vis_neutrino(e_neutrino[index1], mass_proton, mass_neutron, mass_positron)
        # energy resolution of the detector (float):
        sigma_resolution = energy_resolution(corr_vis_neutrino)
        # generate (num)-many random numbers around e_neutrino[index] from convolution, which is described
        # by a gaussian distribution with mean = corr_vis_neutrino and sigma = sigma_resolution (np.array of float64):
        e_visible = np.append(e_visible, np.random.normal(corr_vis_neutrino, sigma_resolution, num))
    return e_visible


def numberflux_dsnb(energy, e_mean, beta, luminosity, redshift, c, hubble_0, f_sn, h_param):
    """ function calculates the number-flux of electron-antineutrinos (Nu_E_bar OR muon- and tau-antineutrinos
        (Nu_x_bar) from diffuse supernova neutrino background (DSNB). The calculation is based on paper of Ando and
        Sato 'Relic neutrino background from cosmological supernovae' from 2004, arXiv:astro-ph/0410061v2

        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        e_mean: mean energy of DSNB antineutrinos in MeV (float)
        beta: pinching parameters (float)
        luminosity: neutrino luminosity in MeV (1 erg = 624151 MeV) (float)
        redshift: redshift z is variable to calculate the flux (np.array of float64)
        c: speed of light in vacuum (float)
        hubble_0: Hubble constant in 1/s (float)
        f_sn: correction factor (float)
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
    numberflux = c / hubble_0 * 0.0122 * 0.32 * f_sn * h_param * flux_constant * flux_integral

    return numberflux


def darkmatter_signal(energy, mass_dm, crosssection, n_target, t, detection_efficiency, num, mass_proton, mass_neutron,
                      mass_positron, num_vis):
    """ Simulate the signal from neutrinos from DM annihilation in the Milky Way:

        List of assumptions or estimations made for the calculation of the signal spectrum:
        1. neutrino flux (phi_signal) is calculated as described in paper 0710.5420
        2. theoretical signal spectrum (N_neutrino) is calculated as described in paper 0710.5420:
        2.1. only IBD on free protons is considered
        3. Delta-function is approximated as very thin gaussian distribution
        3.1. Number of neutrinos with energy e_neutrino_signal is defined by round(float(n_neutrino_signal))
        and values of e_neutrino_signal are generated by a random number of a gaussian distribution with mean = MASS_DM
        and sigma = epsilon -> is equal to generating from theo_spectrum_signal
        3.2. Loop from 1 to length(e_neutrino_signal) and convolve each entry in e_neutrino_signal with faltung
        3.3. faltung is defined by correlation_vis_neutrino and energy_resolution

        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        mass_dm: dark matter mass in MeV (float)
        crosssection: IBD cross-section for the DM signal in cm**2 (float), produced with the function sigma_ibd()
        n_target: number of free protons in JUNO (float)
        t: exposure time in seconds (float)
        detection_efficiency: detection efficiency of IBD in JUNO (float)
        num: defines, how many "pseudo"-neutrinos are generated per real neutrino to get higher statistics (integer)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        num_vis: defines, how many random numbers E_visible are generated from the gaussian
             distribution around E_neutrino (integer)

    :return theo_spectrum_signal: Theoretical spectrum of the electron-antineutrino signal in 1/MeV
            (number of events as function of the electron-antineutrino energy) (np.array of float64)
            e_visible_signal: array of visible energies in MeV (np.array of float)
            n_neutrino_signal: number of electron-antineutrino events in JUNO after "time" (float64)
            n_neutrino_signal_gen: number of electron-antineutrinos events really measured in JUNO (integer)
            sigma_anni: DM annihilation cross-section in cm**3/s (float)
            phi_signal: electron-antineutrino flux at Earth in 1/(MeV * s * cm**2) (float)
    """

    """ Theoretical spectrum of the electron-antineutrino signal, monoenergetic spectrum at mass_DM: """
    # Calculate the electron-antineutrino flux from DM annihilation in the Milky Way at Earth from paper 0710.5420:
    # DM annihilation cross-section necessary to explain the observed abundance of DM in the Universe,
    # in cm**3/s (float):
    sigma_anni = 3 * 10 ** (-26)
    # canonical value of the angular-averaged intensity over the whole Milky Way (integer):
    j_avg = 5
    # solar radius circle in cm, 8.5 kiloparsec, 1kpc = 3.086*10**21 cm (float):
    r_solar = 8.5 * 3.086 * 10 ** 21
    # normalizing DM density, in MeV/cm**3 (float):
    rho_0 = 0.3 * 1000
    # electron-antineutrino flux at Earth in 1/(MeV * s *cm**2) (float):
    phi_signal = sigma_anni / 6 * j_avg * r_solar * rho_0 ** 2 / mass_dm ** 2
    # Number of electron-antineutrino events in JUNO after "time" years in 1/MeV (float64):
    n_neutrino_signal = crosssection * phi_signal * n_target * t * detection_efficiency

    # delta function from theoretical spectrum is approximated as very thin gaussian distribution
    # (ONLY NEEDED TO DISPLAY THE THEORETICAL SPECTRUM):
    # epsilon defines sigma in the gaussian distribution (float):
    epsilon = 10 ** (-5)
    delta_function = 1 / (np.sqrt(2 * np.pi * epsilon ** 2)) * np.exp(-0.5 * (energy - mass_dm) ** 2 / epsilon ** 2)
    # normalize delta_function (integral(delta_function) = 1) (array of float64):
    delta_function = delta_function / sum(delta_function)
    # Theoretical spectrum of the electron-antineutrino signal in 1/MeV (number of events as function of the
    # electron-antineutrino energy) (array of float64):
    theo_spectrum_signal = n_neutrino_signal * delta_function
    # differ between simulation of a Dataset and a MC simulation:
    # for Dataset (num == 1), for MC simulation (num > 1):
    if num == 1:
        # the number of neutrinos generated for a Dataset is defined by the poisson-distributed values
        # around n_neutrino_signal.
        # Generate a poisson distributed integer value around n_neutrino_signal (np.array of integer):
        n_neutrino_signal_gen = np.random.poisson(n_neutrino_signal, 1)
    else:
        # the number of neutrinos generated for MC simulation is defined by the rounded value of num*n_neutrino
        # (this is valid, if num is large enough (num > 1000):
        # round num*n_neutrino to get the number of neutrino, which are generated (integer):
        n_neutrino_signal_gen = round(float(num * n_neutrino_signal))
    # Generate n_neutrino_signal_gen-many neutrinos of energy e_neutrino_signal from the theoretical spectrum
    # theo_spectrum_signal (e_neutrino_signal is generated from gaussian distribution with mean=mass_DM and
    # sigma=epsilon and not from theo_spectrum_signal, because there are problems of np.random.choice() with
    # this very thin distribution).
    # array of neutrino energies generated from theoretical spectrum in MeV (np.array of float):
    e_neutrino_signal = np.random.normal(mass_dm, epsilon, n_neutrino_signal_gen)

    """ Visible spectrum of the electron-antineutrino signal, theoretical spectrum is convolved 
        with Faltung (gaussian distribution): """
    # array of visible energies in MeV (np.array of float):
    e_visible_signal = visible_spectrum(e_neutrino_signal, mass_proton, mass_neutron, mass_positron, num_vis)

    return theo_spectrum_signal, e_visible_signal, n_neutrino_signal, n_neutrino_signal_gen, sigma_anni, phi_signal


def dsnb_background(energy, crosssection, n_target, t, detection_efficiency, c, num, mass_proton, mass_neutron,
                    mass_positron, num_vis):
    """ Simulate the DSNB electron-antineutrino background:

        List of assumptions or estimations made for the calculation of the background spectrum from
        DSNB electron-antineutrinos:
        1. DSNB calculation based on the calculation in the paper of Ando and Sato 'Relic neutrino background from
           cosmological supernovae' from 2004, arXiv:astro-ph/0410061v2
        2. only electron-antineutrinos interacting with free protons are considered (Inverse Beta Decay),
           dominant up to ~80 MeV
        3. for the star formation rate (SFR) per unit comoving volume, a model is adopted based on recent progressive
           results of the rest-frame UV, NIR H_alpha, and FIR/sub-millimeter observations (depending on the correction
           factor f_star). behaviours in higher redshift regions z > 1 are not clear at all.
        4. the supernova rate from SFR is assumed by the Salpeter IMF with a lower cutoff around 0.5 solar masses and
           that all stars with masses greater than 8 solar masses explode as core-collapse supernovae.
        5. for the neutrino spectrum from each supernova, three reference models by different groups are adopted:
           simulations by Lawrence Livermore group (LL), simulations by Thomson, Burrows, Pinto (TBP),
           MC study of spectral formation Keil, Raffelt, Janka (KRJ).
           The most serious problem is that the recent sophisticated hydrodynamic simulations have not obtained the
           supernova explosion itself; the shock wave cannot penetrate the entire core.
           The numerical simulation by the LL group is considered to be the most appropriate for our estimations,
           because it is the only model that succeeded in obtaining a robust explosion
        6. 70 percent of the electron-antineutrinos survive and 30 percent of the muon-/tau-antineutrinos appear
           as electron-antineutrinos at the Earth

        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        n_target: number of free protons in JUNO (float)
        t: exposure time in seconds (float)
        detection_efficiency: detection efficiency of IBD in JUNO (float)
        c: speed of light in vacuum (float)
        num: defines, how many "pseudo"-neutrinos are generated per real neutrino to get higher statistics (integer)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        num_vis: defines, how many random numbers E_visible are generated from the gaussian
             distribution around E_neutrino (integer)

    :return theospectrum_dsnb: Theoretical spectrum of the DSNB electron-antineutrino background in 1/MeV
            (number of events as function of the electron-antineutrino energy) (np.array of float64)
            e_visible_dsnb: array of visible energies in MeV (np.array of float)
            n_neutrino_dsnb: number of DSNB electron-antineutrino events in JUNO after "time" (float64)
            n_neutrino_dsnb_gen: number of DSNB electron-antineutrinos really measured in JUNO (integer)
            e_mean__nu_e_bar: mean energy of the electron-antineutrinos in MeV (float)
            beta__nu_e_bar: pinching parameter for electron-antineutrinos (float)
            e_mean__nu_bar: mean energy of the NON-electron-antineutrinos in MeV (float)
            beta__nu_bar: pinching parameter for NON-electron-antineutrinos (float)
            f_star: correction factor for the star formation rate
    """

    """ Theoretical spectrum of DSNB electron-antineutrino background: """
    # Calculate the electron-antineutrino flux from DSNB from paper 0410061 of Ando/Sato:
    # differential number flux is divided into electron-antineutrinos and non-electron antineutrinos,
    # then the oscillation of the flavors are considered and a final electron-antineutrino flux at Earth is calculated

    # redshift, where the gravitational collapse begun (integer) (page 5, equ. 3):
    z_max = 5
    # redshift is variable to calculate the flux (np.array of float64):
    z = np.arange(0, z_max, 0.01)
    # Hubble constant in 1/s (70 km/(s*Mpc) = 70 * 1000 m / (s*10**6*3.086*10**16 m ) (float) (page 4, equ. 2):
    h_0 = 70 * 1000 / (10 ** 6 * 3.086 * 10 ** 16)
    # correction factor described in page 5 (float):
    f_star = 1
    # parameter: is 1 for a value of h_0 = 70 km/(s*Mpc) (integer):
    h_70 = 1

    """ differential number flux for electron-antineutrinos: """
    # Fitting parameters from LL model:
    # mean energy of electron-antineutrinos in MeV (float) (page 7, table 1):
    e_mean__nu_e_bar = 15.4
    # pinching parameters (float) (page 7, table 1):
    beta__nu_e_bar = 3.8
    # neutrino luminosity in MeV (1 erg = 624151 MeV) (float) (page 7, table 1):
    l__nu_e_bar = 4.9 * 10**52 * 624151
    # differential number flux of electron-antineutrinos in 1/(MeV*s*cm**2) (np.array of float) (page 5, equ. 3):
    number_flux__nu_e_bar = numberflux_dsnb(energy, e_mean__nu_e_bar, beta__nu_e_bar, l__nu_e_bar, z,
                                            c, h_0, f_star, h_70)
    # differential number flux of electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
    # about 70 percent of electron-antineutrinos survive (np.array of float64) (page 9, equ. 7):
    number_flux__nu_e_bar = 0.7 * number_flux__nu_e_bar

    """ differential number flux for NON-electron-antineutrinos (muon- and tau-antineutrinos): """
    # Fitting parameters from LL model:
    # mean energy of NON-electron-antineutrinos in MeV (float) (page 7, table 1):
    e_mean__nu_bar = 21.6
    # pinching parameters (float) (page 7, table 1):
    beta__nu_bar = 1.8
    # neutrino luminosity in MeV (float) (page 7, table 1):
    l__nu_bar = 5.0 * 10**52 * 624151
    # differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) (np.array of float) (page 5, equ. 3):
    number_flux__nu_bar = numberflux_dsnb(energy, e_mean__nu_bar, beta__nu_bar, l__nu_bar, z, c, h_0, f_star, h_70)
    # differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
    # about 30 percent of the emitted muon- and tau-antineutrinos will appear as electron-antineutrinos
    # at the Earth (np.array of float64) (page 9, equ. 7):
    number_flux__nu_bar = 0.3 * number_flux__nu_bar

    # Total number flux of electron anti-neutrinos from DSNB at Earth/JUNO in 1/(MeV * s * cm**2) (np.array of float64):
    number_flux_dsnb = number_flux__nu_e_bar + number_flux__nu_bar

    # Theoretical spectrum of DSNB neutrino events in JUNO after "time" years in 1/MeV (np.array of float64):
    theospectrum_dsnb = crosssection * number_flux_dsnb * n_target * t * detection_efficiency

    # number of neutrino from DSNB background in JUNO detector after "time":
    n_neutrino_dsnb = np.trapz(theospectrum_dsnb, energy)

    """ Visible spectrum of the DSNB electron-antineutrino background, theoretical spectrum is convolved 
        with Faltung (gaussian distribution): """
    # generate n_neutrino_dsnb_gen - many antineutrinos with energies e_neutrino_dsnb in MeV
    # from the theoretical spectrum (np.array of float64):
    e_neutrino_dsnb, n_neutrino_dsnb_gen = energy_neutrino(energy, theospectrum_dsnb, num)
    # array of visible energies of DSNB neutrinos in MeV (np.array of float):
    e_visible_dsnb = visible_spectrum(e_neutrino_dsnb, mass_proton, mass_neutron, mass_positron, num_vis)

    return (theospectrum_dsnb, e_visible_dsnb, n_neutrino_dsnb, n_neutrino_dsnb_gen, e_mean__nu_e_bar, beta__nu_e_bar,
            e_mean__nu_bar, beta__nu_bar, f_star)


def ccatmospheric_background(energy, crosssection, n_target, t, detection_efficiency,
                             num, mass_proton, mass_neutron, mass_positron, num_vis):
    """ Simulate the atmospheric Charged Current electron-antineutrino background:

        List of assumptions or estimations made for the calculation of the atmospheric Charged Current Background:
        1. the values of the differential flux in energy for no oscillation for electron- and muon-antineutrinos
        for solar average at the site of Super-Kamiokande is taken from table 2 and 3 of Battistoni's paper 'The
        atmospheric neutrino flux below 100 MeV: the FLUKA results' from 2005 (1-s2.0-S0927650505000526-main):
            1.1. for only electron-antineutrinos -> is ok, because only IBD on free protons is considered
                 (CC-interaction on bound nuclei do not mimic IBD)
            1.2. for solar average -> is a good approximation
            1.3. at the site of Super-K -> is ok, because JUNO is located at a lower geographical latitude (22.6°N)
                 than Super-K (36.4°N) and therefore the flux in JUNO should be lower
                 -> Flux is overestimated a bit
        2. this data is linear interpolated to get the differential flux corresponding to the binning in E1,
        it is estimated that for e1_atmo = 0 MeV, flux_atmo is also 0.

        3. NO oscillation

        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        n_target: number of free protons in JUNO (float)
        t: exposure time in seconds (float)
        detection_efficiency: detection efficiency of IBD in JUNO (float)
        num: defines, how many "pseudo"-neutrinos are generated per real neutrino to get higher statistics (integer)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        num_vis: defines, how many random numbers E_visible are generated from the gaussian
             distribution around E_neutrino (integer)

    :return theospectrum_ccatmospheric: Theoretical spectrum of the atmospheric CC electron-antineutrino background
            in 1/MeV (number of events as function of the electron-antineutrino energy) (np.array of float64)
            e_visible_ccatmospheric: array of visible energies in MeV (np.array of float)
            n_neutrino_ccatmospheric: number of atmospheric CC electron-antineutrino events in JUNO after "time"
            (float64)
            n_neutrino_ccatmospheric_gen: number of atmospheric CC electron-antineutrino really measured in JUNO
            (integer)
            oscillation: oscillation is considered for oscillation=1, oscillation is not considered for oscillation=0
            (integer)
            prob_e_to_e: survival probability of electron-antineutrinos (electron-antineutrinos oscillate to
            electron-antineutrinos) (float)
            prob_mu_to_e: oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos) (float)
    """

    """ Theoretical spectrum of atmospheric charged-current background: """
    # Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
    e1_atmo = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133, 150,
                        168, 188])

    # differential flux in energy for no oscillation for electron-antineutrinos for solar average at the site
    # of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_e_bar = 10 ** (-4) * np.array([0., 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                                  83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9,
                                                  21.3, 18.3])
    # linear interpolation of the simulated data above to get the differential neutrino flux corresponding to energy,
    # differential flux of electron-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_e_bar = np.interp(energy, e1_atmo, flux_ccatmo_nu_e_bar)

    # differential flux in energy for no oscillation for muon-antineutrinos for solar average at the site of Super-K,
    # in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_mu_bar = 10 ** (-4) * np.array([0., 116., 128., 136., 150., 158., 162., 170., 196., 177., 182.,
                                                   183., 181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0,
                                                   55.6, 47.6, 40.8])
    # linear interpolation of the simulated data above to get the differential neutrino flux corresponding to energy,
    # differential flux of muon-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_mu_bar = np.interp(energy, e1_atmo, flux_ccatmo_nu_mu_bar)

    # total flux of electron-antineutrinos at the detector WITHOUT oscillation in (MeV**(-1) * cm**(-2) * s**(-1))
    # (factors set to 1 and 0) (np.array of float):
    # TODO: consider the flux with oscillation
    # Integer, that defines, if oscillation is considered (oscillation=1) or not (oscillation=0):
    oscillation = 0
    # survival probability of electron-antineutrinos:
    prob_e_to_e = 1
    # oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos):
    prob_mu_to_e = 0
    flux_total_ccatmospheric_nu_e_bar = prob_e_to_e * flux_ccatmo_nu_e_bar + prob_mu_to_e * flux_ccatmo_nu_mu_bar
    # Theoretical spectrum (in events per MeV) of electron-antineutrinos ("number of positron-events") from
    # inverse beta decay on free protons (from paper 0903.5323.pdf, equ. 64) (np.array of float):
    # TODO: "dispersion" is not considered
    theospectrum_ccatmospheric = flux_total_ccatmospheric_nu_e_bar * crosssection * detection_efficiency * n_target * t

    # number of neutrino from CC atmospheric background in JUNO detector after "time":
    n_neutrino_ccatmospheric = np.trapz(theospectrum_ccatmospheric, energy)

    """ Visible spectrum of the atmospheric charged-current electron-antineutrino background, 
        theoretical spectrum is convolved with Faltung (gaussian distribution): """
    # generate n_neutrino_ccatmospheric_gen - many antineutrinos with energies e_neutrino_ccatmospheric in MeV
    # from the theoretical spectrum (np.array of float64):
    e_neutrino_ccatmospheric, n_neutrino_ccatmospheric_gen = energy_neutrino(energy, theospectrum_ccatmospheric, num)
    # array of visible energies of atmospheric CC neutrinos in MeV (np.array of float):
    e_visible_ccatmospheric = visible_spectrum(e_neutrino_ccatmospheric, mass_proton, mass_neutron,
                                               mass_positron, num_vis)

    return (theospectrum_ccatmospheric, e_visible_ccatmospheric, n_neutrino_ccatmospheric, n_neutrino_ccatmospheric_gen,
            oscillation, prob_e_to_e, prob_mu_to_e)


def reactor_background(energy, crosssection, n_target, tyears, detection_efficiency,
                       num, mass_proton, mass_neutron, mass_positron, num_vis):
    """ Simulate the reactor electron-antineutrino background in JUNO:

        List of assumptions or estimations made for the calculation of the reactor-antineutrino background:
        1. the reactor electron-antineutrino flux is calculated according to the paper of Fallot2012
           (PhysRevLett.109.202504.pdf, New antineutrino Energy Spectra Predictions from the Summation of Beta
           Decay Branches of the Fission Products). The fluxes of electron-antineutrinos for each fission product are
           digitized from figure 1 with software engauge digitizer.
        2. neutrino-oscillation is taken into account: same calculation like in NuOscillation.cc and NuOscillation.hh.
           Normal Hierarchy is considered

        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        n_target: number of free protons in JUNO (float)
        tyears: exposure time in years (float)
        detection_efficiency: detection efficiency of IBD in JUNO (float)
        num: defines, how many "pseudo"-neutrinos are generated per real neutrino to get higher statistics (integer)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        num_vis: defines, how many random numbers E_visible are generated from the gaussian
             distribution around E_neutrino (integer)

    :return: theospectrum_reactor: Theoretical spectrum of the reactor electron-antineutrino background
             in 1/MeV (number of events as function of the electron-antineutrino energy) (np.array of float64)
             e_visible_reactor: array of visible energies in MeV (np.array of float)
             n_neutrino_reactor: number of reactor electron-antineutrino events in JUNO after "time"
             (float64)
             n_neutrino_reactor_gen: number of reactor electron-antineutrinos really measured in JUNO (integer)
             power_th: total thermal power of the Yangjiang and Taishan NPP in GW (float)
             fraction235_u: fission fraction of U235 (float)
             fraction238_u: fission fraction of U238 (float)
             fraction239_pu: fission fraction of Pu239 (float)
             fraction241_pu: fission fraction of Pu241 (float)
             l_m: distance from reactor to detector in meter (float)
    """

    """ Theoretical spectrum of reactor electron-antineutrinos: """
    """ Flux of reactor electron-antineutrinos calculated with the code described 
        in KRLReactorFlux.cc and KRLReactorFlux.hh """
    # Total thermal power of the Yangjiang and Taishan nuclear power plants (from PhysicsReport), in GW (float):
    power_th = 35.73
    # Digitized data:
    # corresponding electron-antineutrino energy for U235 in MeV (one data-point of 17 MeV is added, important for
    # interpolation) (np.array of float):
    energy_fallot_u235 = np.array([0.3841, 0.7464, 1.1087, 1.471, 1.7609, 2.1232, 2.4855, 2.8478, 3.1377, 3.5, 3.8623,
                                   4.1522, 4.5145, 4.8043, 5.0942, 5.3841, 5.7464, 6.0362, 6.3261, 6.6159, 6.9058,
                                   7.1957, 7.4855, 7.7754, 7.9928, 8.1377, 8.3551, 8.6449, 8.8623, 9.1522, 9.442,
                                   9.7319, 9.8768, 10.1667, 10.529, 10.8188, 11.1087, 11.3986, 11.6884, 11.8333,
                                   12.0507, 12.1232, 12.1957, 12.2681, 12.3406, 12.4855, 12.7754, 13.1377, 13.3551,
                                   13.5725, 13.9348, 14.2246, 14.442, 14.7319, 14.9493, 15.0217, 15.0942, 15.1667,
                                   15.3841, 15.6739, 15.8913, 16.1087, 17.0])
    # electron-antineutrino flux from figure 1 for U235 in antineutrinos/(MeV*fission) (data-point of
    # 0.0 events/(MeV*fission) for 17 MeV ia added) (np.array of float):
    flux_fallot_u235 = np.array([1.548, 1.797, 1.797, 1.548, 1.334, 1.149, 0.852, 0.734, 0.5446, 0.4041, 0.3481, 0.2224,
                                 0.165, 0.1225, 0.0909, 0.06741, 0.05001, 0.03196, 0.02371, 0.01759, 0.01124, 0.007186,
                                 0.004592, 0.002528, 0.001392, 0.000766, 0.0003632, 0.0002321, 0.0001484, 0.0000948,
                                 0.00006059, 0.00003872, 0.00002132, 0.00001362, 0.00001011, 0.000006459, 0.000004792,
                                 0.000003063, 0.000001686, 0.000000928, 0.0000005931, 0.0000002812, 0.0000001334,
                                 0.00000006323, 0.00000002998, 0.0000000165, 0.00000001225, 0.00000000783,
                                 0.000000005001, 0.000000003196, 0.000000002043, 0.000000001515, 0.000000000969,
                                 0.000000000619, 0.0000000003407, 0.0000000001616, 0.00000000003632, 0.00000000001722,
                                 0.00000000001101, 0.00000000000817, 0.000000000004495, 0.000000000002475, 0.0])
    # corresponding electron-antineutrino energy for U238 in MeV (one data-point of 17 MeV is added, important for
    # interpolation) (np.array of float):
    energy_fallot_u238 = np.array([0.2391, 0.6014, 0.9638, 1.3261, 1.6159, 1.9783, 2.3406, 2.7029, 2.9928, 3.3551,
                                   3.7174, 4.0797, 4.3696, 4.7319, 5.0217, 5.3841, 5.6739, 6.0362, 6.3261, 6.6159,
                                   6.9783, 7.2681, 7.558, 7.8478, 8.1377, 8.3551, 8.6449, 8.9348, 9.2246, 9.5145,
                                   9.8043, 10.0942, 10.3841, 10.7464, 11.0362, 11.3261, 11.5435, 11.7609, 11.9783,
                                   12.1957, 12.2681, 12.3406, 12.413, 12.7754, 13.0652, 13.3551, 13.7174, 14.0072,
                                   14.2971, 14.587, 14.8768, 15.0217, 15.0942, 15.1667, 15.2391, 15.4565, 15.7464,
                                   16.0362, 16.1812, 17.0])
    # electron-antineutrino flux from figure 1 for U238 in electron-antineutrinos/(MeV*fission) (data-point of
    # 0.0 events/(MeV*fission) for 17 MeV ia added) (np.array of float):
    flux_fallot_u238 = np.array([2.422, 2.812, 2.812, 2.087, 1.797, 1.548, 1.149, 0.989, 0.852, 0.734, 0.5446, 0.4041,
                                 0.3481, 0.2582, 0.165, 0.1422, 0.0909, 0.06741, 0.04308, 0.03196, 0.02371, 0.01515,
                                 0.00969, 0.00619, 0.003956, 0.002178, 0.001392, 0.000889, 0.0005684, 0.0003632,
                                 0.0002695, 0.0001722, 0.0001278, 0.0000817, 0.00006059, 0.00003872, 0.00002475,
                                 0.00001582, 0.00000871, 0.000004792, 0.000002272, 0.000001077, 0.0000005109,
                                 0.0000003791, 0.0000003265, 0.0000002812, 0.0000002087, 0.0000001334, 0.0000000989,
                                 0.00000006323, 0.00000004041, 0.00000002224, 0.00000001055, 0.000000001124,
                                 0.0000000005332, 0.0000000001616, 0.0000000001032, 0.00000000006598, 0.00000000003632,
                                 0.0])
    # corresponding electron-antineutrino energy for Pu239 in MeV (one data-point of 17 MeV is added, important for
    # interpolation) (np.array of float):
    energy_fallot_pu239 = np.array([0.3841, 0.7464, 1.0362, 1.3986, 1.7609, 2.0507, 2.413, 2.7029, 3.0652, 3.4275,
                                    3.7174, 4.0072, 4.3696, 4.6594, 5.0217, 5.3116, 5.6014, 5.8913, 6.2536, 6.5435,
                                    6.8333, 7.1232, 7.413, 7.7029, 7.9203, 8.1377, 8.2101, 8.3551, 8.5725, 8.8623,
                                    9.1522, 9.442, 9.6594, 9.9493, 10.1667, 10.4565, 10.7464, 11.0362, 11.3986,
                                    11.6159, 11.7609, 11.9783, 12.1957, 12.3406, 12.4855, 12.7754, 13.0652, 13.2826,
                                    13.4275, 13.5725, 13.6449, 13.7174, 14.0072, 14.2971, 14.5145, 14.8043, 15.0217,
                                    15.0942, 15.1667, 15.2391, 15.4565, 15.7464, 16.0362, 16.1812, 17.0])
    # electron-antineutrino flux from figure 1 for Pu239 in electron-antineutrinos/(MeV*fission) (data-point of
    # 0.0 events/(MeV*fission) for 17 MeV ia added) (np.array of float):
    flux_fallot_pu239 = np.array([1.696, 2.309, 1.979, 1.453, 1.453, 1.067, 0.784, 0.5758, 0.4935, 0.3625, 0.2662,
                                  0.1955, 0.1436, 0.1055, 0.0775, 0.04877, 0.0307, 0.02255, 0.01656, 0.01216, 0.00766,
                                  0.004819, 0.003034, 0.00191, 0.00103, 0.0005557, 0.0002569, 0.0001386, 0.0000872,
                                  0.00004706, 0.00002962, 0.00001865, 0.00001174, 0.00000739, 0.000003986, 0.000002927,
                                  0.00000215, 0.000001579, 0.000000994, 0.0000005363, 0.0000002893, 0.0000001821,
                                  0.0000000982, 0.00000004542, 0.0000000245, 0.00000001542, 0.00000000971,
                                  0.000000006111, 0.000000002825, 0.000000001524, 0.000000000705, 0.0000000003258,
                                  0.0000000002393, 0.0000000001506, 0.0000000000948, 0.00000000005967,
                                  0.00000000003219, 0.00000000001488, 0.000000000006881, 0.000000000003181,
                                  0.000000000002336, 0.000000000001471, 0.000000000000793, 0.000000000000428, 0.0])
    # corresponding electron-antineutrino energy for Pu241 in MeV (one data-point of 17 MeV is added, important for
    # interpolation) (np.array of float):
    energy_fallot_pu241 = np.array([0.3841, 0.7464, 1.0362, 1.3986, 1.7609, 2.1232, 2.413, 2.7754, 3.0652, 3.4275,
                                    3.7899, 4.0797, 4.442, 4.7319, 5.0217, 5.3841, 5.6739, 5.9638, 6.3261, 6.6159,
                                    6.9058, 7.1957, 7.4855, 7.7029, 7.9928, 8.1377, 8.3551, 8.5, 8.7899, 9.0797,
                                    9.3696, 9.587, 9.8768, 10.0942, 10.4565, 10.7464, 11.0362, 11.3261, 11.6159,
                                    11.7609, 11.9783, 12.1232, 12.1957, 12.3406, 12.413, 12.4855, 12.8478, 13.2101,
                                    13.5, 13.7899, 14.0797, 14.442, 14.6594, 14.9493, 15.0217, 15.0942, 15.1667,
                                    15.3841, 15.6739, 15.9638, 16.1087, 17.0])
    # electron-antineutrino flux from figure 1 for Pu241 in electron-antineutrinos/(MeV*fission) (data-point of
    # 0.0 events/(MeV*fission) for 17 MeV ia added) (np.array of float):
    flux_fallot_pu241 = np.array([1.98, 2.695, 2.31, 1.697, 1.454, 1.247, 0.916, 0.785, 0.5765, 0.4234, 0.3629, 0.2666,
                                  0.1958, 0.1438, 0.1057, 0.06652, 0.04886, 0.03076, 0.0226, 0.0166, 0.01045, 0.006579,
                                  0.004142, 0.002235, 0.001407, 0.000759, 0.0003511, 0.0001895, 0.0001193, 0.0000751,
                                  0.00005516, 0.00002976, 0.00001874, 0.00001011, 0.00000743, 0.000005456, 0.000004007,
                                  0.000002523, 0.000001361, 0.000000735, 0.0000003397, 0.0000001833, 0.0000000848,
                                  0.0000000392, 0.00000002115, 0.00000000978, 0.00000000838, 0.00000000719,
                                  0.000000005278, 0.000000003877, 0.000000002848, 0.000000002092, 0.000000001317,
                                  0.000000000829, 0.0000000001773, 0.00000000001754, 0.000000000003751,
                                  0.000000000002362, 0.000000000001735, 0.000000000000936, 0.0000000000005051, 0.0])
    # linear interpolation of the data of the fluxes with respect to energy (np.array of float):
    # TODO: Do I have to consider the binning in figure 1 of Fallot paper: NO, see Fallot_spectrum_analysis.py)
    u235_fallot = np.interp(energy, energy_fallot_u235, flux_fallot_u235)
    u238_fallot = np.interp(energy, energy_fallot_u238, flux_fallot_u238)
    pu239_fallot = np.interp(energy, energy_fallot_pu239, flux_fallot_pu239)
    pu241_fallot = np.interp(energy, energy_fallot_pu241, flux_fallot_pu241)
    # Fractions (data taken from PhysicsReport_1507.05613, page 136, averaged value of the Daya Bay nuclear cores),
    # (float):
    fraction235_u = 0.577
    fraction239_pu = 0.295
    fraction238_u = 0.076
    fraction241_pu = 0.052
    # add the weighted sum of the terms, electron-antineutrino spectrum
    # in units of electron-antineutrinos/(MeV * fission) (data taken from KRLReactorFLux.cc) (np.array of float):
    spec1_reactor = (u235_fallot * fraction235_u + pu239_fallot * fraction239_pu + u238_fallot * fraction238_u +
                     pu241_fallot * fraction241_pu)
    # There are 3.125*10**19 fissions/GW/second, spectrum in units of electron-antineutrino/(MeV * GW * s)
    # (np.array of float):
    spec2_reactor = spec1_reactor * 3.125 * 10 ** 19
    # There are about 3.156*10**7 seconds in a year, spectrum in units of electron-antineutrino/(MeV * GW * year)
    # (np.array of float):
    spec3_reactor = spec2_reactor * 3.156 * 10 ** 7
    # electron-antineutrino flux in units of electron-antineutrino/(MeV * year) (np.array of float):
    flux_reactor = spec3_reactor * power_th

    """ Consider neutrino oscillation for NORMAL HIERARCHY from NuOscillation.cc: """
    # Oscillation parameters:
    # distance reactor to detector in meter (float):
    l_m = 5.25 * 10 ** 4
    # distance reactor to detector in centimeter (float):
    l_cm = l_m * 100
    # mixing angles from PDG 2016 (same in NuOscillation.cc) (float):
    sin2_th12 = 0.297
    sin2_th13 = 0.0214
    # mass squared differences in eV**2 from PDG 2016 (same in NuOscillation.cc) (float):
    dm2_21 = 7.37 * 10 ** (-5)
    dm2_31 = 2.50 * 10 ** (-3) + dm2_21 / 2
    dm2_32 = 2.50 * 10 ** (-3) - dm2_21 / 2
    # calculate the other parameters (float):
    cos2_th12 = 1. - sin2_th12
    sin2_2th12 = 4. * sin2_th12 * cos2_th12
    cos2_th13 = 1. - sin2_th13
    sin2_2th13 = 4. * sin2_th13 * cos2_th13
    cos4_th13 = cos2_th13 ** 2
    # With these parameters calculate survival probability of electron-antineutrinos (np.array of float):
    p21 = sin2_2th12 * cos4_th13 * np.sin(1.267 * dm2_21 * l_m / energy) ** 2
    p31 = sin2_2th13 * cos2_th12 * np.sin(1.267 * dm2_31 * l_m / energy) ** 2
    p32 = sin2_2th13 * sin2_th12 * np.sin(1.267 * dm2_32 * l_m / energy) ** 2
    # Survival probability of electron-antineutrinos for Normal Hierarchy (np.array of float):
    prob_oscillation_nh = 1. - p21 - p31 - p32

    """ Theoretical reactor electron-antineutrino spectrum in JUNO with oscillation """
    # Theoretical spectrum in JUNO for normal hierarchy with oscillation in units of electron-antineutrinos/MeV
    # in "t_years" years (np.array of float):
    theospectrum_reactor = (1 / (4 * np.pi * l_cm ** 2) * flux_reactor * crosssection * detection_efficiency *
                            n_target * tyears * prob_oscillation_nh)

    # number of neutrinos from reactor background in JUNO detector after "time":
    n_neutrino_reactor = np.trapz(theospectrum_reactor, energy)

    """ Visible spectrum of the reactor electron-antineutrino background, theoretical spectrum is convolved 
        with Faltung (gaussian distribution): """
    # generate n_neutrino_reactor_gen - many antineutrinos with energies e_neutrino_reactor in MeV
    # from the theoretical spectrum (np.array of float64):
    e_neutrino_reactor, n_neutrino_reactor_gen = energy_neutrino(energy, theospectrum_reactor, num)
    # array of visible energies of reactor neutrinos in MeV (np.array of float):
    e_visible_reactor = visible_spectrum(e_neutrino_reactor, mass_proton, mass_neutron, mass_positron, num_vis)

    return (theospectrum_reactor, e_visible_reactor, n_neutrino_reactor, n_neutrino_reactor_gen, power_th,
            fraction235_u, fraction238_u, fraction239_pu, fraction241_pu, l_m)


""" Often used values of functions: """
# IBD cross-section for the DM signal in cm**2, must be calculated only for energy = mass_DM (float):
sigma_IBD_signal = sigma_ibd(mass_DM, DELTA, MASS_POSITRON)
# IBD cross-section for the backgrounds in cm**2, must be calculated for the whole energy range E1 (np.array of floats):
sigma_IBD = sigma_ibd(E1, DELTA, MASS_POSITRON)


""" differ between generating a dataset or a MC simulation: """
if number == 1:
    # if number = 1, a dataset is generated:
    simu_type = "dataset"
    print("Simulation of dataset is started...")

    """ SIMULATE THE SIGNAL FROM NEUTRINOS FROM DM ANNIHILATION IN THE MILKY WAY: 
        When DM signal should be simulated, DM_SIGNAL must be True """
    if DM_SIGNAL:
        print("... simulation of DM annihilation signal...")

        # iteration of index has to go to number_dataset_stop + 1, because np.arange(start, stop)
        # generate values in the half-open interval [start, stop) -> stop would be excluded
        for index in np.arange(number_dataset_start, number_dataset_stop + 1):
            TheoSpectrum_signal, E_visible_signal, N_neutrino_signal, N_neutrino_signal_gen, sigma_Anni, Flux_signal = \
                darkmatter_signal(E1, mass_DM, sigma_IBD_signal, N_target, time, detection_eff,
                                  number, MASS_PROTON, MASS_NEUTRON, MASS_POSITRON, number_vis)

            if SAVE_DATA:
                # save E_visible_signal to txt-data-file and information about simulation in txt-info-file:
                print("... save data to file...")
                np.savetxt('gen_simu_spectrum_v1\Dataset\signal_Dataset_DMmass{0:.0f}_data_{1:d}.txt'
                           .format(mass_DM, index), E_visible_signal, fmt='%1.4e',
                           header='Visible energy in MeV of DM annihilation signal '
                                  '(Dataset generated with gen_simu_spectrum_v1.py, {0}):'
                                  '\nDM mass = {1:.0f}MeV, Number of neutrinos = {2:.6f}, number of neutrino events '
                                  'generated = {3}, DM annihilation cross-section = {4:1.4e} cm**3/s, '
                                  'number = {5:d}:'
                           .format(now, mass_DM, N_neutrino_signal, N_neutrino_signal_gen, sigma_Anni, number))
                np.savetxt('gen_simu_spectrum_v1\Dataset\signal_Dataset_DMmass{0:.0f}_info_{1:d}.txt'
                           .format(mass_DM, index),
                           np.array([number, number_vis, E1[0], E1[-1], interval, t_years, N_target, detection_eff,
                                     mass_DM, N_neutrino_signal, N_neutrino_signal_gen, sigma_Anni, Flux_signal]),
                           fmt='%1.9e',
                           header='Information to simulation signal_Dataset_DMmass{0:.0f}_data_{1:d}.txt:\n'
                                  'values below: number, number_vis, E1[0] in MeV, E1[-1] in MeV, interval E1 in MeV,'
                                  ' exposure time t_years in years, number of free protons N_target,\n'
                                  'IBD detection efficiency, DM mass in MeV, Number of neutrinos, Number of neutrino '
                                  'events generated, DM annihilation cross-section in cm**3/s, '
                                  '\nnu_e_bar flux at Earth in 1/(MeV*s*cm**2):'
                           .format(mass_DM, index))

    """ SIMULATE THE DSNB ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
        When DSNB background should be simulated, DSNB_BACKGROUND must be True """
    if DSNB_BACKGROUND:
        print("... simulation of DSNB background...")

        # iteration of index has to go to number_dataset_stop + 1, because np.arange(start, stop)
        # generate values in the half-open interval [start, stop) -> stop would be excluded
        for index in np.arange(number_dataset_start, number_dataset_stop + 1):
            (TheoSpectrum_DSNB, E_visible_DSNB, N_neutrino_DSNB, N_neutrino_DSNB_gen, E_mean_NuEbar, beta_NuEbar,
             E_mean_NuXbar, beta_NuXbar, f_Star) = \
                dsnb_background(E1, sigma_IBD, N_target, time, detection_eff, C_LIGHT, number,
                                MASS_PROTON, MASS_NEUTRON, MASS_POSITRON, number_vis)

            if SAVE_DATA:
                # save E_visible_DSNB to txt-data-file and information about simulation in txt-info-file:
                print("... save data to file...")
                np.savetxt('gen_simu_spectrum_v1\Dataset\DSNB_Dataset_EmeanNuXbar{0:.0f}_data_{1:d}.txt'
                           .format(E_mean_NuXbar, index), E_visible_DSNB, fmt='%1.4e',
                           header='Visible energy in MeV of the DSNB electron-antineutrino background '
                                  '(Dataset generated with gen_simu_spectrum_v1.py, {0}):'
                                  '\nNumber of neutrinos = {1:.6f}, Number of neutrino events generated = {2},'
                                  ' Mean energy of nu_Ebar = {3:.2f} MeV, beta nu_Ebar = {4:.2f}, '
                                  '\nmean energy of nu_Xbar = {5:.2f}, pinching factor of nu_Xbar = {6:.2f}, '
                                  'number = {7:d}:'
                           .format(now, N_neutrino_DSNB, N_neutrino_DSNB_gen, E_mean_NuEbar, beta_NuEbar,
                                   E_mean_NuXbar, beta_NuXbar, number))
                np.savetxt('gen_simu_spectrum_v1\Dataset\DSNB_Dataset_EmeanNuXbar{0:.0f}_info_{1:d}.txt'
                           .format(E_mean_NuXbar, index),
                           np.array([number, number_vis, E1[0], E1[-1], interval, t_years, N_target, detection_eff,
                                     N_neutrino_DSNB, N_neutrino_DSNB_gen, E_mean_NuEbar, beta_NuEbar, E_mean_NuXbar,
                                     beta_NuXbar, f_Star]),
                           fmt='%1.9e',
                           header='Information to simulation DSNB_Dataset_EmeanNuXbar{0:.0f}_data_{1:d}.txt:\n'
                                  'values below: number, number_vis, E1[0] in MeV, E1[-1] in MeV, interval E1 in MeV,'
                                  ' exposure time t_years in years, number of free protons N_target,\n'
                                  'IBD detection efficiency, Number of neutrinos, Number of neutrino events generated,'
                                  ' mean energy of nu_Ebar in MeV, pinching factor for nu_Ebar, '
                                  '\nmean energy of nu_Xbar in MeV, pinching factor for nu_Xbar, '
                                  'correction factor of SFR:'
                           .format(E_mean_NuXbar, index))

    """ SIMULATE THE ATMOSPHERIC CHARGED CURRENT ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
        When atmospheric CC background should be simulated, CCATMOSPHERIC_BACKGROUND must be True """
    if CCATMOSPHERIC_BACKGROUND:
        print("... simulation of atmospheric CC background...")

        # iteration of index has to go to number_dataset_stop + 1, because np.arange(start, stop)
        # generate values in the half-open interval [start, stop) -> stop would be excluded
        for index in np.arange(number_dataset_start, number_dataset_stop + 1):
            (TheoSpectrum_CCatmospheric, E_visible_CCatmospheric, N_neutrino_CCatmospheric,
             N_neutrino_CCatmospheric_gen, Oscillation, Prob_e_to_e, Prob_mu_to_e) = \
                ccatmospheric_background(E1, sigma_IBD, N_target, time, detection_eff,
                                         number, MASS_PROTON, MASS_NEUTRON, MASS_POSITRON, number_vis)

            if SAVE_DATA:
                # save E_visible_CCatmospheric to txt-data-file and information about simulation in txt-info-file:
                print("... save data to file...")
                np.savetxt('gen_simu_spectrum_v1\Dataset\CCatmo_Dataset_Osc{0:d}_data_{1:d}.txt'
                           .format(Oscillation, index), E_visible_CCatmospheric, fmt='%1.4e',
                           header='Visible energy in MeV of the CC atmospheric electron-antineutrino background '
                                  '(Dataset generated with gen_simu_spectrum_v1.py, {0}):'
                                  '\nNumber of neutrinos = {1:.6f}, Number of neutrino events generated = {2}, '
                                  'Is oscillation considered (1=yes, 0=no)? {3:d}, '
                                  '\nsurvival probability of nu_Ebar = {4:.2f}, '
                                  'oscillation prob. nu_Mubar to nu_Ebar = {5:.2f}, number = {6:d}:'
                           .format(now, N_neutrino_CCatmospheric, N_neutrino_CCatmospheric_gen, Oscillation,
                                   Prob_e_to_e, Prob_mu_to_e, number))
                np.savetxt('gen_simu_spectrum_v1\Dataset\CCatmo_Dataset_Osc{0:d}_info_{1:d}.txt'
                           .format(Oscillation, index),
                           np.array([number, number_vis, E1[0], E1[-1], interval, t_years, N_target, detection_eff,
                                     N_neutrino_CCatmospheric, N_neutrino_CCatmospheric_gen, Oscillation,
                                     Prob_e_to_e, Prob_mu_to_e]),
                           fmt='%1.9e',
                           header='Information to simulation CCatmo_Dataset_Osc{0:d}_data_{1:d}.txt:\n'
                                  'values below: number, number_vis, E1[0] in MeV, E1[-1] in MeV, interval E1 in MeV,'
                                  ' exposure time t_years in years, number of free protons N_target,\n'
                                  'IBD detection efficiency, Number of neutrinos, Number of neutrino events generated, '
                                  'Is oscillation considered (1=yes, 0=no)?, '
                                  '\nsurvival probability of electron-antineutrinos, oscillation '
                                  'probability muon-antineutrino to electron-antineutrino:'
                           .format(Oscillation, index))

    """ SIMULATE THE REACTOR ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
        When reactor antineutrino background should be simulated, REACTOR_BACKGROUND must be True """
    if REACTOR_BACKGROUND:
        print("... simulation of reactor neutrino background...")

        # iteration of index has to go to number_dataset_stop + 1, because np.arange(start, stop)
        # generate values in the half-open interval [start, stop) -> stop would be excluded
        for index in np.arange(number_dataset_start, number_dataset_stop + 1):
            (TheoSpectrum_reactor, E_visible_reactor, N_neutrino_reactor, N_neutrino_reactor_gen, power_thermal,
             Fraction_U235, Fraction_U238, Fraction_Pu239, Fraction_Pu241, L_meter) = \
                reactor_background(E1, sigma_IBD, N_target, t_years, detection_eff, number, MASS_PROTON,
                                   MASS_NEUTRON, MASS_POSITRON, number_vis)

            if SAVE_DATA:
                # save E_visible_reactor to txt-data-file and information about simulation in txt-info-file:
                print("... save data to file...")
                np.savetxt('gen_simu_spectrum_v1\Dataset\Reactor_Dataset_NH_power{0:.0f}_data_{1:d}.txt'
                           .format(power_thermal, index), E_visible_reactor, fmt='%1.4e',
                           header='Visible energy in MeV of the reactor electron-antineutrino background '
                                  '(Dataset generated with gen_simu_spectrum_v1.py, {0}):'
                                  '\nNumber of neutrinos = {1:.6f}, Number of neutrino events generated = {2}, '
                                  'normal hierarchy considered, thermal power = {3:.2f} GW, number = {4:d}:'
                           .format(now, N_neutrino_reactor, N_neutrino_reactor_gen, power_thermal, number))
                np.savetxt('gen_simu_spectrum_v1\Dataset\Reactor_Dataset_NH_power{0:.0f}_info_{1:d}.txt'
                           .format(power_thermal, index),
                           np.array([number, number_vis, E1[0], E1[-1], interval, t_years, N_target, detection_eff,
                                     N_neutrino_reactor, N_neutrino_reactor_gen, power_thermal, Fraction_U235,
                                     Fraction_U238, Fraction_Pu239, Fraction_Pu241, L_meter]),
                           fmt='%1.9e',
                           header='Information to simulation Reactor_Dataset_NH_power{0:.0f}_data_{1:d}.txt:\n'
                                  'values below: number, number_vis, E1[0] in MeV, E1[-1] in MeV, interval E1 in MeV,'
                                  ' exposure time t_years in years, number of free protons N_target,\n'
                                  'IBD detection efficiency, Number of neutrinos, Number of neutrino events generated, '
                                  'thermal power in GW, \nfission fraction of U235, U238, Pu239, Pu241, '
                                  'distance reactor to detector in meter:'
                           .format(power_thermal, index))

else:
    # if number > 1, a MC simulation is generated:
    simu_type = "MCsimu"
    print("MC simulation is started...")

    """ SIMULATE THE SIGNAL FROM NEUTRINOS FROM DM ANNIHILATION IN THE MILKY WAY: 
            When DM signal should be simulated, DM_SIGNAL must be True """
    if DM_SIGNAL:
        print("... simulation of DM annihilation signal...")

        TheoSpectrum_signal, E_visible_signal, N_neutrino_signal, N_neutrino_signal_gen, sigma_Anni, Flux_signal = \
            darkmatter_signal(E1, mass_DM, sigma_IBD_signal, N_target, time, detection_eff,
                              number, MASS_PROTON, MASS_NEUTRON, MASS_POSITRON, number_vis)

        if SAVE_DATA:
            # save E_visible_signal to txt-data-file and information about simulation in txt-info-file:
            print("... save data to file...")
            np.savetxt('gen_simu_spectrum_v1\MCsimu\signal_MCsimu_DMmass{0:.0f}_data_{1:d}.txt'
                       .format(mass_DM, number_MCsimu), E_visible_signal, fmt='%1.4e',
                       header='Visible energy in MeV of DM annihilation signal '
                              '(MCsimu generated with gen_simu_spectrum_v1.py, {0}):'
                              '\nDM mass = {1:.0f}MeV, Number of neutrinos = {2:.6f}, Number of neutrino events '
                              'generated = {3}, \nDM annihilation cross-section = {4:1.4e} cm**3/s, number = {5:d}:'
                       .format(now, mass_DM, N_neutrino_signal, N_neutrino_signal_gen, sigma_Anni, number))
            np.savetxt('gen_simu_spectrum_v1\MCsimu\signal_MCsimu_DMmass{0:.0f}_info_{1:d}.txt'
                       .format(mass_DM, number_MCsimu),
                       np.array([number, number_vis, E1[0], E1[-1], interval, t_years, N_target, detection_eff,
                                 mass_DM, N_neutrino_signal, N_neutrino_signal_gen, sigma_Anni, Flux_signal]),
                       fmt='%1.9e',
                       header='Information to simulation signal_MCsimu_DMmass{0:.0f}_data_{1:d}.txt:\n'
                              'values below: number, number_vis, E1[0] in MeV, E1[-1] in MeV, interval E1 in MeV,'
                              ' exposure time t_years in years, number of free protons N_target,\n'
                              'IBD detection efficiency, DM mass in MeV, Number of neutrinos, Number of neutrino '
                              'events generated, DM annihilation cross-section in cm**3/s, '
                              '\nnu_e_bar flux at Earth in 1/(MeV*s*cm**2):'
                       .format(mass_DM, number_MCsimu))

    """ SIMULATE THE DSNB ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
        When DSNB background should be simulated, DSNB_BACKGROUND must be True """
    if DSNB_BACKGROUND:
        print("... simulation of DSNB background...")

        (TheoSpectrum_DSNB, E_visible_DSNB, N_neutrino_DSNB, N_neutrino_DSNB_gen, E_mean_NuEbar, beta_NuEbar,
         E_mean_NuXbar, beta_NuXbar, f_Star) = \
            dsnb_background(E1, sigma_IBD, N_target, time, detection_eff, C_LIGHT, number,
                            MASS_PROTON, MASS_NEUTRON, MASS_POSITRON, number_vis)

        if SAVE_DATA:
            # save E_visible_DSNB to txt-data-file and information about simulation in txt-info-file:
            print("... save data to file...")
            np.savetxt('gen_simu_spectrum_v1\MCsimu\DSNB_MCsimu_EmeanNuXbar{0:.0f}_data_{1:d}.txt'
                       .format(E_mean_NuXbar, number_MCsimu), E_visible_DSNB, fmt='%1.4e',
                       header='Visible energy in MeV of DSNB electron-antineutrino background '
                              '(MCsimu generated with gen_simu_spectrum_v1.py, {0}):'
                              '\nNumber of neutrinos = {1:.6f}, Number of neutrino events generated = {2},'
                              ' number = {3:d}, mean energy nu_Ebar = {4:.2f} MeV, pinching factor nu_Ebar = {5:.2f}, '
                              '\nmean energy nu_Xbar = {6:.2f} MeV, pinching factor nu_Xbar = {7:.2f}:'
                       .format(now, N_neutrino_DSNB, N_neutrino_DSNB_gen, number, E_mean_NuEbar, beta_NuEbar,
                               E_mean_NuXbar, beta_NuXbar))
            np.savetxt('gen_simu_spectrum_v1\MCsimu\DSNB_MCsimu_EmeanNuXbar{0:.0f}_info_{1:d}.txt'
                       .format(E_mean_NuXbar, number_MCsimu),
                       np.array([number, number_vis, E1[0], E1[-1], interval, t_years, N_target, detection_eff,
                                 N_neutrino_DSNB, N_neutrino_DSNB_gen, E_mean_NuEbar, beta_NuEbar, E_mean_NuXbar,
                                 beta_NuXbar, f_Star]),
                       fmt='%1.9e',
                       header='Information to simulation DSNB_MCsimu_EmeanNuXbar{0:.0f}_data_{1:d}.txt:\n'
                              'values below: number, number_vis, E1[0] in MeV, E1[-1] in MeV, interval E1 in MeV,'
                              'exposure time t_years in years, number of free protons N_target,\n'
                              'IBD detection efficiency, Number of neutrinos, Number of neutrino events generated, '
                              'mean energy of nu_Ebar in MeV, pinching factor for nu_Ebar, '
                              '\nmean energy of nu_Xbar in MeV, pinching factor for nu_Xbar, correction factor of SFR:'
                       .format(E_mean_NuXbar, number_MCsimu))

    """ SIMULATE THE ATMOSPHERIC CHARGED CURRENT ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
        When atmospheric CC background should be simulated, CCATMOSPHERIC_BACKGROUND must be True """
    if CCATMOSPHERIC_BACKGROUND:
        print("... simulation of atmospheric CC background...")

        (TheoSpectrum_CCatmospheric, E_visible_CCatmospheric, N_neutrino_CCatmospheric, N_neutrino_CCatmospheric_gen,
         Oscillation, Prob_e_to_e, Prob_mu_to_e) = \
            ccatmospheric_background(E1, sigma_IBD, N_target, time, detection_eff,
                                     number, MASS_PROTON, MASS_NEUTRON, MASS_POSITRON, number_vis)

        if SAVE_DATA:
            # save E_visible_CCatmospheric to txt-data-file and information about simulation in txt-info-file:
            print("... save data to file...")
            np.savetxt('gen_simu_spectrum_v1\MCsimu\CCatmo_MCsimu_Osc{0:d}_data_{1:d}.txt'
                       .format(Oscillation, number_MCsimu), E_visible_CCatmospheric, fmt='%1.4e',
                       header='Visible energy in MeV of CC atmospheric electron-antineutrino background '
                              '(MCsimu generated with gen_simu_spectrum_v1.py, {0}):'
                              '\nNumber of neutrinos = {1:.6f}, Number of neutrino events generated = {2}, '
                              'number = {3:d}, Is oscillation considered (1=yes, 0=no)? {4:d}, '
                              '\nsurvival probability of nu_Ebar = {5:.2f},'
                              'oscillation prob. nu_Mubar to nu_Ebar = {6:.2f}:'
                       .format(now, N_neutrino_CCatmospheric, N_neutrino_CCatmospheric_gen, number,
                               Oscillation, Prob_e_to_e, Prob_mu_to_e))
            np.savetxt('gen_simu_spectrum_v1\MCsimu\CCatmo_MCsimu_Osc{0:d}_info_{1:d}.txt'
                       .format(Oscillation, number_MCsimu),
                       np.array([number, number_vis, E1[0], E1[-1], interval, t_years, N_target, detection_eff,
                                 N_neutrino_CCatmospheric, N_neutrino_CCatmospheric_gen, Oscillation,
                                 Prob_e_to_e, Prob_mu_to_e]),
                       fmt='%1.9e',
                       header='Information to simulation CCatmo_MCsimu_Osc{0:d}_data_{1:d}.txt:\n'
                              'values below: number, number_vis, E1[0] in MeV, E1[-1] in MeV, interval E1 in MeV,'
                              'exposure time t_years in years, number of free protons N_target,\n'
                              'IBD detection efficiency, Number of neutrinos, Number of neutrino events generated, '
                              'Is oscillation considered (1=yes, 0=no)?, \nsurvival probability of nu_Ebar,'
                              'oscillation prob. nu_Mubar to nu_Ebar:'
                       .format(Oscillation, number_MCsimu))

    """ SIMULATE THE REACTOR ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
        When reactor antineutrino background should be simulated, REACTOR_BACKGROUND must be True """
    if REACTOR_BACKGROUND:
        print("... simulation of reactor neutrino background...")

        (TheoSpectrum_reactor, E_visible_reactor, N_neutrino_reactor, N_neutrino_reactor_gen, power_thermal,
         Fraction_U235, Fraction_U238, Fraction_Pu239, Fraction_Pu241, L_meter) = \
            reactor_background(E1, sigma_IBD, N_target, t_years, detection_eff,
                               number, MASS_PROTON, MASS_NEUTRON, MASS_POSITRON, number_vis)

        if SAVE_DATA:
            # save E_visible_reactor to txt-data-file and information about simulation in txt-info-file:
            print("... save data to file...")
            np.savetxt('gen_simu_spectrum_v1\MCsimu\Reactor_MCsimu_NH_power{0:.0f}_data_{1:d}.txt'
                       .format(power_thermal, number_MCsimu), E_visible_reactor, fmt='%1.4e',
                       header='Visible energy in MeV of reactor electron-antineutrino background '
                              '(MCsimu generated with gen_simu_spectrum_v1.py, {0}):'
                              '\nNumber of neutrinos = {1:.6f}, Number of neutrino events generated = {2}, '
                              'normal hierarchy considered, thermal power = {3:.2f} GW, number = {4:d}:'
                              .format(now, N_neutrino_reactor, N_neutrino_reactor_gen, power_thermal, number))
            np.savetxt('gen_simu_spectrum_v1\MCsimu\Reactor_MCsimu_NH_power{0:.0f}_info_{1:d}.txt'
                       .format(power_thermal, number_MCsimu),
                       np.array([number, number_vis, E1[0], E1[-1], interval, t_years, N_target, detection_eff,
                                 N_neutrino_reactor, N_neutrino_reactor_gen, power_thermal,
                                 Fraction_U235, Fraction_U238, Fraction_Pu239, Fraction_Pu241, L_meter]),
                       fmt='%1.9e',
                       header='Information to simulation Reactor_MCsimu_NH_power{0:.0f}_data_{1:d}.txt:\n'
                              'values below: number, number_vis, E1[0] in MeV, E1[-1] in MeV, interval E1 in MeV,'
                              'exposure time t_years in years, number of free protons N_target,\n'
                              'IBD detection efficiency, Number of neutrinos, Number of neutrino events generated, '
                              'thermal power in GW, fission fraction of U235, U238, Pu239, Pu241,'
                              '\ndistance reactor to detector in meter:'
                       .format(power_thermal, number_MCsimu))

""" SIMULATE THE NEUTRAL CURRENT ATMOSPHERIC NEUTRINOS BACKGROUND IN JUNO: """
# TODO: Neutral Current atmospheric background has to be added


""" Display the simulated spectra: """
if DISPLAY_SPECTRA:
    print("... display plots...")
    # Display the theoretical spectra with the settings below:
    h1 = pyplot.figure(1)
    if DM_SIGNAL:
        pyplot.plot(E1, TheoSpectrum_signal, 'r--', label='signal from DM annihilation for '
                                                          '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni))
    if DSNB_BACKGROUND:
        pyplot.plot(E1, TheoSpectrum_DSNB, 'b--', label='DSNB background')
    if CCATMOSPHERIC_BACKGROUND:
        pyplot.plot(E1, TheoSpectrum_CCatmospheric, 'g+', label='atmospheric CC background without oscillation')
    if REACTOR_BACKGROUND:
        pyplot.plot(E1, TheoSpectrum_reactor, 'c--', label='reactor electron-antineutrino background')
    pyplot.xlim(E1[0], E1[-1])
    pyplot.ylim(ymin=0)
    pyplot.xlabel("Electron-antineutrino energy in MeV")
    pyplot.ylabel("Theoretical spectrum dN/dE in 1/MeV")
    pyplot.title(
        "Theoretical electron-antineutrino spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
        .format(t_years, mass_DM))
    pyplot.legend()

    # Display the expected spectra with the settings below:
    h2 = pyplot.figure(2)
    # energy corresponding to the visible energy in MeV, E2 defines the bins in pyplot.hist() (np.array of float64):
    E2 = np.arange(7, 130, 0.1)
    if DM_SIGNAL:
        n_signal, bins1, patches1 = pyplot.hist(E_visible_signal, bins=E2, histtype='step', color='r',
                                                label='signal from DM annihilation for '
                                                      '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni))
    if DSNB_BACKGROUND:
        n_DSNB, bins2, patches2 = pyplot.hist(E_visible_DSNB, bins=E2, histtype='step', color='b',
                                              label='DSNB background')
    if CCATMOSPHERIC_BACKGROUND:
        n_atmospheric, bins4, patches4 = pyplot.hist(E_visible_CCatmospheric, bins=E2, histtype='step', color='g',
                                                     label='atmospheric CC background without oscillation')
    if REACTOR_BACKGROUND:
        n_reactor, bins5, patches5 = pyplot.hist(E_visible_reactor, bins=E2, histtype='step', color='c',
                                                 label='reactor electron-antineutrino background')
    pyplot.xlim(E2[0], E2[-1])
    pyplot.ylim(ymin=0)
    pyplot.xlabel("Visible energy in MeV")
    pyplot.ylabel("Expected spectrum dN/dE in 1/MeV * {0:d}".format(number * number_vis))
    pyplot.title("Expected spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
                 .format(t_years, mass_DM))
    pyplot.legend()

    pyplot.show()
