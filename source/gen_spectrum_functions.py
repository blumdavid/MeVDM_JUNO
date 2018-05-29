""" Script defines the functions, which are used in other scripts: """

# import of the necessary packages:
import numpy as np
from math import gamma


""" Define several functions used in the script: """


def sigma_ibd(energy, delta, mass_positron):
    """ IBD cross-section in cm**2 for neutrinos with E=energy, equation (25) from paper 0302005_IBDcrosssection:
        simple approximation
        INFO-me: agrees with full result of paper within few per-mille for neutrino energies <= 300 MeV

        :param energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float OR float)
        :param delta: difference mass_neutron minus mass_proton in MeV (float)
        :param mass_positron: mass of the positron in MeV (float)

        :return IBD cross-section in cm**2 (float or np.array of float)
        """
    # positron energy defined as energy - delta in MeV (np.array of float64 or float):
    energy_positron = energy - delta
    # positron momentum defined as sqrt(energy_positron**2-mass_positron**2) in MeV (np.array of float64 or float):
    momentum_positron = np.sqrt(energy_positron ** 2 - mass_positron ** 2)
    # IBD cross-section in cm**2 (np.array of float64 or float):
    sigma = (10 ** (-43) * momentum_positron * energy_positron *
             energy ** (-0.07056 + 0.02018 * np.log(energy) - 0.001953 * np.log(energy) ** 3))
    return sigma


def correlation_vis_neutrino(energy, mass_proton, mass_neutron, mass_positron):
    """ correlation between E_visible and e_neutrino for Inverse Beta Decay from paper of Strumia/Vissani
        'Precise quasielastic neutrino/nucleon cross section', 0302055_IBDcrosssection.pdf:
        (detailed information in folder 'correlation_vis_neutrino' using correlation_vis_neutrino.py)

        The average lepton energy E_positron_Strumia is approximated by the equation 16 in the paper
        INFO-me: at better than 1 percent below energies of around 100 MeV
        This is a much better approximation than the usual formula E_positron_Strumia = E_neutrino - DELTA.
        It permits ('erlaubt') to relate E_positron_Strumia with E_neutrino incorporating ('unter Berücksichtigung')
        a large part of the effect due to the recoil of the nucleon.
        INFO-me: a large part of the effect due to the recoil of the nucleon is incorporated
        TODO-me: is quenching of the nucleon (proton) considered???

        INFO-me: only the average positron energy corresponding to the incoming neutrino energy is considered.
        TODO-me: the angle between neutrino and positron is not considered specific, but only on average -> Uncertainty!

        :param energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        :param mass_proton: mass of the proton in MeV (float)
        :param mass_neutron: mass of the neutron in MeV (float)
        :param mass_positron: mass of the positron in MeV (float)

        :return: prompt visible energy in the detector in MeV (float or np.array of float)
        """
    # Correlation between visible and neutrino energy in MeV:
    # first define some useful terms for the calculation (d=delta from page 3 equ. 12, s from page 3):
    d = (mass_neutron**2 - mass_proton**2 - mass_positron**2) / (2 * mass_proton)
    s = 2 * mass_proton * energy + mass_proton**2
    # neutrino energy in center of mass (cm) frame in MeV, page 4, equ. 13:
    e_neutrino_cm = (s - mass_proton**2) / (2 * np.sqrt(s))
    # positron energy in CM frame in MeV, page 4, equ. 13:
    e_positron_cm = (s - mass_neutron**2 + mass_positron**2) / (2 * np.sqrt(s))
    # Average lepton energy in MeV, which can be approximated (at better than 1 percent below ~ 100 MeV) by
    # (page 5, equ. 16):
    e_positron = energy - d - e_neutrino_cm * e_positron_cm / mass_proton
    # prompt visible energy in the detector in MeV:
    corr_vis_neutrino = e_positron + mass_positron
    return corr_vis_neutrino


def energy_resolution(corr_vis_neutrino):
    """ 'energy resolution' of the JUNO detector:
        INFO-me: in detail: energy_resolution returns the width sigma of the gaussian distribution.
        The real energy of the neutrino is smeared by a gaussian distribution characterized by sigma.

        :param corr_vis_neutrino: correlation between visible and neutrino energy,
        characterizes the visible energy in MeV (float or np.array of float)

        :return sigma/width of the gaussian distribution in no units (float or np.array of float)
        """
    # parameters to describe the energy resolution in percent (maximum values of table 13-4, page 196, PhysicsReport):
    # TODO-me: p0, p1 and p2 are determined by the maximum values of the parameters from table 13-4
    # p0: is the leading term dominated by the photon statistics
    p0 = 2.8
    # p1 and p2 come from detector effects such as PMT dark noise, variation of the PMT QE and the
    # reconstructed vertex smearing:
    p1 = 0.26
    p2 = 0.9
    # energy resolution defined as sigma/E_visible in percent, 3-parameter function (page 195, PhysicsReport) (float):
    energy_res = np.sqrt((p0 / np.sqrt(corr_vis_neutrino))**2 + p1**2 + (p2 / corr_vis_neutrino)**2)
    # sigma or width of the gaussian distribution in percent (float):
    sigma_resolution = energy_res * corr_vis_neutrino
    # sigma converted from percent to 'no unit' (float):
    sigma_resolution = sigma_resolution / 100
    return sigma_resolution


def convolution(energy_neutrino, energy_measured, binning_energy_measured, theo_spectrum,
                mass_proton, mass_neutron, mass_positron):
    """
    Function to convolve the theoretical spectrum with the gaussian distribution, which is defined by the correlation
    of measured energy and neutrino energy from IBD and the energy resolution of the detector.
    1. go through every entry of energy_measured
    2. then go through every entry of energy_neutrino
    3. integrate the product of the theo. spectrum (of energy_neutrino[index1]) and gauss (as function of e and
    depending on corr_vis_neutrino(energy_neutrino[index1]) and sigma(corr_vis_neutrino)) over e from
    energy_measured[index2]-binning_energy_measured/2 to energy_measured[index2]+binning_energy_measured/2
    4. do this for every entry in energy_neutrino
    5. you get then the values of s_measured for one entry of energy_measured as function of energy_neutrino
    6. then integrate this value over energy_neutrino and divide it by the bin-width of energy-measured to get the value
    of the spectrum for one entry in energy_measured
    7. do this for every entry in energy_measured

    :param energy_neutrino: energy corresponding to the neutrino energy in MeV (np.array of float)
    :param energy_measured: measured energy in the detector in MeV (is equal to the visible energy) (np.array of float)
    :param binning_energy_measured: bin-width of the measured energy in MeV (np.array of float)
    :param theo_spectrum: theoretical spectrum as function of the neutrino energy of the signal or background in 1/MeV
    (np.array of float)
    :param mass_proton: mass of the proton in MeV (float)
    :param mass_neutron: mass of the neutron in MeV (float)
    :param mass_positron: mass of the positron in MeV (float)

    :return: spectrum_measured in 1/MeV (spectrum of the signal or background taking into account the detector
    properties (energy resolution, correlation of visible and neutrino energy of IBD)) (np.array of float)
    """
    # Preallocate the measured spectrum array (empty np.array):
    spectrum_measured = np.array([])
    # First calculate the function correlation_vis_neutrino and energy_resolution for the given
    # neutrino energy energy_neutrino (np.arrays of float):
    corr_vis_neutrino = correlation_vis_neutrino(energy_neutrino, mass_proton, mass_neutron, mass_positron)
    sigma = energy_resolution(corr_vis_neutrino)

    # loop over entries in energy_measured ('for every entry in energy_measured'):
    for index2 in np.arange(len(energy_measured)):
        # preallocate array (in the notes defined as S_m^(E_m=1) as function of e_neutrino:
        s_measured_em1 = np.array([])
        # for 1 entry in energy_measured, loop over entries in energy_neutrino ('for every entry in e_neutrino'):
        for index1 in np.arange(len(energy_neutrino)):
            # define energy e in MeV, e is in the range of energy_measured[index2]-binning_energy_measured/2 and
            # energy_measured[index2]+binning_energy_measured/2 (np.array of float):
            # IMPORTANT: use np.linspace and NOT np.arange, because "When using a non-integer step in np.arange,
            # such as 0.1, the results will often not be consistent". This leads to errors in the convolution.
            e = np.linspace(energy_measured[index2] - binning_energy_measured/2,
                            energy_measured[index2] + binning_energy_measured/2,
                            101)
            # s_measured of e for the energy energy_neutrino[index1], unit=1/MeV**2 (np.array of float):
            s_measured_e = (theo_spectrum[index1] * 1 / (np.sqrt(2 * np.pi) * sigma[index1]) *
                            np.exp(-0.5 * (e - corr_vis_neutrino[index1]) ** 2 / sigma[index1] ** 2))
            # integrate s_measured_e over e from energy_measured[index2]-binning_energy_measured/2 to
            # energy_measured[index2]+binning_energy_measured/2
            # ('integrate over the bin'), unit=1/MeV (float)
            # (is equal to s_m for one entry in energy_measured and one entry in energy_neutrino):
            s_measured_em1_en1 = np.array([np.trapz(s_measured_e, e)])
            # append value for energy_neutrino[index1] to s_measured_em1 to get an array, unit=1/MeV (np.array of float)
            # (s_measured_em1 is the measured spectrum for energy_measured[index2] as function of energy_neutrino):
            s_measured_em1 = np.append(s_measured_em1, s_measured_em1_en1)

        # integrate s_measured_em1 over e_neutrino, unit='number of neutrinos' (float) (s_measured is the total measured
        # spectrum for energy_measured[index2]:
        s_measured = np.array([np.trapz(s_measured_em1, energy_neutrino)])
        # to consider the binwidth of e_measured, divide the value of s_measured by the binning of energy_measured,
        # unit=1/MeV (float):
        s_measured = s_measured / binning_energy_measured
        # append s_measured to the array spectrum_measured, unit=1/MeV:
        spectrum_measured = np.append(spectrum_measured, s_measured)

    return spectrum_measured


def numberflux_dsnb(energy, e_mean, beta, luminosity, redshift, c, hubble_0, f_sn, h_param):
    """ function calculates the number-flux of electron-antineutrinos (Nu_E_bar OR muon- and tau-antineutrinos
        (Nu_x_bar) from diffuse supernova neutrino background (DSNB). The calculation is based on paper of Ando and
        Sato 'Relic neutrino background from cosmological supernovae' from 2004, arXiv:astro-ph/0410061v2

        1. the details of the calculation below is described in my notes DSNB background from 20.10.2017

        2. the simple functional from for the Star Formation Rate (SFR) per comoving volume Psi_star(z) is taken
        from page 5, equ. 4:
            -As reference model for the SFR, we adopt a model that is based on recent progressive results of
            rest-frame UV, NIR H-alpha, and FIR/sub-millimeter observations
            - the SFR with f_sn=1 is consistent with mildly dust-corrected UV data at low redshift; on the other hand,
            it may underestimate the results of the other wave band observations
            - it predicts the local SFR value of 0.7 * 10**(−2)*h70 M_sun yr−1 Mpc−3 , which is close to the lower
            limit of the estimation by Baldry and Glazebrook
            - Although the SFR-z relation generally tends to increase from z = 0 to ∼ 1, behaviours at the higher
            redshift region z > 1 are not clear at all.
            - INFO-me: In ref. 12, they showed that the SRN flux at E_neutrino > 10 MeV is highly insensitive to the
            difference among the SFR models (owing to the energy redshift, as discussed in section 3.1 of the paper)

        3. in the paper the supernova rate (R_SN(z)) is obtained from the SFR by assuming the
        Salpeter Initial Mass Function (IMF) (Phi(m) ~ m**(−2.35)) with a lower cutoff around 0.5*M_sun,
        and that all stars with M > 8*M_sun explode as core-collapse supernovae (page 5, equation 5)
            - assumption: INFO-me: IMF does not change with time,
            which may be a good approximation provided there are no significant correlations between the IMF and
            the environment in which stars are born
            -> extant evidence seems to argue against such correlations over the redshift range of interest (z > 2)
            - INFO-me: The resulting local SN rate evaluated with f_sn = 1 agrees within errors with the observed value

        4. Neutrino Spectrum from each supernova:
            - INFO-me:  PROBLEM: the recent sophisticated hydrodynamic simulations have not obtained the supernova
                        explosion itself; the shock wave cannot penetrate the entire core.
            - Many points remain controversial, for example, the average energy ratio among neutrinos of different
            flavors,  or how the gravitational binding energy is distributed to each flavour
            - INFO-me: serious problem in the estimation, since the binding energy released as electron-antineutrino
                    changes the normalization of the SRN flux, and the average energy affects the SRN spectral shape
            - The numerical simulation by the LL group is considered to be the most appropriate for our estimation,
            because it is the only model that succeeded in obtaining a robust explosion and in calculating the
            neutrino spectrum during the entire burst
            - the neutrino spectrum obtained by the LL simulation can be described by equation 6 on page 7
            - INFO-me:  the result of the LL group has been criticized because it lacked many relevant neutrino
            processes that are now recognized as important
            - INFO-me: all different models (LL, TBP, KRJ) can be well fitted with equ. 6 on page 7
            - TBP:  - include all the relevant neutrino processes, such as neutrino bremsstrahlung and neutrino-nucleon
                    scattering with nucleon recoil
                    - BUT: calculation obtain no explosion and neutrino spectrum ends at 0.25 s after core bounce
                    INFO-me:  In the strict sense, we cannot use their result as our reference model because the
                                fully time-integrated neutrino spectrum is definitely necessary in our estimate
                    - perfect equipartition between flavours is assumed (because the totally time-integrated neutrino
                    spectrum is not known)
            - KRJ: - calculations did not couple with the hydrodynamics, but it focused on the spectral formation
                   of neutrinos of each flavour using an MC simulation
                   - INFO-me: since the totally time-integrated neutrino flux is unknown from such temporary
                   information, we assume perfect equipartition

            INFO-me: perfect equipartition means a luminosity of 5*10**(52) erg for nu_e_bar and nu_x_bar

        5. the differential number flux of supernova relic neutrinos (SRN) is calculated with equation 3 on page 5

        TODO-me: the main uncertainty in the number flux of SRN lies in the neutrino spectrum from each supernova!

        :param energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        :param e_mean: mean energy of DSNB antineutrinos in MeV (float)
        :param beta: pinching parameters (float)
        :param luminosity: neutrino luminosity in MeV (1 erg = 624151 MeV) (float)
        :param redshift: redshift z is variable to calculate the flux (np.array of float64)
        :param c: speed of light in vacuum (float)
        :param hubble_0: Hubble constant in 1/s (float)
        :param f_sn: correction factor, represents f_star, a factor of order unity (float)
        :param h_param: parameter h_70 is 1 for a value of H_0 = 70 km/(s*Mpc) (integer)

        :return differential number flux of DSNB antineutrinos in 1/(MeV*s*cm**2) (np.array of float)
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


def darkmatter_signal_v1(energy_neutrino, energy_visible, mass_dm, crosssection, n_target, t, detection_efficiency,
                         mass_proton, mass_neutron, mass_positron):
    """ Simulate the signal from neutrinos from DM annihilation in the Milky Way:

        ! Version 1: Convolution of the theoretical spectrum with gaussian distribution is not correct
          (binning of energy_visible is not considered) !

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

        energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        energy_visible: energy corresponding to the visible energy in MeV (np.array of float)
        mass_dm: dark matter mass in MeV (float)
        crosssection: IBD cross-section for the DM signal in cm**2 (float), produced with the function sigma_ibd()
        n_target: number of free protons in JUNO (float)
        t: exposure time in seconds (float)
        detection_efficiency: detection efficiency of IBD in JUNO (float)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)

    :return spectrum_signal: visible spectrum of the electron-antineutrino signal from DM annihilation
            after t years for the JUNO detector in 1/MeV
            n_neutrino_signal_vis: number of neutrinos calculated from spectrum_signal (float)
            theo_spectrum_signal: theoretical spectrum of the signal in 1/MeV (np.array of float)
            n_neutrino_signal_theo: number of electron-antineutrino events in JUNO after "time" (float64)
            sigma_anni: DM annihilation cross-section in cm**3/s (float)
            j_avg: angular-averaged intensity over the whole Milky Way (float)
            phi_signal: electron-antineutrino flux at Earth in 1/(MeV * s * cm**2) (float)
    """

    """ Theoretical spectrum of the electron-antineutrino signal, monoenergetic spectrum at mass_DM: """
    # Calculate the electron-antineutrino flux from DM annihilation in the Milky Way at Earth from paper 0710.5420:
    # DM annihilation cross-section necessary to explain the observed abundance of DM in the Universe,
    # in cm**3/s (float):
    sigma_anni = 3 * 10 ** (-26)
    # canonical value of the angular-averaged intensity over the whole Milky Way (float):
    j_avg = 5.0
    # solar radius circle in cm, 8.5 kiloparsec, 1kpc = 3.086*10**21 cm (float):
    r_solar = 8.5 * 3.086 * 10 ** 21
    # normalizing DM density, in MeV/cm**3 (float):
    rho_0 = 0.3 * 1000
    # electron-antineutrino flux at Earth in 1/(MeV * s *cm**2) (float):
    phi_signal = sigma_anni / 6 * j_avg * r_solar * rho_0 ** 2 / mass_dm ** 2
    # Number of electron-antineutrino events in JUNO after "time" years in 1/MeV (float64):
    n_neutrino_signal_theo = crosssection * phi_signal * n_target * t * detection_efficiency

    # delta function from theoretical spectrum is approximated as very thin gaussian distribution
    # epsilon defines sigma in the gaussian distribution (float):
    epsilon = 10 ** (-6)
    delta_function = (1 / (np.sqrt(2 * np.pi * epsilon ** 2)) *
                      np.exp(-0.5 * (energy_neutrino - mass_dm) ** 2 / epsilon ** 2))
    # normalize delta_function (integral(delta_function) = 1) (array of float64):
    delta_function = delta_function / np.trapz(delta_function, energy_neutrino)
    # Theoretical spectrum of the electron-antineutrino signal in 1/MeV (number of events as function of the
    # electron-antineutrino energy) (array of float64):
    theo_spectrum_signal = n_neutrino_signal_theo * delta_function

    """ Calculate spectrum of the electron-antineutrino signal, theoretical spectrum is convolved 
        with gaussian distribution: """
    # Preallocate the spectrum array (np.array):
    spectrum_signal = np.array([])
    # convolve the 'theoretical'-spectrum with the gaussian distribution:
    for index in np.arange(0, len(energy_visible)):
        # correlation between visible and neutrino energy (visible energy depending on neutrino energy)
        # (np.array of float):
        corr_vis_neutrino = correlation_vis_neutrino(energy_neutrino, mass_proton, mass_neutron, mass_positron)
        # sigma of the gaussian distribution, defined by the energy resolution of the detector (np.array of float):
        sigma_resolution = energy_resolution(corr_vis_neutrino)
        # gaussian distribution characterized by corr_vis_neutrino and sigma_resolution (np.array of float):
        gauss = (1 / (np.sqrt(2 * np.pi) * sigma_resolution) *
                 np.exp(-0.5 * (energy_visible[index] - corr_vis_neutrino) ** 2 / (sigma_resolution ** 2)))
        # defines the integrand, which will be integrated over the neutrino energy energy_neutrino (np.array of float):
        integrand = theo_spectrum_signal * gauss
        # integrate the integrand over the neutrino energy energy_neutrino to get the value of the spectrum
        # for one visible energy energy_visible[index] in 1/MeV:
        spectrum_signal_index = np.array([np.trapz(integrand, energy_neutrino)])
        # append the single values spectrum_signal_index to the array spectrum_signal:
        spectrum_signal = np.append(spectrum_signal, spectrum_signal_index)

    # calculate the number of neutrinos from the spectrum_signal as test (float)
    # (compare it with n_neutrino_signal_theo to check if the convolution works well):
    n_neutrino_signal_vis = np.trapz(spectrum_signal, energy_visible)

    return (spectrum_signal, n_neutrino_signal_vis, theo_spectrum_signal, n_neutrino_signal_theo,
            sigma_anni, j_avg, phi_signal)


def darkmatter_signal_v2(energy_neutrino, energy_visible, binning_energy_visible, mass_dm, crosssection, n_target,
                         t, detection_efficiency, mass_proton, mass_neutron, mass_positron):
    """ Simulate the signal from neutrinos from DM annihilation in the Milky Way:

            ! Version 2: Convolution of the theoretical spectrum with gaussian distribution is calculated with the
             function convolution() !

            List of assumptions or estimations made for the calculation of the signal spectrum:
            1. neutrino flux (phi_signal) is calculated as described in paper 0710.5420
            1.1. #INFO-me: Assumption the branching ratio into neutrinos is dominant in DM self-annihilation
                (-> branching ratio into photons or electron-positron pairs is neglected)
            1.2. DM Halo profile uncertainties:
                - the neutrino flux is averaged over the entire galaxy, because of the poor information on the direction
                for this energies -> this helps in maximizing the observed neutrino flux while minimizing the impact of
                the choice of the DM profile
                - upper limit of the line-of-sight integration l_max depends on size of halo R_halo (equ. 2)
                    -> As contribution at large scales is negligible, different choices of R_halo do not affect
                    J_avg in a significant way
                    -> # INFO-me: as long as R_halo is a factor of a few larger than the scale radius r_s
                - while DM profiles tend to agree at large scales, uncertainties are still present for the inner
                region of the galaxy
                - neutrino flux scales with rho^2 → leads to an uncertainty in the overall normalization of the flux
                - Impact of halo profiles have been studied (three spherically symmetric profiles with isotropic
                velocity dispersion from more to less cuspy): MQGSL, NFW, KKBP
                - the values for the DM density at solar circle rho_sc of all three models satisfy the present
                constraints from the allowed values for the local rotational velocity, for the amount of flatness
                of the rotational curve of the Milky Way and for the maximal amount of its non-halo components
                - # INFO-me: By choosing the same rho_0 for all profiles, all the uncertainties in the neutrino flux,
                # INFO-me: which come from the lack of knowledge of the halo profile, lie in J_avg
                - in Table 1 of the paper the parameters considered for the halo profiles are shown and in paper
                0707.0196_DMhalo.pdf (reference [27] in 0710.5420) the values of J_avg are calculated
                    -> from 0707.0196: J_avg_MQGSL = 8, J_avg_NFW = 3, J_avg_KKBP = 2.6, canonical J_avg = 5

                -> # INFO-me: the differences of J_avg due to the choice of the halo profile from the canonical value
                # INFO-me: are small, less than a factor of 2

            1.3. in calculating the flux:
                - the factor 1/2 accounts for DM being its own antiparticle
                - # INFO-me: factor of 1/3 comes from the assumption that the branching ratio of annihilation is the
                # INFO-me: same in the three neutrino flavors
                    -> this assumption is ok! Thanks to neutrino oscillation, there is a guaranteed flux of neutrinos
                    in all flavors.

            2. theoretical signal spectrum (N_neutrino) is calculated as described in paper 0710.5420:
            2.1. # INFO-me: only IBD on free protons is considered

            3. Delta-function is approximated as very thin gaussian distribution

            :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
            :param energy_visible: energy corresponding to the visible energy in MeV (np.array of float)
            :param binning_energy_visible: bin-width of the visible energy in MeV (float)
            :param mass_dm: dark matter mass in MeV (float)
            :param crosssection: IBD cross-section for the DM signal in cm**2 (float), produced with the
            function sigma_ibd()
            :param n_target: number of free protons in JUNO (float)
            :param t: exposure time in seconds (float)
            :param detection_efficiency: detection efficiency of IBD in JUNO (float)
            :param mass_proton: mass of the proton in MeV (float)
            :param mass_neutron: mass of the neutron in MeV (float)
            :param mass_positron: mass of the positron in MeV (float)

        :return spectrum_signal: visible spectrum of the electron-antineutrino signal from DM annihilation
                after t years for the JUNO detector in 1/MeV
                n_neutrino_signal_vis: number of neutrinos calculated from spectrum_signal (float)
                theo_spectrum_signal: theoretical spectrum of the signal in 1/MeV (np.array of float)
                n_neutrino_signal_theo: number of electron-antineutrino events in JUNO after "time" (float64)
                sigma_anni: DM annihilation cross-section in cm**3/s (float)
                j_avg: angular-averaged intensity over the whole Milky Way (float)
                phi_signal: electron-antineutrino flux at Earth in 1/(MeV * s * cm**2) (float)
        """

    """ Theoretical spectrum of the electron-antineutrino signal, monoenergetic spectrum at mass_DM: """
    # Calculate the electron-antineutrino flux from DM annihilation in the Milky Way at Earth from paper 0710.5420:

    # DM self-annihilation cross-section (times the relative velocity) necessary to explain the observed abundance
    # of DM in the Universe, in cm**3/s (float):
    sigma_anni = 3 * 10 ** (-26)
    # angular-averaged DM intensity over the whole Milky Way, i.e. the average number flux (float):
    # (depending on the halo profile used j_avg can vary. The canonical value is j_avg_canonical = 5.
    # In paper 0707.0196_DMhalo.pdf the values of J_avg are calculated for the three halo profiles:
    # J_avg_MQGSL = 8, J_avg_NFW = 3, J_avg_KKBP = 2.6):
    # TODO-me: consider also other values of J_avg -> not only the canonical value
    j_avg = 5.0
    # solar radius circle in cm, 8.5 kiloparsec, 1kpc = 3.086*10**21 cm (float):
    r_solar = 8.5 * 3.086 * 10 ** 21
    # normalizing DM density, which is equal to commonly quoted DM density at r_solar, in MeV/cm**3 (float):
    rho_0 = 0.3 * 1000
    # electron-antineutrino flux at Earth in 1/(MeV * s *cm**2) (float):
    phi_signal = sigma_anni / 6 * j_avg * r_solar * rho_0 ** 2 / mass_dm ** 2
    # Number of electron-antineutrino events in JUNO after "time" years (float64):
    n_neutrino_signal_theo = crosssection * phi_signal * n_target * t * detection_efficiency

    # delta function from theoretical spectrum is approximated as very thin gaussian distribution
    # epsilon defines sigma in the gaussian distribution (float):
    epsilon = 10 ** (-6)
    delta_function = (1 / (np.sqrt(2 * np.pi * epsilon ** 2)) *
                      np.exp(-0.5 * (energy_neutrino - mass_dm) ** 2 / epsilon ** 2))
    # normalize delta_function (integral(delta_function) = 1) (array of float64):
    delta_function = delta_function / np.trapz(delta_function, energy_neutrino)
    # Theoretical spectrum of the electron-antineutrino signal in 1/MeV (number of events as function of the
    # electron-antineutrino energy) (array of float64):
    theo_spectrum_signal = n_neutrino_signal_theo * delta_function

    """ Calculate spectrum of the electron-antineutrino signal in 1/MeV, theoretical spectrum is convolved 
            with gaussian distribution: """
    spectrum_signal = convolution(energy_neutrino, energy_visible, binning_energy_visible, theo_spectrum_signal,
                                  mass_proton, mass_neutron, mass_positron)

    # calculate the number of neutrinos from the spectrum_signal as test (float)
    # (compare it with n_neutrino_signal_theo to check if the convolution works well):
    n_neutrino_signal_vis = np.trapz(spectrum_signal, energy_visible)

    return (spectrum_signal, n_neutrino_signal_vis, theo_spectrum_signal, n_neutrino_signal_theo,
            sigma_anni, j_avg, phi_signal)


def dsnb_background_v1(energy_neutrino, energy_visible, crosssection, n_target, t, detection_efficiency, c,
                       mass_proton, mass_neutron, mass_positron):
    """ Simulate the DSNB electron-antineutrino background:

        ! Version 1: Convolution of the theoretical spectrum with gaussian distribution is not correct
          (binning of energy_visible is not considered) !

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

        energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        energy_visible: energy corresponding to the visible energy in MeV (np.array of float)
        crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        n_target: number of free protons in JUNO (float)
        t: exposure time in seconds (float)
        detection_efficiency: detection efficiency of IBD in JUNO (float)
        c: speed of light in vacuum (float)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)

    :return Spectrum_DSNB: spectrum of electron-antineutrino of DSNB background in 1/MeV (np.array of float)
            n_neutrino_dsnb_vis: number of DSNB neutrinos calculated from spectrum_DSNB (np.array of float)
            theo_spectrum_dsnb: Theoretical spectrum of the DSNB electron-antineutrino background in 1/MeV
            (number of events as function of the electron-antineutrino energy) (np.array of float64)
            n_neutrino_dsnb_theo: number of DSNB electron-antineutrinos from theoretical spectrum (float64)
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
    number_flux__nu_e_bar = numberflux_dsnb(energy_neutrino, e_mean__nu_e_bar, beta__nu_e_bar, l__nu_e_bar, z,
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
    number_flux__nu_bar = numberflux_dsnb(energy_neutrino, e_mean__nu_bar, beta__nu_bar, l__nu_bar, z,
                                          c, h_0, f_star, h_70)
    # differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
    # about 30 percent of the emitted muon- and tau-antineutrinos will appear as electron-antineutrinos
    # at the Earth (np.array of float64) (page 9, equ. 7):
    number_flux__nu_bar = 0.3 * number_flux__nu_bar

    # Total number flux of electron anti-neutrinos from DSNB at Earth/JUNO in 1/(MeV * s * cm**2) (np.array of float64):
    number_flux_dsnb = number_flux__nu_e_bar + number_flux__nu_bar

    # Theoretical spectrum of DSNB neutrino events in JUNO after "time" years in 1/MeV (np.array of float64):
    theo_spectrum_dsnb = crosssection * number_flux_dsnb * n_target * t * detection_efficiency

    # number of neutrinos from DSNB background in JUNO detector after "time":
    n_neutrino_dsnb_theo = np.trapz(theo_spectrum_dsnb, energy_neutrino)

    """ Spectrum of the DSNB electron-antineutrino background, theoretical spectrum is convolved 
        with gaussian distribution: """
    # Preallocate the spectrum array (np.array):
    spectrum_dsnb = np.array([])
    # convolve the 'theoretical'-spectrum with the gaussian distribution:
    for index in np.arange(0, len(energy_visible)):
        # correlation between visible and neutrino energy (visible energy depending on neutrino energy)
        # (np.array of float):
        corr_vis_neutrino = correlation_vis_neutrino(energy_neutrino, mass_proton, mass_neutron, mass_positron)
        # sigma of the gaussian distribution, defined by the energy resolution of the detector (np.array of float):
        sigma_resolution = energy_resolution(corr_vis_neutrino)
        # gaussian distribution characterized by corr_vis_neutrino and sigma_resolution (np.array of float):
        gauss = (1 / (np.sqrt(2 * np.pi) * sigma_resolution) *
                 np.exp(-0.5 * (energy_visible[index] - corr_vis_neutrino) ** 2 / (sigma_resolution ** 2)))
        # defines the integrand, which will be integrated over the neutrino energy energy_neutrino (np.array of float):
        integrand = theo_spectrum_dsnb * gauss
        # integrate the integrand over the neutrino energy energy_neutrino to get the value of the spectrum
        # for one visible energy energy_visible[index] in 1/MeV:
        spectrum_dsnb_index = np.array([np.trapz(integrand, energy_neutrino)])
        # append the single values spectrum_signal_index to the array spectrum_signal:
        spectrum_dsnb = np.append(spectrum_dsnb, spectrum_dsnb_index)

    # calculate the number of neutrinos from the spectrum_dsnb as test (float)
    # (compare it with n_neutrino_dsnb_theo to check if the convolution works well):
    n_neutrino_dsnb_vis = np.trapz(spectrum_dsnb, energy_visible)

    return (spectrum_dsnb, n_neutrino_dsnb_vis, theo_spectrum_dsnb, n_neutrino_dsnb_theo,
            e_mean__nu_e_bar, beta__nu_e_bar, e_mean__nu_bar, beta__nu_bar, f_star)


def dsnb_background_v2(energy_neutrino, energy_visible, binning_energy_visible, crosssection, n_target, t,
                       detection_efficiency, c, mass_proton, mass_neutron, mass_positron):
    """ Simulate the DSNB electron-antineutrino background:

        ! Version 2: Convolution of the theoretical spectrum with gaussian distribution is calculated with the
        function convolution() !

        List of assumptions or estimations made for the calculation of the background spectrum from
        DSNB electron-antineutrinos:
        1. DSNB calculation based on the calculation in the paper of Ando and Sato 'Relic neutrino background from
           cosmological supernovae' from 2004, arXiv:astro-ph/0410061v2, AndoSato2004_0410061.pdf
        2. only electron-antineutrinos interacting with free protons are considered (Inverse Beta Decay),
           dominant up to ~80 MeV
        3. for the star formation rate (SFR) per unit comoving volume, a model is adopted based on recent progressive
           results of the rest-frame UV, NIR H_alpha, and FIR/sub-millimeter observations (depending on the correction
           factor f_star). behaviours in higher redshift regions z > 1 are not clear at all.
           INFO-me: to take the uncertainty of SFR into account, we have introduced the correction factor f_star
           INFO-me: f_star provides a wide range (0.7–4.2), which is still allowed from the SFR observation.
        4. the supernova rate from SFR is assumed by the Salpeter IMF with a lower cutoff around 0.5 solar masses and
           that all stars with masses greater than 8 solar masses explode as core-collapse supernovae.
        5. for the neutrino spectrum from each supernova, three reference models by different groups are adopted:
           simulations by Lawrence Livermore group (LL), simulations by Thomson, Burrows, Pinto (TBP),
           MC study of spectral formation Keil, Raffelt, Janka (KRJ).
           INFO-me: The most serious problem is that the recent sophisticated hydrodynamic simulations have not obtained
            the supernova explosion itself; the shock wave cannot penetrate the entire core.
           INFO-me: The numerical simulation by the LL group is considered to be the most appropriate for our
           estimations, because it is the only model that succeeded in obtaining a robust explosion
        6. 70 percent of the electron-antineutrinos survive and 30 percent of the muon-/tau-antineutrinos appear
           as electron-antineutrinos at the Earth (see page 9 of the paper)
           INFO-me: normal hierarchy is assumed
           TODO-me: for inverted hierarchy the neutrino flux at Earth differs from the flux for normal hierarchy

        TODO-me: the uncertainty about the SN neutrino spectrum and its luminosity gives at least a factor 2 to 4
                 ambiguity (Unklarheit) to the expected SRN flux in the energy region of our interest

        TODO-me: DSNB background depends on black hole star formation rate, has to be considered
        TODO-me: (see Talk of Julia at DPG and Janka, Kresse, Ertl from MPA Garching)

        It is found that in the energy range of our interest, more than 70 % of the flux comes from local supernova
        explosions at z < 1, while the high-redshift (z > 2) supernova contribution is very small.

        INFO-me: resonant spin flip of neutrino can also be considered (section 6.2 of the paper)
        INFO-me: decaying neutrino can also be considered (section 6.3 of the paper)

        :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        :param energy_visible: energy corresponding to the visible energy in MeV (np.array of float)
        :param binning_energy_visible: bin-width of the visible energy in MeV (float)
        :param crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        :param n_target: number of free protons in JUNO (float)
        :param t: exposure time in seconds (float)
        :param detection_efficiency: detection efficiency of IBD in JUNO (float)
        :param c: speed of light in vacuum (float)
        :param mass_proton: mass of the proton in MeV (float)
        :param mass_neutron: mass of the neutron in MeV (float)
        :param mass_positron: mass of the positron in MeV (float)

    :return Spectrum_DSNB: spectrum of electron-antineutrino of DSNB background in 1/MeV (np.array of float)
            n_neutrino_dsnb_vis: number of DSNB neutrinos calculated from spectrum_DSNB (np.array of float)
            theo_spectrum_dsnb: Theoretical spectrum of the DSNB electron-antineutrino background in 1/MeV
            (number of events as function of the electron-antineutrino energy) (np.array of float64)
            n_neutrino_dsnb_theo: number of DSNB electron-antineutrinos from theoretical spectrum (float64)
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
    # parameter: is 1 for a value of h_0 = 70 km/(s*Mpc) (integer):
    h_70 = 1
    # correction factor described in page 5 (float):
    # INFO-me: f_star is introduced to take the uncertainty of the SRF into account, provides a wide range (0.7 to 4.2)
    # TODO-me: use different values for the correction factor f_star
    f_star = 1

    """ differential number flux for electron-antineutrinos: """
    # TODO-me: use different models (LL or TBP or KRJ) for the neutrino spectrum -> different values of E_mean and beta
    # TODO-me: calculate the DSNB flux also for inverted hierarchy!!
    # Fitting parameters from LL model:
    # mean energy of electron-antineutrinos in MeV (float) (page 7, table 1):
    e_mean__nu_e_bar = 15.4
    # pinching parameters (float) (page 7, table 1):
    beta__nu_e_bar = 3.8
    # neutrino luminosity in MeV (1 erg = 624151 MeV) (float) (page 7, table 1):
    l__nu_e_bar = 4.9 * 10 ** 52 * 624151
    # differential number flux of electron-antineutrinos in 1/(MeV*s*cm**2) (np.array of float) (page 5, equ. 3):
    number_flux__nu_e_bar = numberflux_dsnb(energy_neutrino, e_mean__nu_e_bar, beta__nu_e_bar, l__nu_e_bar, z,
                                            c, h_0, f_star, h_70)
    # differential number flux of electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
    # about 70 percent of electron-antineutrinos survive (np.array of float64) (page 9, equ. 7):
    # INFO-me: is correct for normal hierarchy
    number_flux__nu_e_bar = 0.7 * number_flux__nu_e_bar

    """ differential number flux for NON-electron-antineutrinos (muon- and tau-antineutrinos): """
    # Fitting parameters from LL model:
    # mean energy of NON-electron-antineutrinos in MeV (float) (page 7, table 1):
    e_mean__nu_bar = 21.6
    # pinching parameters (float) (page 7, table 1):
    beta__nu_bar = 1.8
    # neutrino luminosity in MeV (float) (page 7, table 1):
    l__nu_bar = 5.0 * 10 ** 52 * 624151
    # differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) (np.array of float) (page 5, equ. 3):
    number_flux__nu_bar = numberflux_dsnb(energy_neutrino, e_mean__nu_bar, beta__nu_bar, l__nu_bar, z,
                                          c, h_0, f_star, h_70)
    # differential number flux of NON-electron-antineutrinos in 1/(MeV*s*cm**2) at Earth -> Oscillation is considered,
    # about 30 percent of the emitted muon- and tau-antineutrinos will appear as electron-antineutrinos
    # at the Earth (np.array of float64) (page 9, equ. 7):
    # INFO-me: is correct for normal hierarchy
    number_flux__nu_bar = 0.3 * number_flux__nu_bar

    # Total number flux of electron anti-neutrinos from DSNB at Earth/JUNO in 1/(MeV * s * cm**2) (np.array of float64):
    number_flux_dsnb = number_flux__nu_e_bar + number_flux__nu_bar

    # Theoretical spectrum of DSNB neutrino events in JUNO after "time" years in 1/MeV (np.array of float64):
    theo_spectrum_dsnb = crosssection * number_flux_dsnb * n_target * t * detection_efficiency

    # number of neutrinos from DSNB background in JUNO detector after "time":
    n_neutrino_dsnb_theo = np.trapz(theo_spectrum_dsnb, energy_neutrino)

    """ Spectrum of the DSNB electron-antineutrino background in 1/MeV, theoretical spectrum is convolved 
        with gaussian distribution: """
    spectrum_dsnb = convolution(energy_neutrino, energy_visible, binning_energy_visible, theo_spectrum_dsnb,
                                mass_proton, mass_neutron, mass_positron)

    # calculate the number of neutrinos from the spectrum_dsnb as test (float)
    # (compare it with n_neutrino_dsnb_theo to check if the convolution works well):
    n_neutrino_dsnb_vis = np.trapz(spectrum_dsnb, energy_visible)

    return (spectrum_dsnb, n_neutrino_dsnb_vis, theo_spectrum_dsnb, n_neutrino_dsnb_theo,
            e_mean__nu_e_bar, beta__nu_e_bar, e_mean__nu_bar, beta__nu_bar, f_star)


def reactor_background_v1(energy_neutrino, energy_visible, crosssection, n_target, tyears, detection_efficiency,
                          mass_proton, mass_neutron, mass_positron):
    """ Simulate the reactor electron-antineutrino background in JUNO:

        ! Version 1: Convolution of the theoretical spectrum with gaussian distribution is not correct
          (binning of energy_visible is not considered) !

        List of assumptions or estimations made for the calculation of the reactor-antineutrino background:
        1. the reactor electron-antineutrino flux is calculated according to the paper of Fallot2012
           (PhysRevLett.109.202504.pdf, New antineutrino Energy Spectra Predictions from the Summation of Beta
           Decay Branches of the Fission Products). The fluxes of electron-antineutrinos for each fission product are
           digitized from figure 1 with software engauge digitizer.
        2. neutrino-oscillation is taken into account: same calculation like in NuOscillation.cc and NuOscillation.hh.
           Normal Hierarchy is considered

        energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        energy_visible: energy corresponding to the visible energy in MeV (np.array of float)
        crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        n_target: number of free protons in JUNO (float)
        tyears: exposure time in years (float)
        detection_efficiency: detection efficiency of IBD in JUNO (float)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)

    :return: spectrum_reactor: spectrum of electron-antineutrinos of reactor background in 1/MeV (np.array of float)
             n_neutrino_reactor_vis: number of neutrinos from calculated spectrum (float)
             theo_spectrum_reactor: Theoretical spectrum of the reactor electron-antineutrino background
             in 1/MeV (number of events as function of the electron-antineutrino energy) (np.array of float64)
             n_neutrino_reactor_theo: number of reactor electron-antineutrino events in JUNO after "time"
             (float64)
             power_th: total thermal power of the Yangjiang and Taishan NPP in GW (float)
             fraction235_u: fission fraction of U235 (float)
             fraction238_u: fission fraction of U238 (float)
             fraction239_pu: fission fraction of Pu239 (float)
             fraction241_pu: fission fraction of Pu241 (float)
             l_m: distance from reactor to detector in meter (float)
    """

    """ Theoretical spectrum of reactor electron-antineutrinos: """
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
    # INFO-me: Do I have to consider the binning in figure 1 of Fallot paper: NO, see Fallot_spectrum_analysis.py)
    u235_fallot = np.interp(energy_neutrino, energy_fallot_u235, flux_fallot_u235)
    u238_fallot = np.interp(energy_neutrino, energy_fallot_u238, flux_fallot_u238)
    pu239_fallot = np.interp(energy_neutrino, energy_fallot_pu239, flux_fallot_pu239)
    pu241_fallot = np.interp(energy_neutrino, energy_fallot_pu241, flux_fallot_pu241)
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
    p21 = sin2_2th12 * cos4_th13 * np.sin(1.267 * dm2_21 * l_m / energy_neutrino) ** 2
    p31 = sin2_2th13 * cos2_th12 * np.sin(1.267 * dm2_31 * l_m / energy_neutrino) ** 2
    p32 = sin2_2th13 * sin2_th12 * np.sin(1.267 * dm2_32 * l_m / energy_neutrino) ** 2
    # Survival probability of electron-antineutrinos for Normal Hierarchy (np.array of float):
    prob_oscillation_nh = 1. - p21 - p31 - p32

    """ Theoretical reactor electron-antineutrino spectrum in JUNO with oscillation """
    # Theoretical spectrum in JUNO for normal hierarchy with oscillation in units of electron-antineutrinos/MeV
    # in "t_years" years (np.array of float):
    theo_spectrum_reactor = (1 / (4 * np.pi * l_cm ** 2) * flux_reactor * crosssection * detection_efficiency *
                             n_target * tyears * prob_oscillation_nh)

    # number of neutrinos from reactor background in JUNO detector after "time":
    n_neutrino_reactor_theo = np.trapz(theo_spectrum_reactor, energy_neutrino)

    """ Spectrum of the reactor electron-antineutrino background, theoretical spectrum is convolved 
        with gaussian distribution: """
    # Preallocate the spectrum array (np.array):
    spectrum_reactor = np.array([])
    # convolve the 'theoretical'-spectrum with the gaussian distribution:
    for index in np.arange(0, len(energy_visible)):
        # correlation between visible and neutrino energy (visible energy depending on neutrino energy)
        # (np.array of float):
        corr_vis_neutrino = correlation_vis_neutrino(energy_neutrino, mass_proton, mass_neutron, mass_positron)
        # sigma of the gaussian distribution, defined by the energy resolution of the detector (np.array of float):
        sigma_resolution = energy_resolution(corr_vis_neutrino)
        # gaussian distribution characterized by corr_vis_neutrino and sigma_resolution (np.array of float):
        gauss = (1 / (np.sqrt(2 * np.pi) * sigma_resolution) *
                 np.exp(-0.5 * (energy_visible[index] - corr_vis_neutrino) ** 2 / (sigma_resolution ** 2)))
        # defines the integrand, which will be integrated over the neutrino energy energy_neutrino (np.array of float):
        integrand = theo_spectrum_reactor * gauss
        # integrate the integrand over the neutrino energy energy_neutrino to get the value of the spectrum
        # for one visible energy energy_visible[index] in 1/MeV:
        spectrum_reactor_index = np.array([np.trapz(integrand, energy_neutrino)])
        # append the single values spectrum_reactor_index to the array spectrum_reactor:
        spectrum_reactor = np.append(spectrum_reactor, spectrum_reactor_index)

    # calculate the number of neutrinos from the spectrum_reactor as test (float)
    # (compare it with n_neutrino_reactor_theo to check if the convolution works well):
    n_neutrino_reactor_vis = np.trapz(spectrum_reactor, energy_visible)

    return (spectrum_reactor, n_neutrino_reactor_vis, theo_spectrum_reactor, n_neutrino_reactor_theo, power_th,
            fraction235_u, fraction238_u, fraction239_pu, fraction241_pu, l_m)


def reactor_background_v2(energy_neutrino, energy_visible, binning_energy_visible, crosssection, n_target, tyears,
                          detection_efficiency, mass_proton, mass_neutron, mass_positron):
    """ Simulate the reactor electron-antineutrino background in JUNO:

        ! Version 2: Convolution of the theoretical spectrum with gaussian distribution is calculated with the
        function convolution() !

        The different paper of Vogel1989 (used in JUNO-offline generator KRLReactorFlux.cc), of Müller2011,
        of Huber2011, of Novella2015 and of Fallot2012 are described in detailed in my notes

        List of assumptions or estimations made for the calculation of the reactor-antineutrino background:
        1. the reactor electron-antineutrino flux is calculated according to the paper of Fallot2012
           (PhysRevLett.109.202504.pdf, New antineutrino Energy Spectra Predictions from the Summation of Beta
           Decay Branches of the Fission Products). The fluxes of electron-antineutrinos for each fission product are
           digitized from figure 1 with software 'engauge digitizer'.

        1.1. Fallot2012 paper:
            'For each fission product from any of the databases, we reconstruct its antineutrino energy spectrum
            through the summation of its individual beta branches, using the prescription of Huber2011 and taking
            into account the transition type, when given, up to the third forbidden unique transition.'

        2. neutrino-oscillation is taken into account: same calculation like in NuOscillation.cc and NuOscillation.hh.
           Normal Hierarchy is considered

        # INFO-me: for DM masses >= 20 MeV the contribution of reactor neutrinos to the spectrum is very small
        # INFO-me: (for DM mass = 20 MeV and energy window from 15 to 25 MeV, only 0.00003 reactor background events
        # INFO-me: are expected)

        # TODO-me: the reactor flux and its uncertainties becomes interesting for DM masses below 20 MeV

        :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        :param energy_visible: energy corresponding to the visible energy in MeV (np.array of float)
        :param binning_energy_visible: bin-width of the visible energy in MeV (float)
        :param crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        :param n_target: number of free protons in JUNO (float)
        :param tyears: exposure time in years (float)
        :param detection_efficiency: detection efficiency of IBD in JUNO (float)
        :param mass_proton: mass of the proton in MeV (float)
        :param mass_neutron: mass of the neutron in MeV (float)
        :param mass_positron: mass of the positron in MeV (float)

        :return: spectrum_reactor: spectrum of electron-antineutrinos of reactor background in 1/MeV (np.array of float)
                 n_neutrino_reactor_vis: number of neutrinos from calculated spectrum (float)
                 theo_spectrum_reactor: Theoretical spectrum of the reactor electron-antineutrino background
                 in 1/MeV (number of events as function of the electron-antineutrino energy) (np.array of float64)
                 n_neutrino_reactor_theo: number of reactor electron-antineutrino events in JUNO after "time"
                 (float64)
                 power_th: total thermal power of the Yangjiang and Taishan NPP in GW (float)
                 fraction235_u: fission fraction of U235 (float)
                 fraction238_u: fission fraction of U238 (float)
                 fraction239_pu: fission fraction of Pu239 (float)
                 fraction241_pu: fission fraction of Pu241 (float)
                 l_m: distance from reactor to detector in meter (float)
        """

    """ Theoretical spectrum of reactor electron-antineutrinos: """
    # Total thermal power of the Yangjiang and Taishan nuclear power plants (from PhysicsReport, page 29), in GW
    # (float):
    # INFO-me: maybe two cores are not ready until 2020, in which case the total thermal power will be 26.55 GW
    power_th = 35.73
    # Digitized data:
    # corresponding electron-antineutrino energy for U235 in MeV (np.array of float):
    # INFO-me: one data-point of 17 MeV is added, important for interpolation
    energy_fallot_u235 = np.array([0.3841, 0.7464, 1.1087, 1.471, 1.7609, 2.1232, 2.4855, 2.8478, 3.1377, 3.5, 3.8623,
                                   4.1522, 4.5145, 4.8043, 5.0942, 5.3841, 5.7464, 6.0362, 6.3261, 6.6159, 6.9058,
                                   7.1957, 7.4855, 7.7754, 7.9928, 8.1377, 8.3551, 8.6449, 8.8623, 9.1522, 9.442,
                                   9.7319, 9.8768, 10.1667, 10.529, 10.8188, 11.1087, 11.3986, 11.6884, 11.8333,
                                   12.0507, 12.1232, 12.1957, 12.2681, 12.3406, 12.4855, 12.7754, 13.1377, 13.3551,
                                   13.5725, 13.9348, 14.2246, 14.442, 14.7319, 14.9493, 15.0217, 15.0942, 15.1667,
                                   15.3841, 15.6739, 15.8913, 16.1087, 17.0])
    # electron-antineutrino flux from figure 1 for U235 in antineutrinos/(MeV*fission) (np.array of float):
    # INFO-me: data-point of 0.0 events/(MeV*fission) for 17 MeV is added
    flux_fallot_u235 = np.array([1.548, 1.797, 1.797, 1.548, 1.334, 1.149, 0.852, 0.734, 0.5446, 0.4041, 0.3481, 0.2224,
                                 0.165, 0.1225, 0.0909, 0.06741, 0.05001, 0.03196, 0.02371, 0.01759, 0.01124, 0.007186,
                                 0.004592, 0.002528, 0.001392, 0.000766, 0.0003632, 0.0002321, 0.0001484, 0.0000948,
                                 0.00006059, 0.00003872, 0.00002132, 0.00001362, 0.00001011, 0.000006459, 0.000004792,
                                 0.000003063, 0.000001686, 0.000000928, 0.0000005931, 0.0000002812, 0.0000001334,
                                 0.00000006323, 0.00000002998, 0.0000000165, 0.00000001225, 0.00000000783,
                                 0.000000005001, 0.000000003196, 0.000000002043, 0.000000001515, 0.000000000969,
                                 0.000000000619, 0.0000000003407, 0.0000000001616, 0.00000000003632, 0.00000000001722,
                                 0.00000000001101, 0.00000000000817, 0.000000000004495, 0.000000000002475, 0.0])
    # corresponding electron-antineutrino energy for U238 in MeV (np.array of float):
    # INFO-me: one data-point of 17 MeV is added, important for interpolation
    energy_fallot_u238 = np.array([0.2391, 0.6014, 0.9638, 1.3261, 1.6159, 1.9783, 2.3406, 2.7029, 2.9928, 3.3551,
                                   3.7174, 4.0797, 4.3696, 4.7319, 5.0217, 5.3841, 5.6739, 6.0362, 6.3261, 6.6159,
                                   6.9783, 7.2681, 7.558, 7.8478, 8.1377, 8.3551, 8.6449, 8.9348, 9.2246, 9.5145,
                                   9.8043, 10.0942, 10.3841, 10.7464, 11.0362, 11.3261, 11.5435, 11.7609, 11.9783,
                                   12.1957, 12.2681, 12.3406, 12.413, 12.7754, 13.0652, 13.3551, 13.7174, 14.0072,
                                   14.2971, 14.587, 14.8768, 15.0217, 15.0942, 15.1667, 15.2391, 15.4565, 15.7464,
                                   16.0362, 16.1812, 17.0])
    # electron-antineutrino flux from figure 1 for U238 in electron-antineutrinos/(MeV*fission) (np.array of float):
    # INFO-me: data-point of 0.0 events/(MeV*fission) for 17 MeV is added
    flux_fallot_u238 = np.array([2.422, 2.812, 2.812, 2.087, 1.797, 1.548, 1.149, 0.989, 0.852, 0.734, 0.5446, 0.4041,
                                 0.3481, 0.2582, 0.165, 0.1422, 0.0909, 0.06741, 0.04308, 0.03196, 0.02371, 0.01515,
                                 0.00969, 0.00619, 0.003956, 0.002178, 0.001392, 0.000889, 0.0005684, 0.0003632,
                                 0.0002695, 0.0001722, 0.0001278, 0.0000817, 0.00006059, 0.00003872, 0.00002475,
                                 0.00001582, 0.00000871, 0.000004792, 0.000002272, 0.000001077, 0.0000005109,
                                 0.0000003791, 0.0000003265, 0.0000002812, 0.0000002087, 0.0000001334, 0.0000000989,
                                 0.00000006323, 0.00000004041, 0.00000002224, 0.00000001055, 0.000000001124,
                                 0.0000000005332, 0.0000000001616, 0.0000000001032, 0.00000000006598, 0.00000000003632,
                                 0.0])
    # corresponding electron-antineutrino energy for Pu239 in MeV (np.array of float):
    # #INFO-me: one data-point of 17 MeV is added, important for interpolation
    energy_fallot_pu239 = np.array([0.3841, 0.7464, 1.0362, 1.3986, 1.7609, 2.0507, 2.413, 2.7029, 3.0652, 3.4275,
                                    3.7174, 4.0072, 4.3696, 4.6594, 5.0217, 5.3116, 5.6014, 5.8913, 6.2536, 6.5435,
                                    6.8333, 7.1232, 7.413, 7.7029, 7.9203, 8.1377, 8.2101, 8.3551, 8.5725, 8.8623,
                                    9.1522, 9.442, 9.6594, 9.9493, 10.1667, 10.4565, 10.7464, 11.0362, 11.3986,
                                    11.6159, 11.7609, 11.9783, 12.1957, 12.3406, 12.4855, 12.7754, 13.0652, 13.2826,
                                    13.4275, 13.5725, 13.6449, 13.7174, 14.0072, 14.2971, 14.5145, 14.8043, 15.0217,
                                    15.0942, 15.1667, 15.2391, 15.4565, 15.7464, 16.0362, 16.1812, 17.0])
    # electron-antineutrino flux from figure 1 for Pu239 in electron-antineutrinos/(MeV*fission) (np.array of float):
    # INFO-me: data-point of 0.0 events/(MeV*fission) for 17 MeV is added
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
    # corresponding electron-antineutrino energy for Pu241 in MeV (np.array of float):
    # INFO-me: one data-point of 17 MeV is added, important for interpolation
    energy_fallot_pu241 = np.array([0.3841, 0.7464, 1.0362, 1.3986, 1.7609, 2.1232, 2.413, 2.7754, 3.0652, 3.4275,
                                    3.7899, 4.0797, 4.442, 4.7319, 5.0217, 5.3841, 5.6739, 5.9638, 6.3261, 6.6159,
                                    6.9058, 7.1957, 7.4855, 7.7029, 7.9928, 8.1377, 8.3551, 8.5, 8.7899, 9.0797,
                                    9.3696, 9.587, 9.8768, 10.0942, 10.4565, 10.7464, 11.0362, 11.3261, 11.6159,
                                    11.7609, 11.9783, 12.1232, 12.1957, 12.3406, 12.413, 12.4855, 12.8478, 13.2101,
                                    13.5, 13.7899, 14.0797, 14.442, 14.6594, 14.9493, 15.0217, 15.0942, 15.1667,
                                    15.3841, 15.6739, 15.9638, 16.1087, 17.0])
    # electron-antineutrino flux from figure 1 for Pu241 in electron-antineutrinos/(MeV*fission) (np.array of float):
    # INFO-me: data-point of 0.0 events/(MeV*fission) for 17 MeV is added
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
    # INFO-me: binning in figure 1 of Fallot paper has NOT to be considered, see Fallot_spectrum_analysis.py)
    u235_fallot = np.interp(energy_neutrino, energy_fallot_u235, flux_fallot_u235)
    u238_fallot = np.interp(energy_neutrino, energy_fallot_u238, flux_fallot_u238)
    pu239_fallot = np.interp(energy_neutrino, energy_fallot_pu239, flux_fallot_pu239)
    pu241_fallot = np.interp(energy_neutrino, energy_fallot_pu241, flux_fallot_pu241)
    # Fractions (data taken from PhysicsReport_1507.05613, page 136, averaged value of the Daya Bay nuclear cores),
    # (float):
    # TODO-me: are values of the fractions of the fission products from Daya Bay correct?
    fraction235_u = 0.577
    fraction239_pu = 0.295
    fraction238_u = 0.076
    fraction241_pu = 0.052
    # add the weighted sum of the terms, electron-antineutrino spectrum
    # in units of electron-antineutrinos/(MeV * fission) (data taken from KRLReactorFLux.cc) (np.array of float):
    spec1_reactor = (u235_fallot * fraction235_u + pu239_fallot * fraction239_pu + u238_fallot * fraction238_u +
                     pu241_fallot * fraction241_pu)
    # There are 3.125*10**19 fissions/GW/second (reference: KRLReactorFlux.cc -> reference there:
    # http://www.nuc.berkeley.edu/dept/Courses/NE-150/Criticality.pdf, BUT: "page not found"
    #  spectrum in units of electron-antineutrino/(MeV * GW * s) (np.array of float):
    # TODO-me: no reference of the number of fission/GW/second
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
    p21 = sin2_2th12 * cos4_th13 * np.sin(1.267 * dm2_21 * l_m / energy_neutrino) ** 2
    p31 = sin2_2th13 * cos2_th12 * np.sin(1.267 * dm2_31 * l_m / energy_neutrino) ** 2
    p32 = sin2_2th13 * sin2_th12 * np.sin(1.267 * dm2_32 * l_m / energy_neutrino) ** 2
    # Survival probability of electron-antineutrinos for Normal Hierarchy (np.array of float):
    prob_oscillation_nh = 1. - p21 - p31 - p32

    """ Theoretical reactor electron-antineutrino spectrum in JUNO with oscillation """
    # Theoretical spectrum in JUNO for normal hierarchy with oscillation in units of electron-antineutrinos/MeV
    # in "t_years" years (np.array of float):
    theo_spectrum_reactor = (1 / (4 * np.pi * l_cm ** 2) * flux_reactor * crosssection * detection_efficiency *
                             n_target * tyears * prob_oscillation_nh)

    # number of neutrinos from reactor background in JUNO detector after "time" (float):
    n_neutrino_reactor_theo = np.trapz(theo_spectrum_reactor, energy_neutrino)

    """ Spectrum of the reactor electron-antineutrino background in 1/MeV, theoretical spectrum is convolved 
        with gaussian distribution: """
    spectrum_reactor = convolution(energy_neutrino, energy_visible, binning_energy_visible, theo_spectrum_reactor,
                                   mass_proton, mass_neutron, mass_positron)

    # calculate the number of neutrinos from the spectrum_reactor as test (float)
    # (compare it with n_neutrino_reactor_theo to check if the convolution works well):
    n_neutrino_reactor_vis = np.trapz(spectrum_reactor, energy_visible)

    return (spectrum_reactor, n_neutrino_reactor_vis, theo_spectrum_reactor, n_neutrino_reactor_theo, power_th,
            fraction235_u, fraction238_u, fraction239_pu, fraction241_pu, l_m)


def ccatmospheric_background_v1(energy_neutrino, energy_visible, crosssection, n_target, t, detection_efficiency,
                                mass_proton, mass_neutron, mass_positron):
    """ Simulate the atmospheric Charged Current electron-antineutrino background:

        ! Version 1: Convolution of the theoretical spectrum with gaussian distribution is not correct
          (binning of energy_visible is not considered) !

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

        energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        energy_visible: energy corresponding to the visible energy in the detector (np.array of float)
        crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        n_target: number of free protons in JUNO (float)
        t: exposure time in seconds (float)
        detection_efficiency: detection efficiency of IBD in JUNO (float)
        mass_proton: mass of the proton in MeV (float)
        mass_neutron: mass of the neutron in MeV (float)
        mass_positron: mass of the positron in MeV (float)

    :return spectrum_ccatmospheric: spectrum of electron-antineutrinos of CC atmospheric background (np.array of float)
            n_neutrino_ccatmospheric_vis: number of neutrinos from calculated spectrum (float9)
            theo_spectrum_ccatmospheric: Theoretical spectrum of the atmospheric CC electron-antineutrino background
            in 1/MeV (number of events as function of the electron-antineutrino energy) (np.array of float64)
            e_visible_ccatmospheric: array of visible energies in MeV (np.array of float)
            n_neutrino_ccatmospheric_theo: number of atmospheric CC electron-antineutrino events in JUNO after "time"
            (float64)
            oscillation: oscillation is considered for oscillation=1, oscillation is not considered for oscillation=0
            (integer)
            prob_e_to_e: survival probability of electron-antineutrinos (electron-antineutrinos oscillate to
            electron-antineutrinos) (float)
            prob_mu_to_e: oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos) (float)
    """

    """ Theoretical spectrum of atmospheric charged-current background: """
    # Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
    e_data = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133, 150,
                       168, 188])

    # differential flux in energy for no oscillation for electron-antineutrinos for solar average at the site
    # of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_e_bar_data = 10 ** (-4) * np.array([0., 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101.,
                                                       96.1, 83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9,
                                                       28.8, 24.9, 21.3, 18.3])
    # linear interpolation of the simulated data above to get the differential neutrino flux corresponding to energy,
    # differential flux of electron-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_e_bar = np.interp(energy_neutrino, e_data, flux_ccatmo_nu_e_bar_data)

    # differential flux in energy for no oscillation for muon-antineutrinos for solar average at the site of Super-K,
    # in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_mu_bar_data = 10 ** (-4) * np.array([0., 116., 128., 136., 150., 158., 162., 170., 196., 177., 182.,
                                                        183., 181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5,
                                                        64.0, 55.6, 47.6, 40.8])
    # linear interpolation of the simulated data above to get the differential neutrino flux corresponding to energy,
    # differential flux of muon-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_mu_bar = np.interp(energy_neutrino, e_data, flux_ccatmo_nu_mu_bar_data)

    # total flux of electron-antineutrinos at the detector WITHOUT oscillation in (MeV**(-1) * cm**(-2) * s**(-1))
    # (factors set to 1 and 0) (np.array of float):
    # INFO-me: consider the flux with oscillation
    # Integer, that defines, if oscillation is considered (oscillation=1) or not (oscillation=0):
    oscillation = 0
    # survival probability of electron-antineutrinos:
    prob_e_to_e = 1
    # oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos):
    prob_mu_to_e = 0
    # total flux in 1/(MeV * cm**2 * s) (np.array of float):
    flux_total_ccatmospheric_nu_e_bar = prob_e_to_e * flux_ccatmo_nu_e_bar + prob_mu_to_e * flux_ccatmo_nu_mu_bar
    # Theoretical spectrum (in events per MeV) of electron-antineutrinos ("number of positron-events") from
    # inverse beta decay on free protons (from paper 0903.5323.pdf, equ. 64) (np.array of float):
    theo_spectrum_ccatmospheric = (flux_total_ccatmospheric_nu_e_bar * crosssection *
                                   detection_efficiency * n_target * t)

    # number of neutrino from CC atmospheric background in JUNO detector after "time":
    n_neutrino_ccatmospheric_theo = np.trapz(theo_spectrum_ccatmospheric, energy_neutrino)

    """ Spectrum of the atmospheric charged-current electron-antineutrino background, 
        theoretical spectrum is convolved with gaussian distribution: """
    # Preallocate the spectrum array (np.array):
    spectrum_ccatmospheric = np.array([])
    # convolve the 'theoretical'-spectrum with the gaussian distribution:
    for index in np.arange(0, len(energy_visible)):
        # correlation between visible and neutrino energy (visible energy depending on neutrino energy)
        # (np.array of float):
        corr_vis_neutrino = correlation_vis_neutrino(energy_neutrino, mass_proton, mass_neutron, mass_positron)
        # sigma of the gaussian distribution, defined by the energy resolution of the detector (np.array of float):
        sigma_resolution = energy_resolution(corr_vis_neutrino)
        # gaussian distribution characterized by corr_vis_neutrino and sigma_resolution (np.array of float):
        gauss = (1 / (np.sqrt(2 * np.pi) * sigma_resolution) *
                 np.exp(-0.5 * (energy_visible[index] - corr_vis_neutrino) ** 2 / (sigma_resolution ** 2)))
        # defines the integrand, which will be integrated over the neutrino energy energy_neutrino (np.array of float):
        integrand = theo_spectrum_ccatmospheric * gauss
        # integrate the integrand over the neutrino energy energy_neutrino to get the value of the spectrum
        # for one visible energy energy_visible[index] in 1/MeV:
        spectrum_ccatmospheric_index = np.array([np.trapz(integrand, energy_neutrino)])
        # append the single values spectrum_ccatmospheric_index to the array spectrum_ccatmospheric:
        spectrum_ccatmospheric = np.append(spectrum_ccatmospheric, spectrum_ccatmospheric_index)

    # calculate the number of neutrinos from the spectrum_ccatmospheric as test (float)
    # (compare it with n_neutrino_ccatmospheric_theo to check if the convolution works well):
    n_neutrino_ccatmospheric_vis = np.trapz(spectrum_ccatmospheric, energy_visible)

    return (spectrum_ccatmospheric, n_neutrino_ccatmospheric_vis, theo_spectrum_ccatmospheric,
            n_neutrino_ccatmospheric_theo, oscillation, prob_e_to_e, prob_mu_to_e)


def ccatmospheric_background_v2(energy_neutrino, energy_visible, binning_energy_visible, crosssection, n_target,
                                t, detection_efficiency, mass_proton, mass_neutron, mass_positron):
    """ Simulate the atmospheric Charged Current electron-antineutrino background:

        ! Version 2: Convolution of the theoretical spectrum with gaussian distribution is calculated with the
        function convolution() !

        The paper of Battistoni2005 'The atmospheric neutrino fluxes below 100 MeV: The FLUKA results' is described in
        detail in the my notes. In the paper the electron- and muon-antineutrino flux is simulated for energies
        from 10 MeV to 100 MeV.


        List of assumptions or estimations made for the calculation of the atmospheric Charged Current Background:

        1. the values of the differential flux in energy for no oscillation for electron- and muon-antineutrinos
        for solar average at the site of Super-Kamiokande is taken from table 2 and 3 of Battistoni's paper 'The
        atmospheric neutrino flux below 100 MeV: the FLUKA results' from 2005 (1-s2.0-S0927650505000526-main):
            1.1 the difference in site (LNGS and Super-K) is only due to the geomagnetic cutoff (the ground profile
                (mountain shape, elevation, etc.) and local atmosphere is not yet implemented)
                INFO-me: these may account for a few percent systematics in the determination of the flux
            1.2 baseline choices of the FLUKA calculation:
                - the primary spectrum in the region of kinetic energy per nucleon relevant for this topic
                (between 0.5 GeV and 100 GeV) is constrained by the results of AMS and BESS for proton and helium.
                CAPRICE data exhibit a normalization about 25 % lower
                INFO-me: This is a relevant source of uncertainty
                - the primary flux is converted into an all-nucleon flux -> superposition model in FLUKA
                INFO-me: does not include models for nucleus-nucleus collisions yet
                - spherical representation of the Earth and the surrounding atmosphere up to 70 km
                - the average USA standard atmosphere is used over the whole Earth surface (atmosphere is layered in
                100 shells with a density scaling according to the chosen profile as a function of height)
                - The geomagnetic cutoff is applied by means of the back-tracing technique using the IGRF model
                - the solar modulation model in reference 19 is used
            1.3 INFO-me: only electron-antineutrinos, because only IBD on free protons is considered
                (CC-interaction on bound nuclei do not mimic IBD)
            1.4 INFO-me: for solar average -> is a good approximation
            1.5 at the site of Super-K -> is ok, because JUNO is located at a lower geographical latitude (22.6°N)
                than Super-K (36.4°N) and therefore the flux in JUNO should be lower
                INFO-me: -> Flux is overestimated a bit, because of the lower geographical latitude of JUNO
            1.6 INFO-me: there is a difference is flux between the FLUKA results and the results of the Bartol group
                -> three main ingredients: the primary spectrum, the particle production model, and the 3-dimensional
                spherical geometrical representation of Earth and atmosphere vs. the flat geometry
                INFO-me: the primary spectrum is known to produce more neutrinos at low energy with respect to
                previous choices
            1.7 INFO-me: the systematic error on particle production in FLUKA in the whole energy region
                between pion threshold and 100 MeV is not larger than 15 % (from comparison with accelerator results)
            1.8 INFO-me: the most important uncertainties are still related to the knowledge of the primary spectrum
                and in part to the hadronic interaction models

            INFO-me: the overall uncertainty on the absolute value of the fluxes is estimated to be smaller than 25 %


        2. this data is linear interpolated to get the differential flux corresponding to the binning in E1,
        INFO-me: it is estimated, that for e1_atmo = 0 MeV flux_atmo is also 0.


        3. Oscillation is considered from the Appendix B of the paper of Fogli et al. from 2004 with title
            "Three-generation flavor transitions and decays of supernova relic neutrinos":
        3.1 The title of appendix B is "Remarks on 3-neutrino oscillations of the atmospheric neutrino background"
        3.2 pure 2-neutrino oscillation in the channel muon-neutrino -> tau-neutrino (delta m^2 = 0) is valid for
            typical atmospheric neutrino-energies (> 1 GeV), but NOT for lower energies (order of 100 MeV)
        3.3 in the appendix only some rough estimates of the relevant flavor oscillation probabilities P_ab for
            delta m^2 unequal 0 are made:

        3.4 INFO-me: See notes "Atmospheric Charged Current background", 21.11.17
        3.5 the oscillation probabilities are calculated for the values sin(theta_13)^2 = 0 AND sin(theta_13)^2 = 0.067
            (current best fit result from PDG is sin(theta_13)^2 = 0.0214)
        3.6 in the figures atmosphericCC_flux.png and atmosphericCC_spectrum.png (generated with
            atmosphericCC_neutrino_comparison.py) the results of the flux and spectrum for calculations of different
            papers are shown
        3.7 in figure atmosphericCC_flux_deviation.png the deviation of the fluxes with oscillation to the flux
            without oscillation is shown
            INFO-me: deviation to the flux without oscillation is below 7 percent
            INFO-me: deviation to the Fogli flux (with sin2_th13=0.067) is below 3 percent
        3.8 The flux and spectrum of Fogli2004 with sin(theta_13)^2=0.067 is considered here, because it describes
            the real flux (defined by sin(theta_13)^2=0.0214) best. (maybe the flux is overestimated a little bit,
            flux agrees with flux of Peres very well for energies > 50 MeV)
        INFO-me: Taking the values of Fogli2004 is a very conservative approach to include oscillation

        4. Result:
        the atmospheric CC flux is overestimated a bit because of the oscillation probabilities of Fogli2004
        (sin(theta_13)^2=0.067) and because of the lower geographical latitude of JUNO (compared to Super-K)


        :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        :param energy_visible: energy corresponding to the visible energy in the detector (np.array of float)
        :param binning_energy_visible: bin-width of the visible energy in MeV (float)
        :param crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        :param n_target: number of free protons in JUNO (float)
        :param t: exposure time in seconds (float)
        :param detection_efficiency: detection efficiency of IBD in JUNO (float)
        :param mass_proton: mass of the proton in MeV (float)
        :param mass_neutron: mass of the neutron in MeV (float)
        :param mass_positron: mass of the positron in MeV (float)

    :return spectrum_ccatmospheric: spectrum of electron-antineutrinos of CC atmospheric background (np.array of float)
            n_neutrino_ccatmospheric_vis: number of neutrinos from calculated spectrum (float9)
            theo_spectrum_ccatmospheric: Theoretical spectrum of the atmospheric CC electron-antineutrino background
            in 1/MeV (number of events as function of the electron-antineutrino energy) (np.array of float64)
            e_visible_ccatmospheric: array of visible energies in MeV (np.array of float)
            n_neutrino_ccatmospheric_theo: number of atmospheric CC electron-antineutrino events in JUNO after "time"
            (float64)
            oscillation: oscillation is considered for oscillation=1, oscillation is not considered for oscillation=0
            (integer)
            prob_e_to_e: survival probability of electron-antineutrinos (electron-antineutrinos oscillate to
            electron-antineutrinos) (float)
            prob_mu_to_e: oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos) (float)
        """

    """ Theoretical spectrum of atmospheric charged-current background: """

    # TODO-me: include the atmospheric CC flux from HONDA at JUNO site

    # INFO-me: i have checked the values of the fluxes on 06.02.2018
    # Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
    e_data = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133, 150,
                       168, 188])

    # differential flux in energy for no oscillation for electron-antineutrinos for solar average at the site
    # of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_e_bar_data = 10 ** (-4) * np.array([0., 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101.,
                                                       96.1, 83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9,
                                                       28.8, 24.9, 21.3, 18.3])
    # linear interpolation of the simulated data above to get the differential neutrino flux corresponding to energy,
    # differential flux of electron-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_e_bar = np.interp(energy_neutrino, e_data, flux_ccatmo_nu_e_bar_data)

    # differential flux in energy for no oscillation for muon-antineutrinos for solar average at the site of Super-K,
    # in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_mu_bar_data = 10 ** (-4) * np.array([0., 116., 128., 136., 150., 158., 162., 170., 196., 177., 182.,
                                                        183., 181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5,
                                                        64.0, 55.6, 47.6, 40.8])
    # linear interpolation of the simulated data above to get the differential neutrino flux corresponding to energy,
    # differential flux of muon-antineutrinos in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_ccatmo_nu_mu_bar = np.interp(energy_neutrino, e_data, flux_ccatmo_nu_mu_bar_data)

    # total flux of electron-antineutrinos at the detector with or without oscillation in MeV**(-1) * cm**(-2) * s**(-1)
    # (factors set to 1 and 0) (np.array of float):
    # INFO-me: oscillation is considered (Fogli2004 for sin(theta_13)^2 = 0.67)
    # Integer, that defines, if oscillation is considered (oscillation = 1) or not (oscillation = 0):
    oscillation = 1
    # survival probability of electron-antineutrinos (prob_e_to_e = 0.67):
    prob_e_to_e = 0.67
    # oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos) (prob_mu_to_e = 0.17):
    prob_mu_to_e = 0.17
    # total flux in 1/(MeV * cm**2 * s) (np.array of float):
    flux_total_ccatmospheric_nu_e_bar = prob_e_to_e * flux_ccatmo_nu_e_bar + prob_mu_to_e * flux_ccatmo_nu_mu_bar
    # Theoretical spectrum (in events per MeV) of electron-antineutrinos ("number of positron-events") from
    # inverse beta decay on free protons (from paper 0903.5323.pdf, equ. 64) (np.array of float):
    theo_spectrum_ccatmospheric = (flux_total_ccatmospheric_nu_e_bar * crosssection *
                                   detection_efficiency * n_target * t)

    # number of neutrino from CC atmospheric background in JUNO detector after "time":
    n_neutrino_ccatmospheric_theo = np.trapz(theo_spectrum_ccatmospheric, energy_neutrino)

    """ Spectrum of the atmospheric charged-current electron-antineutrino background in 1/MeV, 
        theoretical spectrum is convolved with gaussian distribution: """
    spectrum_ccatmospheric = convolution(energy_neutrino, energy_visible, binning_energy_visible,
                                         theo_spectrum_ccatmospheric, mass_proton, mass_neutron, mass_positron)

    # calculate the number of neutrinos from the spectrum_ccatmospheric as test (float)
    # (compare it with n_neutrino_ccatmospheric_theo to check if the convolution works well):
    n_neutrino_ccatmospheric_vis = np.trapz(spectrum_ccatmospheric, energy_visible)

    return (spectrum_ccatmospheric, n_neutrino_ccatmospheric_vis, theo_spectrum_ccatmospheric,
            n_neutrino_ccatmospheric_theo, oscillation, prob_e_to_e, prob_mu_to_e)


def ccatmospheric_background_v3(energy_neutrino, energy_visible, binning_energy_visible, crosssection, n_target,
                                t, detection_efficiency, mass_proton, mass_neutron, mass_positron):
    """ Simulate the atmospheric Charged Current electron-antineutrino background:

        ! Version 3:  !

        Convolution of the theoretical spectrum with gaussian distribution is calculated with the
        function convolution()

        The paper of Battistoni2005 'The atmospheric neutrino fluxes below 100 MeV: The FLUKA results' is described in
        detail in the my notes. In the paper the electron- and muon-antineutrino flux is simulated for energies
        from 10 MeV to 100 MeV.


    :param energy_neutrino:
    :param energy_visible:
    :param binning_energy_visible:
    :param crosssection:
    :param n_target:
    :param t:
    :param detection_efficiency:
    :param mass_proton:
    :param mass_neutron:
    :param mass_positron:
    :return:
    """

    # TODO-me: include the atmospheric CC flux from HONDA at JUNO site
    # TODO-me: Why is the flux in Julia's talk higher than mine?????

    return


def compare_4fileinputs(signal, dsnb, ccatmo, reactor, output_string):
    """
    function, which compares the input values of 4 files (signal, DSNB, CCatmo, reactor).
    If all four values are the same, the function returns the value of the signal-file.
    If one value differs from the other, the function prints a warning and returns a string.

    :param signal: value from the signal-file (float)
    :param dsnb: value from the DSNB-file (float)
    :param ccatmo: value from the CCatmo-file (float)
    :param reactor: value from the Reactor-file (float)
    :param output_string: string variable, which describes the value (e.g. 'interval_E_visible') (string)
    :return: output: either the value of the signal-file (float) or the string output_string (string)
    """

    if (signal == dsnb and ccatmo == reactor) and signal == ccatmo:
        output = signal
    else:
        output = output_string
        print("ERROR: variable {0} is not the same for the different files!".format(output))

    return output


def compare_fileinputs(input1, input2, output_string):
    """
    function to check if the input arguments are equal or not.
    :param input1: input argument
    :param input2: input argument
    :param output_string: string, which describes the input parameters (string)
    :return: either the value of one input parameter or an error message
    """
    if input1 == input2:
        output = input1
    else:
        output = output_string
        print("ERROR: variable {0} is not the same for the different files!".format(output))

    return output


def limit_annihilation_crosssection(s_90, dm_mass, j_avg, n_target, time_in_sec, epsilon_ibd,
                                    mass_neutron, mass_proton, mass_positron):
    """
    Function to calculate the 90 percent upper probability limit of the averaged self-annihilation cross-section
    times the relative velocity of the annihilating particles.

    :param s_90: 90 percent upper probability limit of the signal contribution (from output_analysis_v1.py) (float)
    :param dm_mass: Dark matter mass in MeV (float)
    :param j_avg: angular-averaged dark matter intensity over the whole Milky Way (float)
    :param n_target: number of targets in the JUNO detector, equivalent to the number of free protons (float)
    :param time_in_sec: exposure time in seconds (float)
    :param epsilon_ibd: detection efficiency of the JUNO detector for Inverse Beta Decay (float)
    :param mass_neutron: mass of the neutron in MeV (float)
    :param mass_proton: mass of the proton in MeV (float)
    :param mass_positron: mass of the positron in MeV (float)

    :return: 90% upper probability limit of the self-annihilation cross-section (float)
    """
    # Inverse Beta Decay cross-section in cm**2 (float):
    ibd_crosssection = sigma_ibd(dm_mass, mass_neutron-mass_proton, mass_positron)
    # solar radius circle in cm, 8.5 kiloparsec, 1kpc = 3.086*10**21 cm (float):
    r_solar = 8.5 * 3.086 * 10 ** 21
    # normalizing DM density, in MeV/cm**3 (float):
    rho_0 = 0.3 * 1000

    # Calculate the 90% probability limit of the annihilation cross-section (float):
    result = (6 * s_90 * dm_mass**2 /
              (j_avg * r_solar * rho_0**2 * ibd_crosssection * n_target * time_in_sec * epsilon_ibd))

    return result
