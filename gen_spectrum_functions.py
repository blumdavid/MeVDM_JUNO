""" Script defines the functions, which are used in other scripts: """

# import of the necessary packages:
import numpy as np
from math import gamma
from matplotlib import pyplot as plt


""" Define several functions used in the script: """


def double_array_entries(array_y, binwidth_y, binwidth_x):
    """
    function to take every entry of array_y and append the same value at position index + 1
    for binwidth_y/binwidth_x times
    -> afterwards array_y and array_x have same length

    Example: binwidth_y/binwidth_x = 2, array_y = [1, 2, 3, 4, 5] -> new array = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    :param array_y:
    :param binwidth_y:
    :param binwidth_x:
    :return:
    """
    # calculate multiplicity factor:
    number = int(binwidth_y / binwidth_x)
    # preallocate array:
    new_array = []
    # loop over array_y:
    for index in range(len(array_y)):
        # check if value of array_y is zero:
        if array_y[index] == 0:
            # take value corresponding to index-1 (the last entry of )
            value = array_y[index-1]
        else:
            value = array_y[index]

        # loop over number:
        for index2 in range(number):
            # append value of array_y to the new array:
            new_array.append(value)

    return new_array


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

        The same function is used for atmospheric NC events in
        /home/astro/blum/read_rootfile/atmospheric_NC_background/NC_background_functions.py

        :param corr_vis_neutrino: correlation between visible and neutrino energy,
        characterizes the visible energy in MeV (float or np.array of float)

        :return sigma/width of the gaussian distribution in no units (float or np.array of float)
        """
    # parameters to describe the energy resolution in percent (parameters from
    # Energy_reconstruction_status_review_2020Nanning_page34.pdf (reconstructed vertex and Dark Noise):
    # p0: is the leading term dominated by the photon statistics (in percent):
    p0 = 2.976
    # p1 and p2 come from detector effects such as PMT dark noise, variation of the PMT QE and the
    # reconstructed vertex smearing (in percent):
    p1 = 0.768
    p2 = 0.970
    # energy resolution defined as sigma/E_visible in percent, 3-parameter function (page 195, PhysicsReport) (float):
    energy_res = np.sqrt((p0 / np.sqrt(corr_vis_neutrino))**2 + p1**2 + (p2 / corr_vis_neutrino)**2)
    # sigma or width of the gaussian distribution in percent (float):
    sigma_resolution = energy_res * corr_vis_neutrino
    # sigma converted from percent to 'no unit' (float):
    sigma_resolution = sigma_resolution / 100
    return sigma_resolution


def generate_vis_spectrum(spectrum_theo, number_theo, e_neutrino, interval_e_neutrino, n_of_e_nu, e_visible,
                          n_of_theta, interval_theta, mass_proton, mass_positron, delta):
    """

    :param spectrum_theo: theoretical spectrum in 1/MeV (array)
    :param number_theo: number of events from theoretical spectrum (float)
    :param e_neutrino: neutrino energy array in MeV
    :param interval_e_neutrino: bin-width of e_neutrino in MeV
    :param n_of_e_nu: number of random number, that should be generated from theoretical spectrum (integer)
    :param e_visible: visible energy array in MeV
    :param n_of_theta: number of random numbers of theta, that should be generated for IBD kinematics (integer)
    :param interval_theta: bin-width of theta between 0 and 180 degree (float)
    :param mass_proton: proton mass in MeV
    :param mass_positron: positron mass in MeV
    :param delta: mass_neutron - mass_proton in MeV
    :return:
    """
    # calculate theoretical spectrum in 1/bin:
    spectrum_theo = spectrum_theo * interval_e_neutrino

    print("number of events in theo. spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
          .format(np.sum(spectrum_theo), min(e_neutrino), max(e_neutrino)))

    # normalize spectrum_theo to 1 to get probability function:
    spectrum_theo_norm = spectrum_theo / np.sum(spectrum_theo)

    print("normalized theo. spectrum = {0:.3f}".format(np.sum(spectrum_theo_norm)))

    # generate n_of_e_nu random values of E_neutrino from theoretical spectrum (array):
    array_e_neutrino = np.random.choice(e_neutrino, p=spectrum_theo_norm, size=n_of_e_nu)

    # plt.hist(array_e_neutrino, bins=np.arange(10, 120, 0.5))
    # plt.show()

    # preallocate array of visible spectrum in events/bin:
    vis_spectrum = np.zeros(len(e_visible))

    # loop over entries in array_e_neutrino:
    for index in range(len(array_e_neutrino)):

        # consider IBD kinematics to get the visible spectrum (with smeared energy) for one value of array_e_neutrino:
        array_spectrum_single, number_single = ibd_kinematics(array_e_neutrino[index], e_visible, number_theo,
                                                              mass_proton, mass_positron, delta, interval_theta,
                                                              n_of_theta)

        # check if number_single is NaN:
        if np.isnan(number_single):
            # number_single is NaN
            continue

        # normalize array_spectrum_single to 1:
        array_spectrum_single = array_spectrum_single / number_theo

        # plt.hist(array_spectrum_single, bins=np.arange(10, 120, 0.5))
        # plt.show()

        # add spectrum for one value of E_nu to vis_spectrum:
        vis_spectrum = np.add(vis_spectrum, array_spectrum_single)

    return vis_spectrum


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


def convolution_neutron_b11(energy_neutrino, energy_measured, binning_energy_measured, theo_spectrum,
                            mass_neutron, mass_positron):
    """
    Same as function convolution(), but not for IBD events,
    but for the interaction nu_e_bar + C12 -> positron + neutron + B11

    Function to convolve the theoretical spectrum with the gaussian distribution, which is defined by the correlation
    of measured energy and neutrino energy and the energy resolution of the detector.
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
    :param mass_neutron: mass of the neutron in MeV (float)
    :param mass_positron: mass of the positron in MeV (float)

    :return: spectrum_measured in 1/MeV (spectrum of the signal or background taking into account the detector
    properties (energy resolution, correlation of visible and neutrino energy of IBD)) (np.array of float)
    """
    # Preallocate the measured spectrum array (empty np.array):
    spectrum_measured = np.array([])
    # First calculate the function correlation_vis_neutrino and energy_resolution for the given
    # neutrino energy energy_neutrino (np.arrays of float):

    # interaction channel (nu_e_bar + C12 -> positron + neutron + B11;
    # most likely channel of nu_e_bar + C12 -> positron + neutron + ...):
    # positron energy in MeV (very simple kinematics (nu_e_bar + C12 -> positron + neutron + B11;
    # E_e = E_nu -(m_B11 + m_n - m_C12 + m_e)), kinetic energy of neutron is neglected.
    # To get E_vis: E_vis = E_e + mass_positron:
    corr_vis_neutrino = energy_neutrino - (11.0093 * 931.494 + mass_neutron - 12.0 * 931.494 + mass_positron)
    corr_vis_neutrino += mass_positron

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
                         t, detection_efficiency, mass_proton, mass_neutron, mass_positron, exposure_ratio_muon):
    """ Simulate the signal from neutrinos from DM annihilation in the Milky Way:

            ! Version 2: Convolution of the theoretical spectrum with gaussian distribution is calculated with the
             function convolution() !

            List of assumptions or estimations made for the calculation of the signal spectrum:
            1. neutrino flux (phi_signal) is calculated as described in paper 0710.5420
            1.1. # INFO-me: Assumption the branching ratio into neutrinos is dominant in DM self-annihilation
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
            :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

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
    n_neutrino_signal_theo = crosssection * phi_signal * n_target * t * detection_efficiency * exposure_ratio_muon

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


def darkmatter_signal_v3(mass_dm, crosssection, n_target, t, detection_efficiency, exposure_ratio_muon):
    """ Simulate the signal from neutrinos from DM annihilation in the Milky Way:

            ! Version 3: in v3, only the number of neutrino events for a given DM mass is calculated
                        -> no convolution is applied
                        -> the conversion from neutrino energy to visible energy is done afterwards
            !

            List of assumptions or estimations made for the calculation of the signal spectrum:
            1. neutrino flux (phi_signal) is calculated as described in paper 0710.5420
            1.1. # INFO-me: Assumption the branching ratio into neutrinos is dominant in DM self-annihilation
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

            2. N_neutrino is calculated as described in paper 0710.5420:
            2.1. # INFO-me: only IBD on free protons is considered


            :param mass_dm: dark matter mass in MeV (float)
            :param crosssection: IBD cross-section for the DM signal in cm**2 (float), produced with the
            function sigma_ibd()
            :param n_target: number of free protons in JUNO (float)
            :param t: exposure time in seconds (float)
            :param detection_efficiency: detection efficiency of IBD in JUNO (float)
            :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

        :return n_neutrino_signal_theo: number of electron-antineutrino events in JUNO after "time" (float64)
                energy_neutrino: neutrino energy corresponding to the DM mass in MeV (float)
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
    n_neutrino_signal_theo = crosssection * phi_signal * n_target * t * detection_efficiency * exposure_ratio_muon

    # set the neutrino energy corresponding to the DM mass in MeV:
    energy_neutrino = mass_dm

    return n_neutrino_signal_theo, energy_neutrino, sigma_anni, j_avg, phi_signal


def dsigma_dcostheta(cosinetheta, energy_pos_0, energy_pos_1, velocity_pos_1, momentum_pos_0, momentum_pos_1, sig_0,
                     gamma_factor, f_factor, g_factor, mass_proton):
    """
    differential cross-section at first order of 1/M (equ. 12)

    :param cosinetheta: cosine of angle theta between neutrino and positron direction
    :param energy_pos_0: positron energy at zeroth order of 1/M in MeV (equ. 6)
    :param energy_pos_1: positron energy at first order of 1/M in MeV (equ. 11)
    :param velocity_pos_1: positron velocity at first order of 1/M in MeV (velocity = momentum/energy)
    :param momentum_pos_0: positron momentum at zeroth order of 1/M in MeV (momentum = sqrt(energy^2 - m_pos^2))
    :param momentum_pos_1: positron momentum at first order of 1/M in MeV (momentum = sqrt(energy^2 - m_pos^2))
    :param sig_0: normalizing constant in 1/MeV^2, including the energy-independent inner radiative corrections (equ. 8)
    :param gamma_factor: gamma factor (equ. 13)
    :param f_factor: vector coupling constant
    :param g_factor: axial vector coupling constant
    :param mass_proton: proton mass in MeV
    :return:
    """
    # calculate first line of equation 12:
    first_line = (sig_0 / 2.0 * ((f_factor**2 + 3 * g_factor**2) + (f_factor**2 - g_factor**2) * velocity_pos_1
                                 * cosinetheta) * energy_pos_1 * momentum_pos_1)
    # calculate second line of equation 12:
    second_line = sig_0 / 2.0 * gamma_factor / mass_proton * energy_pos_0 * momentum_pos_0

    # calculate differential cross-section (equ. 12):
    cross_section = first_line - second_line

    return cross_section


def gamma_function(cosinetheta, energy_pos_0, velocity_pos_0, f_factor, f2_factor, g_factor, delta, mass_positron):
    """
    gamma function (equ. 13)

    :param cosinetheta: cosine of angle theta between neutrino and positron direction
    :param energy_pos_0: positron energy at zeroth order of 1/M in MeV (equ. 6)
    :param velocity_pos_0: positron velocity at zeroth order of 1/M in MeV (velocity = momentum/energy)
    :param f_factor: vector coupling constant
    :param f2_factor: anomalous nucleon iso-vector magnetic moment
    :param g_factor: axial vector coupling constant
    :param delta: mass_neutron - mass_proton in MeV
    :param mass_positron: positron mass in MeV
    :return:
    """
    first_line = (2 * (f_factor + f2_factor) * g_factor *
                  ((2 * energy_pos_0 + delta) * (1 - velocity_pos_0 * cosinetheta) - mass_positron ** 2 / energy_pos_0))

    second_line = (f_factor ** 2 + g_factor ** 2) * (delta * (1 + velocity_pos_0 * cosinetheta) + mass_positron ** 2 /
                                                     energy_pos_0)

    third_line = (f_factor ** 2 + 3 * g_factor ** 2) * ((energy_pos_0 + delta) * (1 - cosinetheta / velocity_pos_0)
                                                        - delta)

    fourth_line = (f_factor ** 2 - g_factor ** 2) * ((energy_pos_0 + delta) * (1 - cosinetheta / velocity_pos_0) -
                                                     delta) * velocity_pos_0 * cosinetheta

    gamma_factor = first_line + second_line + third_line + fourth_line

    return gamma_factor


def energy_positron_1(cosinetheta, energy_pos_0, energy_neutrino, velocity_pos_0, mass_proton, delta, mass_positron):
    """
    positron energy depending on scattering angle theta at first order in 1/M (equ. 11)

    :param cosinetheta: cosine of angle theta between neutrino and positron direction
    :param energy_pos_0: positron energy at zeroth order of 1/M in MeV (equ. 6)
    :param energy_neutrino: neutrino energy in MeV
    :param velocity_pos_0: positron velocity at zeroth order of 1/M in MeV (velocity = momentum/energy)
    :param mass_proton: proton mass in MeV
    :param delta: mass_neutron - mass_proton in MeV
    :param mass_positron: positron mass in MeV
    :return:
    """
    # calculate y_square in MeV^2:
    y_square = (delta**2 - mass_positron**2) / 2

    # calculate positron energy in MeV at first order of 1/M (equ. 11):
    energy_pos_1 = (energy_pos_0 * (1 - energy_neutrino / mass_proton * (1 - velocity_pos_0 * cosinetheta))
                    - y_square / mass_proton)

    return energy_pos_1


def ibd_kinematics(energy_neutrino, energy_visible, n_neutrino_theo, mass_proton, mass_positron, delta,
                   theta_interval, number_of_thetas):
    """
    function to calculate the visible energy of positrons for a specific neutrino energy interacting via IBD on protons.
    This function is based on script test_IBD_kinematics.py and therefore based on paper
    'angular distribution of neutron inverse beta decay' by Vogel and Beacom
    (this kinematics is also used in IBD generator of JUNO offline)

    With this function, also the angular dependency cos(theta) between neutrino and positron is considered.
    This is not done in the function convolution(), where only the average value of cos(theta) is considered

    :param energy_neutrino: neutrino energy in MeV (float)
    :param energy_visible: energy corresponding to the visible energy in MeV (np.array of float)
    :param n_neutrino_theo: theoretical number of neutrino events from DM annihilation
    :param mass_proton: proton mass in MeV
    :param mass_positron: positron mass in MeV
    :param delta: mass_neutron - mass_proton in MeV
    :param theta_interval: bin-width of the array that defines theta in degree
    :param number_of_thetas: number of random theta values that are generated

    :return:    visible_spectrum: visible spectrum of the electron-antineutrino signal from DM annihilation
                after t years for the JUNO detector in 1/bin
                n_neutrino_signal_vis: number of neutrinos calculated from spectrum_signal (float)
    """
    # set angle between neutrino and positron in degree:
    cos_theta_min = -1.0
    cos_theta_max = 1.0
    cos_theta_array = np.arange(cos_theta_min, cos_theta_max+0.01, 0.01)

    """ define constants for cross-section calculation from paper: """
    # fermi constant in 1/MeV^2:
    g_f = 1.16637 * 10**(-11)
    # cosine of theta_c:
    cos_theta_c = 0.974
    # delta_R_inner:
    delta_r_inner = 0.024
    # vector coupling constant:
    f = 1.0
    # axial vector coupling constant:
    g = 1.26
    # f2_factor: anomalous nucleon iso-vector magnetic moment:
    f_2 = 3.706
    # calculate sigma_0 (equ. 8):
    sigma_0 = g_f ** 2 * cos_theta_c ** 2 / np.pi * (1 + delta_r_inner)

    """ calculate properties ar zeroth order of 1/M: """
    # calculate positron energy in MeV at zeroth order of 1/M (equ. 6):
    e_pos_0 = energy_neutrino - delta
    # calculate positron momentum in MeV:
    p_pos_0 = np.sqrt(e_pos_0**2 - mass_positron**2)
    # calculate positron velocity in MeV:
    v_pos_0 = p_pos_0 / e_pos_0

    # preallocate array, where differential cross-section is stored in cm^2:
    array_dsig_dcos_th = []

    # generate neutrino events:
    for cos_theta in cos_theta_array:

        """ calculate positron energy (equ. 11): """
        e_pos_1 = energy_positron_1(cos_theta, e_pos_0, energy_neutrino, v_pos_0, mass_proton, delta, mass_positron)

        """ calculate IBD cross-section (equ. 12): 
        You need the differential cross-section as probability function to generate random values of theta, that are 
        distributed correctly. """
        # positron momentum O(1/M) in MeV:
        p_pos_1 = np.sqrt(e_pos_1**2 - mass_positron**2)
        # positron velocity O(1/M) in MeV:
        v_pos_1 = p_pos_1 / e_pos_1

        # calculate gamma factor (equ. 13):
        gamma_factor = gamma_function(cos_theta, e_pos_0, v_pos_0, f, f_2, g, delta, mass_positron)

        # differential cross-section at first order of 1/M (equ.12):
        d_sigma_dcos_theta = dsigma_dcostheta(cos_theta, e_pos_0, e_pos_1, v_pos_1, p_pos_0, p_pos_1,
                                              sigma_0, gamma_factor, f, g, mass_proton)

        # for higher neutrino energies (above around 70 MeV), the differential cross-section becomes negative for
        # higher angles of theta (around 160 degree) and NO random numbers can be generated from a negative probability
        # function. Therefore set the differential cross-section to zero in this case.
        # The error, you make, is not that large, because for these high energies the angular distribution is forward.
        if d_sigma_dcos_theta < 0.0:
            d_sigma_dcos_theta = 0.0

        # append d_sigma_dcos_theta to array:
        array_dsig_dcos_th.append(d_sigma_dcos_theta)

    # normalize array_dsig_dcos_th to 1 to get probability function (do not use the last bin, because the value is the
    # last bin is much higher than the other ones because of a bug):
    array_dsig_dcos_th = array_dsig_dcos_th[:-1] / np.sum(array_dsig_dcos_th[:-1])

    # generate random cos(theta) values (array):
    array_cos_theta_random = np.random.choice(cos_theta_array[:-1], p=array_dsig_dcos_th, size=number_of_thetas)

    # calculate positron energy with this randomly generated theta values (array):
    array_e_pos_1_random = energy_positron_1(array_cos_theta_random, e_pos_0, energy_neutrino, v_pos_0, mass_proton,
                                             delta, mass_positron)

    # calculate visible energy in MeV (array):
    array_e_visible = array_e_pos_1_random + mass_positron

    # get sigma from energy resolution in MeV (array):
    sigma = energy_resolution(array_e_visible)

    # smear E_visible with sigma (array):
    array_e_visible_smeared = np.random.normal(array_e_visible, sigma)

    # build histogram of array_e_visible_smeared for bins of energy_visible (units are 1/bin)
    # hist_e_visible_smeared represents the visible energy spectrum:
    hist_e_visible_smeared, bin_edges = np.histogram(array_e_visible_smeared, bins=energy_visible)

    # normalized the visible spectrum to 1:
    hist_e_visible_smeared = hist_e_visible_smeared / np.sum(hist_e_visible_smeared)

    # consider the theoretical number of neutrino events to get the visible spectrum with the correct number of events:
    visible_spectrum = hist_e_visible_smeared * n_neutrino_theo
    # append one value (0.0) to the end of visible_spectrum, because, when creating hist_e_visible_smeared only
    # energies from 10.0 to 99.5 are set (because bins defines the bin-edges and therefore energy = 100.0 is skipped):
    visible_spectrum = np.append(visible_spectrum, 0.0)

    # calculate the number of events from the visible spectrum (must be equal to n_neutrino_theo):
    n_neutrino_signal_vis = np.sum(visible_spectrum)

    return visible_spectrum, n_neutrino_signal_vis


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
                       detection_efficiency, c, mass_proton, mass_neutron, mass_positron, exposure_ratio_muon):
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
        :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

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
    # INFO-me: is correct for normal hierarchy (for inverted hierarchy see section 6.1, page 21)
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
    theo_spectrum_dsnb = crosssection * number_flux_dsnb * n_target * t * detection_efficiency * exposure_ratio_muon

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


def dsnb_background_v3(energy_neutrino, crosssection, n_target, t, detection_efficiency, c, exposure_ratio_muon):
    """ Simulate the DSNB electron-antineutrino background:

        ! Version 3: visible energy is calculated with neutrino energy and the IBD kinematics (theta is considered).

        theoretical spectrum is not convolved with gaussian distribution to get visible spectrum !

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
        :param crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        :param n_target: number of free protons in JUNO (float)
        :param t: exposure time in seconds (float)
        :param detection_efficiency: detection efficiency of IBD in JUNO (float)
        :param c: speed of light in vacuum (float)
        :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

    :return theo_spectrum_dsnb: Theoretical spectrum of the DSNB electron-antineutrino background in 1/MeV
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
    # INFO-me: is correct for normal hierarchy (for inverted hierarchy see section 6.1, page 21)
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
    theo_spectrum_dsnb = crosssection * number_flux_dsnb * n_target * t * detection_efficiency * exposure_ratio_muon

    # number of neutrinos from DSNB background in JUNO detector after "time":
    n_neutrino_dsnb_theo = np.trapz(theo_spectrum_dsnb, energy_neutrino)

    return (theo_spectrum_dsnb, n_neutrino_dsnb_theo, e_mean__nu_e_bar, beta__nu_e_bar, e_mean__nu_bar, beta__nu_bar,
            f_star)


def dsnb_background_v4(energy_neutrino, crosssection, n_target, t, detection_efficiency, exposure_ratio_muon):
    """ Simulate the DSNB electron-antineutrino background:

        ! Version 4: DSNB flux is taken from file 1705.02122_DSNB_flux_Figure_8.txt of paper 1705.02122
        (provided by Julia).


        ! visible energy is calculated with neutrino energy and the IBD kinematics (cos(theta) is considered).

        theoretical spectrum is NOT convolved with gaussian distribution to get visible spectrum !

        List of assumptions or estimations made for the calculation of the background spectrum from
        DSNB electron-antineutrinos:
        1.  simulation results for neutron star-forming and black hole-forming stellar collapses from the Garching
            group. Scenarios with different distributions of black-hole forming collapses with the progenitor mass
            are discussed, and the uncertainty on the cosmological rate of collapses is included.

        2.  Initial mass function of Salpeter group (Salpeter IMF, phi(M) ~ M^(-2.35)) is used

        3.  Star formation rate R_SF(z) is described by functional fit

        4.  three possibilities for the fraction of collapses that result in direct BH formation are considered:
            - f_BH = 0.09 (all stars with M >= 40 solar masses result in failed supernova)
            - f_BH = 0.14 (BH formation for M >= 40 solar masses and solar masses between 25 and 30)
            - f_BH = 0.27 (all stars with M >= 20 solar masses collapse into a black hole)

            f_BH = 0.27 is used, because it leads to the largest DSNB background. Also both other f_BH could be used to
            get the sensitivity on f_BH.
            Important: "As a cautionary note, we stress that the mechanism of collapse into a black hole is still not
            fully understood, therefore our results based on the scenarios above have the character of illustration
            only."

        5.  for the diffuse flux simulation z_max = 2 is used. It is found that in the energy range of our interest,
            more than 70 % of the flux comes from local supernova explosions at z < 1, while the high-redshift (z > 2)
            supernova contribution is very small

        6.  Oscillation is considered (p_bar = 0.68, 68 percent of the electron-antineutrinos survive and 32 percent
            of the muon-/tau-antineutrinos appear as electron-antineutrinos at the Earth
            INFO-me: normal hierarchy is assumed
            TODO-me: for inverted hierarchy the neutrino flux at Earth differs from the flux for normal hierarchy
            Neutrino energies from 0 to 50 MeV. Fixed star formation rate to R_cc(0) = 1.25 * 10^(-4) yr^(-1) Mpc^(-3)

        7. only electron-antineutrinos interacting with free protons are considered (Inverse Beta Decay),
           dominant up to ~80 MeV

        :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        :param crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        :param n_target: number of free protons in JUNO (float)
        :param t: exposure time in seconds (float)
        :param detection_efficiency: detection efficiency of IBD in JUNO (float)
        :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

    :return:
            theo_spectrum_dsnb: Theoretical spectrum of the DSNB electron-antineutrino background in 1/MeV
            (number of events as function of the electron-antineutrino energy) (np.array of float64)
            n_neutrino_dsnb_theo: number of DSNB electron-antineutrinos from theoretical spectrum (float64)
    """
    # Neutrino energy in MeV corresponding to the data points of the DSNB flux:
    energy_data = np.arange(10, 50.05, 0.05)
    energy_data_2 = np.array([50.1])
    # add data point of 0.0 / (cm^2 * MeV * s) for 50.1 MeV:
    energy_data = np.append(energy_data, energy_data_2)

    # DSNB flux of electron-antineutrinos for f_BH = 0.27, z_max = 2 and fixed star formation rate in 1/(cm^2 MeV s):
    """
    flux_data = np.array([0.62774, 0.622145, 0.616551, 0.610956, 0.605361, 0.599767, 0.594172, 0.588577, 0.582983,
                          0.577388, 0.571794, 0.5666, 0.561406, 0.556212, 0.551018, 0.545824, 0.54063, 0.535437,
                          0.530243, 0.525049, 0.519855, 0.515062, 0.51027, 0.505477, 0.500685, 0.495892, 0.4911,
                          0.486307, 0.481515, 0.476722, 0.47193, 0.46753, 0.463129, 0.458729, 0.454329,
                          0.449929, 0.445529, 0.441129, 0.436729, 0.432329, 0.427929, 0.423904, 0.41988, 0.415855,
                          0.411831, 0.407807, 0.403782, 0.399758, 0.395733, 0.391709, 0.387685, 0.384016, 0.380346,
                          0.376677, 0.373008, 0.369339, 0.36567, 0.362001, 0.358332, 0.354663, 0.350994, 0.347657,
                          0.34432, 0.340984, 0.337647, 0.33431, 0.330973, 0.327636, 0.3243, 0.320963, 0.317626,
                          0.314598, 0.311569, 0.308541, 0.305513, 0.302484, 0.299456, 0.296428, 0.293399, 0.290371,
                          0.287343, 0.284599, 0.281854, 0.27911, 0.276366, 0.273622, 0.270878, 0.268133, 0.265389,
                          0.262645, 0.259901, 0.257417, 0.254934, 0.25245, 0.249967, 0.247483, 0.244999, 0.242516,
                          0.240032, 0.237549, 0.235065, 0.23282, 0.230574, 0.228328, 0.226083, 0.223837, 0.221591,
                          0.219346, 0.2171, 0.214855, 0.212609, 0.21058, 0.208551, 0.206522, 0.204493, 0.202464,
                          0.200435, 0.198406, 0.196377, 0.194348, 0.192319, 0.190487, 0.188655, 0.186822, 0.18499,
                          0.183158, 0.181326, 0.179493, 0.177661, 0.175829, 0.173996, 0.172342, 0.170688, 0.169034,
                          0.16738, 0.165726, 0.164072, 0.162418, 0.160764, 0.15911, 0.157456, 0.155963, 0.15447,
                          0.152977, 0.151484, 0.149991, 0.148499, 0.147006, 0.145513, 0.14402, 0.142527, 0.14118,
                          0.139833, 0.138486, 0.137138, 0.135791, 0.134444, 0.133097, 0.13175, 0.130402, 0.129055,
                          0.127839, 0.126624, 0.125408, 0.124192, 0.122976, 0.121761, 0.120545, 0.119329, 0.118113,
                          0.116897, 0.1158, 0.114703, 0.113606, 0.112509, 0.111411, 0.110314, 0.109217, 0.10812,
                          0.107022, 0.105925, 0.104935, 0.103944, 0.102954, 0.101964, 0.100973, 0.0999828, 0.0989924,
                          0.098002, 0.0970116, 0.0960212, 0.0951271, 0.094233, 0.0933388, 0.0924447, 0.0915506,
                          0.0906564, 0.0897623, 0.0888682, 0.087974, 0.0870799, 0.0862724, 0.085465, 0.0846575,
                          0.08385, 0.0830426, 0.0822351, 0.0814276, 0.0806202, 0.0798127, 0.0790052, 0.0782758,
                          0.0775464, 0.0768169, 0.0760875, 0.0753581, 0.0746286, 0.0738992, 0.0731698, 0.0724403,
                          0.0717109, 0.0710518, 0.0703926, 0.0697335, 0.0690743, 0.0684152, 0.067756, 0.0670969,
                          0.0664377, 0.0657786, 0.0651194, 0.0645236, 0.0639278, 0.063332, 0.0627362, 0.0621403,
                          0.0615445, 0.0609487, 0.0603529, 0.059757, 0.0591612, 0.0586224, 0.0580837, 0.0575449,
                          0.0570061, 0.0564673, 0.0559285, 0.0553897, 0.0548509, 0.0543121, 0.0537734, 0.053286,
                          0.0527986, 0.0523112, 0.0518238, 0.0513364, 0.050849, 0.0503616, 0.0498742, 0.0493868,
                          0.0488994, 0.0484583, 0.0480172, 0.0475761, 0.0471351, 0.046694, 0.0462529, 0.0458118,
                          0.0453707, 0.0449296, 0.0444885, 0.0440891, 0.0436897, 0.0432903, 0.0428909, 0.0424915,
                          0.0420922, 0.0416928, 0.0412934, 0.040894, 0.0404946, 0.0401328, 0.0397711, 0.0394093,
                          0.0390475, 0.0386857, 0.038324, 0.0379622, 0.0376004, 0.0372386,
                          0.0368768, 0.036549, 0.0362211, 0.0358933, 0.0355654, 0.0352375,  0.0349097, 0.0345818,
                          0.034254, 0.0339261, 0.0335982, 0.033301, 0.0330037, 0.0327065, 0.0324092, 0.0321119,
                          0.0318147, 0.0315174, 0.0312202, 0.0309229, 0.0306257, 0.030356, 0.0300864, 0.0298167,
                          0.0295471, 0.0292775, 0.0290078, 0.0287382, 0.0284685, 0.0281989, 0.0279293, 0.0276845,
                          0.0274398, 0.0271951, 0.0269504, 0.0267057, 0.026461, 0.0262163, 0.0259716, 0.0257269,
                          0.0254822, 0.02526, 0.0250379, 0.0248157, 0.0245935, 0.0243713, 0.0241491, 0.0239269,
                          0.0237048, 0.0234826, 0.0232604, 0.0230586, 0.0228567, 0.0226549,
                          0.0224531, 0.0222513, 0.0220494, 0.0218476, 0.0216458, 0.0214439,
                          0.0212421, 0.0210587, 0.0208753, 0.0206918, 0.0205084, 0.020325,
                          0.0201415, 0.0199581, 0.0197747, 0.0195913, 0.0194078, 0.019241,
                          0.0190743, 0.0189075, 0.0187407, 0.0185739, 0.0184071, 0.0182404, 0.0180736, 0.0179068,
                          0.01774, 0.0175883, 0.0174366, 0.0172848, 0.0171331, 0.0169814, 0.0168297, 0.016678,
                          0.0165262, 0.0163745, 0.0162228, 0.0160847, 0.0159466, 0.0158086, 0.0156705, 0.0155324,
                          0.0153943, 0.0152562, 0.0151181, 0.0149801, 0.014842, 0.0147162, 0.0145905, 0.0144648,
                          0.014339, 0.0142133, 0.0140876, 0.0139618, 0.0138361, 0.0137104, 0.0135846, 0.0134701,
                          0.0133556, 0.013241, 0.0131265, 0.0130119, 0.0128974, 0.0127828, 0.0126683, 0.0125538,
                          0.0124392, 0.0123348, 0.0122304, 0.012126, 0.0120216, 0.0119172,
                          0.0118128, 0.0117084, 0.011604, 0.0114996, 0.0113952, 0.0113, 0.0112048, 0.0111096, 0.0110144,
                          0.0109192, 0.010824, 0.0107288, 0.0106336, 0.0105384, 0.0104432, 0.0103564, 0.0102695,
                          0.0101827, 0.0100958, 0.010009, 0.00992212, 0.00983528, 0.00974843, 0.00966158, 0.00957474,
                          0.00949548, 0.00941622, 0.00933696, 0.0092577, 0.00917844, 0.00909918, 0.00901992, 0.00894066,
                          0.0088614, 0.00878214, 0.00870978, 0.00863741, 0.00856504, 0.00849267, 0.0084203, 0.00834793,
                          0.00827556, 0.00820319, 0.00813082, 0.00805845, 0.00799235, 0.00792624, 0.00786014,
                          0.00779403, 0.00772792, 0.00766182, 0.00759571, 0.00752961, 0.0074635, 0.00739739, 0.00733698,
                          0.00727656, 0.00721615, 0.00715573, 0.00709531, 0.0070349, 0.00697448, 0.00691407, 0.00685365,
                          0.00679323, 0.006738, 0.00668276, 0.00662752, 0.00657228, 0.00651704, 0.0064618, 0.00640656,
                          0.00635132, 0.00629609, 0.00624085, 0.00619032, 0.00613979, 0.00608926, 0.00603873, 0.0059882,
                          0.00593767, 0.00588714, 0.00583661, 0.00578608, 0.00573555, 0.00568931, 0.00564307,
                          0.00559683, 0.00555059, 0.00550435, 0.00545811, 0.00541186, 0.00536562, 0.00531938,
                          0.00527314, 0.00523081, 0.00518848, 0.00514614, 0.00510381, 0.00506148, 0.00501915,
                          0.00497681, 0.00493448, 0.00489215, 0.00484981, 0.00481104, 0.00477227, 0.0047335, 0.00469473,
                          0.00465596, 0.00461719, 0.00457842, 0.00453965, 0.00450088, 0.00446211, 0.00442658,
                          0.00439106, 0.00435554, 0.00432001, 0.00428449, 0.00424897, 0.00421344, 0.00417792, 0.0041424,
                          0.00410687, 0.00407431, 0.00404175, 0.00400919, 0.00397663, 0.00394407, 0.0039115, 0.00387894,
                          0.00384638, 0.00381382, 0.00378126, 0.0037514, 0.00372154, 0.00369168, 0.00366182, 0.00363196,
                          0.0036021, 0.00357224, 0.00354238, 0.00351252, 0.00348266, 0.00345526, 0.00342787, 0.00340048,
                          0.00337308, 0.00334569, 0.0033183, 0.00329091, 0.00326351, 0.00323612, 0.00320873, 0.00318359,
                          0.00315845, 0.00313331, 0.00310817, 0.00308303, 0.00305789, 0.00303275, 0.00300761,
                          0.00298246, 0.00295732, 0.00293424, 0.00291116, 0.00288808, 0.002865, 0.00284192, 0.00281884,
                          0.00279576, 0.00277268, 0.0027496, 0.00272652, 0.00270532, 0.00268412, 0.00266292, 0.00264172,
                          0.00262052, 0.00259932, 0.00257812, 0.00255692, 0.00253572, 0.00251452, 0.00249505,
                          0.00247557, 0.00245609, 0.00243661, 0.00241714, 0.00239766, 0.00237818, 0.0023587, 0.00233923,
                          0.00231975, 0.00230185, 0.00228394, 0.00226604, 0.00224814, 0.00223023, 0.00221233,
                          0.00219443, 0.00217652, 0.00215862, 0.00214072, 0.00212426, 0.0021078, 0.00209134, 0.00207487,
                          0.00205841, 0.00204195, 0.00202549, 0.00200903, 0.00199257, 0.00197611, 0.00196097,
                          0.00194583, 0.00193068, 0.00191554, 0.0019004, 0.00188526, 0.00187012, 0.00185498, 0.00183984,
                          0.00182469, 0.00181076, 0.00179683, 0.0017829, 0.00176897, 0.00175504, 0.0017411, 0.00172717,
                          0.00171324, 0.00169931, 0.00168538, 0.00167255, 0.00165973, 0.00164691, 0.00163408,
                          0.00162126, 0.00160844, 0.00159561, 0.00158279, 0.00156996, 0.00155714, 0.00154533,
                          0.00153353, 0.00152172, 0.00150991, 0.0014981, 0.0014863, 0.00147449, 0.00146268, 0.00145088,
                          0.00143907, 0.00142819, 0.00141732, 0.00140644, 0.00139556, 0.00138469, 0.00137381,
                          0.00136294, 0.00135206, 0.00134118, 0.00133031, 0.00132029, 0.00131027, 0.00130025,
                          0.00129023, 0.0012802, 0.00127018, 0.00126016, 0.00125014, 0.00124012, 0.0012301, 0.00122086,
                          0.00121163, 0.00120239, 0.00119316, 0.00118392, 0.00117468, 0.00116545,
                          0.00115621, 0.00114698, 0.00113774, 0.00112923, 0.00112071, 0.0011122, 0.00110368, 0.00109517,
                          0.00108665, 0.00107814, 0.00106962, 0.00106111, 0.00105259, 0.00104474, 0.00103689,
                          0.00102904, 0.00102118, 0.00101333, 0.00100548, 0.000997626, 0.000989774, 0.000981921,
                          0.000974069, 0.000966825, 0.000959581, 0.000952337, 0.000945093, 0.000937849, 0.000930605,
                          0.000923361, 0.000916117, 0.000908873, 0.000901629, 0.000894944, 0.000888258, 0.000881573,
                          0.000874888, 0.000868202, 0.000861517, 0.000854831, 0.000848146, 0.00084146, 0.000834775,
                          0.000828603, 0.000822432, 0.000816261, 0.000810089, 0.000803918, 0.000797747, 0.000791575,
                          0.000785404, 0.000779233, 0.000773061, 0.000767363, 0.000761665, 0.000755966, 0.000750268,
                          0.000744569, 0.000738871, 0.000733173, 0.000727474, 0.000721776, 0.000716078, 0.000710815,
                          0.000705552, 0.00070029, 0.000695027, 0.000689764, 0.000684502, 0.000679239, 0.000673976,
                          0.000668714, 0.000663451, 0.00065859, 0.000653728, 0.000648867, 0.000644005, 0.000639144,
                          0.000634282, 0.00062942, 0.000624559, 0.000619697, 0.000614836, 0.000610343, 0.000605851,
                          0.000601358, 0.000596866, 0.000592373, 0.000587881, 0.000583388, 0.000578896, 0.000574404,
                          0.000569911, 0.000565758, 0.000561604, 0.000557451, 0.000553298, 0.000549144, 0.000544991,
                          0.000540837, 0.000536684, 0.000532531, 0.000528377, 0.0])
    """

    # """
    # DSNB flux of electron-antineutrinos for f_BH = 0.14 z_max = 2 and fixed star formation rate in 1/(cm^2 MeV s):
    flux_data = np.array([0.649048, 0.643014, 0.63698, 0.630946, 0.624912, 0.618878, 0.612844, 0.60681, 0.600776,
                          0.594742, 0.588708, 0.583129, 0.57755, 0.571972, 0.566393, 0.560814, 0.555235, 0.549657,
                          0.544078, 0.538499, 0.53292, 0.527794, 0.522667, 0.517541, 0.512414, 0.507288, 0.502161,
                          0.497035, 0.491908, 0.486782, 0.481655, 0.476968, 0.472281, 0.467594, 0.462907, 0.45822,
                          0.453533, 0.448846, 0.444158, 0.439471, 0.434784, 0.430515, 0.426247, 0.421978, 0.417709,
                          0.41344, 0.409171, 0.404903, 0.400634, 0.396365, 0.392096, 0.388221, 0.384346, 0.380471,
                          0.376595, 0.37272, 0.368845, 0.36497, 0.361095, 0.357219, 0.353344, 0.349835, 0.346326,
                          0.342817, 0.339308, 0.335799, 0.33229, 0.328781, 0.325272, 0.321763, 0.318254, 0.315083,
                          0.311913, 0.308742, 0.305571, 0.302401, 0.29923, 0.296059, 0.292889, 0.289718, 0.286548,
                          0.283687, 0.280827, 0.277966, 0.275106, 0.272245, 0.269385, 0.266524, 0.263664, 0.260804,
                          0.257943, 0.255366, 0.252789, 0.250212, 0.247635, 0.245058, 0.24248, 0.239903, 0.237326,
                          0.234749, 0.232172, 0.229852, 0.227532, 0.225213, 0.222893, 0.220573, 0.218254, 0.215934,
                          0.213614, 0.211294, 0.208975, 0.206888, 0.204802, 0.202716, 0.200629, 0.198543, 0.196457,
                          0.19437, 0.192284, 0.190198, 0.188111, 0.186236, 0.18436, 0.182485, 0.180609, 0.178734,
                          0.176858, 0.174983, 0.173107, 0.171232, 0.169356, 0.167671, 0.165986, 0.164301, 0.162615,
                          0.16093, 0.159245, 0.15756, 0.155875, 0.154189, 0.152504, 0.15099, 0.149476, 0.147962,
                          0.146448, 0.144934, 0.143421, 0.141907, 0.140393, 0.138879, 0.137365, 0.136005, 0.134645,
                          0.133285, 0.131925, 0.130566, 0.129206, 0.127846, 0.126486, 0.125126, 0.123766, 0.122545,
                          0.121323, 0.120102, 0.11888, 0.117659, 0.116438, 0.115216, 0.113995, 0.112773, 0.111552,
                          0.110455, 0.109357, 0.10826, 0.107163, 0.106066, 0.104969, 0.103871, 0.102774, 0.101677,
                          0.10058, 0.0995942, 0.0986085, 0.0976228, 0.0966371, 0.0956514, 0.0946657, 0.09368, 0.0926943,
                          0.0917086, 0.0907229, 0.0898372, 0.0889515, 0.0880658, 0.0871801, 0.0862944, 0.0854087,
                          0.084523, 0.0836373, 0.0827516, 0.0818659, 0.0810698, 0.0802738, 0.0794777, 0.0786816,
                          0.0778855, 0.0770895, 0.0762934, 0.0754973, 0.0747012, 0.0739052, 0.0731894, 0.0724737,
                          0.0717579, 0.0710422, 0.0703265, 0.0696107, 0.068895, 0.0681792, 0.0674635, 0.0667478,
                          0.066104, 0.0654603, 0.0648166, 0.0641729, 0.0635292, 0.0628855, 0.0622417, 0.061598,
                          0.0609543, 0.0603106, 0.0597315, 0.0591524, 0.0585732, 0.0579941, 0.057415, 0.0568359,
                          0.0562568, 0.0556776, 0.0550985, 0.0545194, 0.0539982, 0.053477, 0.0529558, 0.0524346,
                          0.0519134, 0.0513922, 0.050871, 0.0503498, 0.0498286, 0.0493074, 0.0488382, 0.0483689,
                          0.0478997, 0.0474305, 0.0469612, 0.046492, 0.0460227, 0.0455535, 0.0450843, 0.044615,
                          0.0441924, 0.0437697, 0.0433471, 0.0429244, 0.0425018, 0.0420791, 0.0416565, 0.0412338,
                          0.0408112, 0.0403885, 0.0400077, 0.0396268, 0.0392459, 0.0388651, 0.0384842, 0.0381034,
                          0.0377225, 0.0373416, 0.0369608, 0.0365799, 0.0362366, 0.0358932, 0.0355499, 0.0352065,
                          0.0348632, 0.0345198, 0.0341765, 0.0338331, 0.0334898, 0.0331464, 0.0328367, 0.0325271,
                          0.0322174, 0.0319077, 0.031598, 0.0312883, 0.0309787, 0.030669, 0.0303593, 0.0300496,
                          0.0297702, 0.0294908, 0.0292114, 0.0289319, 0.0286525, 0.0283731, 0.0280936, 0.0278142,
                          0.0275348, 0.0272553, 0.0270031, 0.0267508, 0.0264986, 0.0262463, 0.0259941, 0.0257418,
                          0.0254896, 0.0252373, 0.0249851, 0.0247328, 0.024505, 0.0242772, 0.0240493, 0.0238215,
                          0.0235937, 0.0233659, 0.023138, 0.0229102, 0.0226824, 0.0224545, 0.0222487, 0.0220428,
                          0.0218369, 0.0216311, 0.0214252, 0.0212193, 0.0210134, 0.0208076, 0.0206017, 0.0203958,
                          0.0202097, 0.0200236, 0.0198375, 0.0196514, 0.0194652, 0.0192791, 0.019093, 0.0189069,
                          0.0187208, 0.0185347, 0.0183663, 0.018198, 0.0180296, 0.0178613, 0.017693, 0.0175246,
                          0.0173563, 0.017188, 0.0170196, 0.0168513, 0.016699, 0.0165466, 0.0163943, 0.016242,
                          0.0160897, 0.0159373, 0.015785, 0.0156327, 0.0154804, 0.015328, 0.0151901, 0.0150522,
                          0.0149143, 0.0147764, 0.0146385, 0.0145006, 0.0143627, 0.0142248, 0.0140869, 0.013949,
                          0.0138241, 0.0136992, 0.0135743, 0.0134494, 0.0133245, 0.0131996, 0.0130747, 0.0129498,
                          0.0128249, 0.0127, 0.0125868, 0.0124736, 0.0123604, 0.0122472, 0.012134, 0.0120208, 0.0119076,
                          0.0117945, 0.0116813, 0.0115681, 0.0114655, 0.0113628, 0.0112602, 0.0111576, 0.011055,
                          0.0109524, 0.0108497, 0.0107471, 0.0106445, 0.0105419, 0.0104488, 0.0103557, 0.0102626,
                          0.0101695, 0.0100765, 0.00998337, 0.00989028, 0.0097972, 0.00970411, 0.00961103, 0.00952655,
                          0.00944208, 0.00935761, 0.00927314, 0.00918866, 0.00910419, 0.00901972, 0.00893525,
                          0.00885077, 0.0087663, 0.00868961, 0.00861292, 0.00853623, 0.00845953, 0.00838284, 0.00830615,
                          0.00822946, 0.00815277, 0.00807607, 0.00799938, 0.00792973, 0.00786007, 0.00779042,
                          0.00772076, 0.0076511, 0.00758145, 0.00751179, 0.00744214, 0.00737248, 0.00730283, 0.00723953,
                          0.00717624, 0.00711294, 0.00704965, 0.00698635, 0.00692306, 0.00685976, 0.00679647,
                          0.00673317, 0.00666987, 0.00661233, 0.00655479, 0.00649725, 0.00643971, 0.00638217,
                          0.00632463, 0.00626709, 0.00620955, 0.00615201, 0.00609447, 0.00604214, 0.0059898, 0.00593747,
                          0.00588513, 0.00583279, 0.00578046, 0.00572812, 0.00567579, 0.00562345, 0.00557111,
                          0.00552349, 0.00547587, 0.00542824, 0.00538062, 0.005333, 0.00528537, 0.00523775, 0.00519013,
                          0.00514251, 0.00509488, 0.00505152, 0.00500817, 0.00496481, 0.00492145, 0.00487809,
                          0.00483473, 0.00479138, 0.00474802, 0.00470466, 0.0046613, 0.00462181, 0.00458232, 0.00454283,
                          0.00450335, 0.00446386, 0.00442437, 0.00438488, 0.00434539, 0.0043059, 0.00426641, 0.00423043,
                          0.00419445, 0.00415846, 0.00412248, 0.0040865, 0.00405052, 0.00401454, 0.00397855, 0.00394257,
                          0.00390659, 0.00387379, 0.00384099, 0.00380819, 0.00377539, 0.0037426, 0.0037098, 0.003677,
                          0.0036442, 0.0036114, 0.0035786, 0.00354869, 0.00351878, 0.00348887, 0.00345896, 0.00342905,
                          0.00339914, 0.00336923, 0.00333931, 0.0033094, 0.00327949, 0.0032522, 0.00322491, 0.00319762,
                          0.00317034, 0.00314305, 0.00311576, 0.00308847, 0.00306118, 0.00303389, 0.0030066, 0.00298169,
                          0.00295678, 0.00293187, 0.00290696, 0.00288205, 0.00285714, 0.00283223, 0.00280732, 0.0027824,
                          0.00275749, 0.00273475, 0.002712, 0.00268925, 0.00266651, 0.00264376, 0.00262101, 0.00259826,
                          0.00257552, 0.00255277, 0.00253002, 0.00250924, 0.00248846, 0.00246768, 0.0024469, 0.00242612,
                          0.00240533, 0.00238455, 0.00236377, 0.00234299, 0.00232221, 0.00230322, 0.00228423,
                          0.00226523, 0.00224624, 0.00222725, 0.00220826, 0.00218927, 0.00217027, 0.00215128,
                          0.00213229, 0.00211492, 0.00209756, 0.00208019, 0.00206283, 0.00204546, 0.0020281, 0.00201073,
                          0.00199337, 0.001976, 0.00195863, 0.00194275, 0.00192687, 0.00191098, 0.0018951, 0.00187922,
                          0.00186333, 0.00184745, 0.00183157, 0.00181568, 0.0017998, 0.00178526, 0.00177073, 0.00175619,
                          0.00174166, 0.00172712, 0.00171259, 0.00169805, 0.00168352, 0.00166898, 0.00165445,
                          0.00164114, 0.00162784, 0.00161453, 0.00160123, 0.00158792, 0.00157462, 0.00156131, 0.001548,
                          0.0015347, 0.00152139, 0.00150921, 0.00149702, 0.00148484, 0.00147265, 0.00146047, 0.00144828,
                          0.00143609, 0.00142391, 0.00141172, 0.00139954, 0.00138837, 0.00137721, 0.00136605,
                          0.00135488, 0.00134372, 0.00133255, 0.00132139, 0.00131023, 0.00129906, 0.0012879, 0.00127767,
                          0.00126743, 0.0012572, 0.00124697, 0.00123674, 0.00122651, 0.00121627, 0.00120604, 0.00119581,
                          0.00118558, 0.0011762, 0.00116682, 0.00115743, 0.00114805, 0.00113867, 0.00112929, 0.00111991,
                          0.00111053, 0.00110115, 0.00109176, 0.00108316, 0.00107455, 0.00106595, 0.00105734,
                          0.00104874, 0.00104013, 0.00103153, 0.00102292, 0.00101432, 0.00100571, 0.000997815,
                          0.000989919, 0.000982023, 0.000974127, 0.000966231, 0.000958335, 0.000950439, 0.000942543,
                          0.000934647, 0.000926751, 0.000919503, 0.000912255, 0.000905007, 0.000897759, 0.000890511,
                          0.000883263, 0.000876015, 0.000868767, 0.000861519, 0.000854271, 0.000847616, 0.000840961,
                          0.000834306, 0.000827651, 0.000820996, 0.000814341, 0.000807686, 0.000801031, 0.000794376,
                          0.000787721, 0.000781608, 0.000775496, 0.000769383, 0.00076327, 0.000757158, 0.000751045,
                          0.000744933, 0.00073882, 0.000732707, 0.000726595, 0.000720978, 0.000715361, 0.000709744,
                          0.000704127, 0.00069851, 0.000692893, 0.000687277, 0.00068166, 0.000676043, 0.000670426,
                          0.000665262, 0.000660098, 0.000654933, 0.000649769, 0.000644605, 0.000639441, 0.000634276,
                          0.000629112, 0.000623948, 0.000618784, 0.000614035, 0.000609285, 0.000604536, 0.000599787,
                          0.000595038, 0.000590289, 0.00058554, 0.000580791, 0.000576042, 0.000571293, 0.000566924,
                          0.000562555, 0.000558187, 0.000553818, 0.000549449, 0.00054508, 0.000540712, 0.000536343,
                          0.000531974, 0.000527606, 0.000523586, 0.000519566, 0.000515547, 0.000511527, 0.000507507,
                          0.000503488, 0.000499468, 0.000495449, 0.000491429, 0.000487409, 0.00048371, 0.00048001,
                          0.00047631, 0.000472611, 0.0000468911, 0.000465211, 0.000461512, 0.000457812, 0.000454113,
                          0.000450413, 0.000447006, 0.0004436, 0.000440193, 0.000436787, 0.00043338, 0.000429974,
                          0.000426567, 0.00042316, 0.000419754, 0.000416347, 0.000413209, 0.00041007, 0.000406932,
                          0.000403793, 0.000400655, 0.000397516, 0.000394377, 0.000391239, 0.0003881, 0.000384962, 0.0])
    # """

    # interpolate DSNB flux for neutrino energies from 10 to 100 MeV (flux in 1 / (cm^2 * MeV * s)):
    flux_dsnb = np.interp(energy_neutrino, energy_data, flux_data)

    # Theoretical spectrum of DSNB neutrino events in JUNO after "time" years in 1/MeV (np.array of float64):
    theo_spectrum_dsnb = crosssection * flux_dsnb * n_target * t * detection_efficiency * exposure_ratio_muon

    # number of neutrinos from DSNB background in JUNO detector after "time":
    n_neutrino_dsnb_theo = np.trapz(theo_spectrum_dsnb, energy_neutrino)

    return theo_spectrum_dsnb, n_neutrino_dsnb_theo


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


def reactor_background_v2(energy_neutrino, energy_visible, binning_energy_visible, crosssection, n_target, t_sec,
                          detection_efficiency, mass_proton, mass_neutron, mass_positron, exposure_ratio_muon):
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

        # TODO-me: the reactor flux and its uncertainties become interesting for DM masses below 20 MeV

        :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        :param energy_visible: energy corresponding to the visible energy in MeV (np.array of float)
        :param binning_energy_visible: bin-width of the visible energy in MeV (float)
        :param crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        :param n_target: number of free protons in JUNO (float)
        :param t_sec: exposure time in seconds (float)
        :param detection_efficiency: detection efficiency of IBD in JUNO (float)
        :param mass_proton: mass of the proton in MeV (float)
        :param mass_neutron: mass of the neutron in MeV (float)
        :param mass_positron: mass of the positron in MeV (float)
        :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

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
    # electron-antineutrino flux in units of electron-antineutrino/(MeV * s) (np.array of float):
    flux_reactor = spec2_reactor * power_th

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
                             n_target * t_sec * prob_oscillation_nh * exposure_ratio_muon)

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


def reactor_background_v3(energy_neutrino, crosssection, n_target, t_sec, detection_efficiency, exposure_ratio_muon):
    """ Simulate the reactor electron-antineutrino background in JUNO:

        ! Version 3:
        - in this function only the theoretical spectrum is calculated! IBD kinematics are considered afterwards!

        - visible energy is calculated with neutrino energy and the IBD kinematics (theta is considered).

        - theoretical spectrum is not convolved with gaussian distribution to get visible spectrum !

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

        # TODO-me: the reactor flux and its uncertainties become interesting for DM masses below 20 MeV

        :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
        :param crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
        :param n_target: number of free protons in JUNO (float)
        :param t_sec: exposure time in seconds (float)
        :param detection_efficiency: detection efficiency of IBD in JUNO (float)
        :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

        :return: theo_spectrum_reactor: Theoretical spectrum of the reactor electron-antineutrino background
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
    # electron-antineutrino flux in units of electron-antineutrino/(MeV * s) (np.array of float):
    flux_reactor = spec2_reactor * power_th

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
                             n_target * t_sec * prob_oscillation_nh * exposure_ratio_muon)

    # number of neutrinos from reactor background in JUNO detector after "time" (float):
    n_neutrino_reactor_theo = np.trapz(theo_spectrum_reactor, energy_neutrino)

    return (theo_spectrum_reactor, n_neutrino_reactor_theo, power_th, fraction235_u, fraction238_u, fraction239_pu,
            fraction241_pu, l_m)


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
                                t, detection_efficiency, mass_proton, mass_neutron, mass_positron, exposure_ratio_muon):
    """ Simulate the atmospheric Charged Current electron-antineutrino background:

        ! Version 3: the atmospheric CC neutrino flux is based on the simulations of HONDA for the JUNO location!

        Results of the HONDA simulation are based on the paper of Honda2015: 'Atmospheric neutrino flux calculation
        using the NRLMSISE-00 atmospheric model')

        The HONDA flux is simulated for JUNO site only for energies above 100 MeV. In the range below 100 MeV the
        spectral shape of the FLUKA simulation in the range from 10 to 100 MeV is used, but normalized to the flux
        from the HONDA simulation.

        Detailed information about the calculations are in the python script 'atmospheric_flux.py'.

        Convolution of the theoretical spectrum with gaussian distribution is calculated with the
        function convolution()

        The paper of Battistoni2005 'The atmospheric neutrino fluxes below 100 MeV: The FLUKA results' is described in
        detail in the my notes. In the paper the electron- and muon-antineutrino flux is simulated for energies
        from 10 MeV to 100 MeV.

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
    :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

    :return:spectrum_ccatmospheric: spectrum of electron-antineutrinos of CC atmospheric background (np.array of float)
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

    # INFO-me: in the notes "Vergleich Atmospheric Flux aus Paper mit Julia's Werten", the flux of Julia was higher
    # than in the data of the paper. BUT in Julia's talk in Strasbourg, the spectrum was lower than in the previous
    # spectrum. Now the spectrum of Julia and from atmospheric_flux.py are more similar.

    """ Theoretical spectrum of atmospheric charged-current background: """

    """ Results of the FLUKA simulation (from the paper of Battistoni2005 'The atmospheric neutrino fluxes below 
    100 MeV: The FLUKA results'): """
    # Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
    energy_fluka = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133,
                             150, 168, 188, 211, 237, 266, 299, 335, 376, 422, 473, 531, 596, 668, 750, 841, 944])

    # differential flux from FLUKA in energy for no oscillation for electron-antineutrinos for solar average at the site
    # of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float).
    # Assumption: for energy = 0 MeV, the flux is also 0!
    flux_nuebar_fluka = 10 ** (-4) * np.array([0, 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                               83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9, 21.3,
                                               18.3, 15.4, 12.9, 10.6, 8.80, 7.13, 5.75, 4.60, 3.68, 2.88, 2.28,
                                               1.87, 1.37, 1.06, 0.800])

    # differential flux from FLUKA in energy for no oscillation for muon-antineutrinos for solar average at the site of
    # Super-K, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float).
    # Assumption: for energy = 0 MeV, the flux is also 0!
    flux_numubar_fluka = 10 ** (-4) * np.array([0, 116., 128., 136., 150., 158., 162., 170., 196., 177., 182., 183.,
                                                181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0, 55.6,
                                                47.6, 40.8, 34.1, 28.6, 23.5, 19.3, 15.7, 12.6, 10.2, 8.15, 6.48,
                                                5.02, 3.94, 3.03, 2.33, 1.79])

    """ Results of the HONDA simulation (based on the paper of Honda2015: 'Atmospheric neutrino flux calculation using
    the NRLMSISE-00 atmospheric model'): """
    # Neutrino energy in MeV from the table from file HONDA_juno-ally-01-01-solmin.d (is equal to neutrino energy
    # in HONDA_juno-ally-01-01-solmax.d) (np.array of float):
    energy_honda = 10**3 * np.array([1.0000E-01, 1.1220E-01, 1.2589E-01, 1.4125E-01, 1.5849E-01, 1.7783E-01, 1.9953E-01,
                                     2.2387E-01, 2.5119E-01, 2.8184E-01, 3.1623E-01, 3.5481E-01, 3.9811E-01, 4.4668E-01,
                                     5.0119E-01, 5.6234E-01, 6.3096E-01, 7.0795E-01, 7.9433E-01, 8.9125E-01,
                                     1.0000E+00])

    """ for solar minimum (HONDA_juno-ally-01-01-solmin.d): """
    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar minimum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    # INFO-me: Solid angle (Raumwinkel) of the whole spherical angle  is = 4*pi sr! -> factor 4*pi must be correct!
    flux_nuebar_min_honda = 10**(-7) * 4*np.pi * np.array([2.9367E+03, 2.5746E+03, 2.2332E+03, 1.9206E+03, 1.6395E+03,
                                                           1.3891E+03, 1.1679E+03, 9.7454E+02, 8.0732E+02, 6.6312E+02,
                                                           5.4052E+02, 4.3731E+02, 3.5122E+02, 2.8033E+02, 2.2264E+02,
                                                           1.7581E+02, 1.3804E+02, 1.0776E+02, 8.3623E+01, 6.4555E+01,
                                                           4.9632E+01])

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar minimum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_numubar_min_honda = 10**(-7) * 4*np.pi * np.array([6.2903E+03, 5.5084E+03, 4.8032E+03, 4.1620E+03, 3.5763E+03,
                                                            3.0444E+03, 2.5663E+03, 2.1426E+03, 1.7736E+03, 1.4575E+03,
                                                            1.1890E+03, 9.6400E+02, 7.7693E+02, 6.2283E+02, 4.9647E+02,
                                                            3.9325E+02, 3.1003E+02, 2.4324E+02, 1.9004E+02, 1.4788E+02,
                                                            1.1447E+02])

    """ for solar maximum (HONDA_juno-ally-01-01-solmax.d): """
    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar maximum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_nuebar_max_honda = 10**(-7) * 4*np.pi * np.array([2.7733E+03, 2.4332E+03, 2.1124E+03, 1.8187E+03, 1.5545E+03,
                                                           1.3190E+03, 1.1105E+03, 9.2820E+02, 7.7040E+02, 6.3403E+02,
                                                           5.1790E+02, 4.1997E+02, 3.3811E+02, 2.7054E+02, 2.1539E+02,
                                                           1.7049E+02, 1.3418E+02, 1.0499E+02, 8.1651E+01, 6.3166E+01,
                                                           4.8654E+01])

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar maximum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_numubar_max_honda = 10**(-7) * 4*np.pi * np.array([5.8966E+03, 5.1676E+03, 4.5104E+03, 3.9127E+03, 3.3665E+03,
                                                            2.8701E+03, 2.4238E+03, 2.0277E+03, 1.6821E+03, 1.3857E+03,
                                                            1.1333E+03, 9.2144E+02, 7.4476E+02, 5.9875E+02, 4.7865E+02,
                                                            3.8024E+02, 3.0060E+02, 2.3645E+02, 1.8519E+02, 1.4444E+02,
                                                            1.1204E+02])

    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar AVERAGE at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_atmo_nuebar_honda = (flux_nuebar_min_honda + flux_nuebar_max_honda) / 2

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar AVERAGE at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_atmo_numubar_honda = (flux_numubar_min_honda + flux_numubar_max_honda) / 2

    """ Extrapolate the HONDA flux to the energies of the FLUKA simulation from 10 MeV to 100 MeV: """
    """ Assumption:
        1. the shape of the FLUKA flux as function of energy do NOT depend on the location
            -> the shape of the flux at Super-K can also be used at JUNO site
        
        2. the absolute value of the FLUKA flux at Super-K should be normalized to the location of JUNO
            ->  therefore get the normalization factor by comparing the HONDA flux and the FLUKA flux in the energy 
                range from 100 MeV to 1 GeV        
    """
    # define the energy-array, in which the normalization will be calculated (neutrino energy in MeV)
    # (np.array of float):
    energy_norm = np.arange(min(energy_honda), max(energy_fluka)+0.1, 0.1)

    """ For electron antineutrinos: """
    # Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_nuebar_fluka_interpolated = np.interp(energy_norm, energy_fluka, flux_nuebar_fluka)

    # Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_nuebar_honda_interpolated = np.interp(energy_norm, energy_honda, flux_atmo_nuebar_honda)

    # Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
    integral_nuebar_fluka = np.trapz(flux_nuebar_fluka_interpolated, energy_norm)

    # Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
    integral_nuebar_honda = np.trapz(flux_nuebar_honda_interpolated, energy_norm)

    # Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 115 MeV, in 1/(MeV*s*cm**2)
    # (np.array of float):
    flux_nuebar_fluka_interesting = np.interp(energy_neutrino, energy_fluka, flux_nuebar_fluka)

    # Normalize flux_nuebar_fluka_interesting at Super-K to the electron-antineutrino flux at JUNO,
    # in 1/(MeV * s * cm**2) (np.array of float):
    flux_nuebar_juno = flux_nuebar_fluka_interesting * integral_nuebar_honda / integral_nuebar_fluka

    """ For muon antineutrinos: """
    # Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_numubar_fluka_interpolated = np.interp(energy_norm, energy_fluka, flux_numubar_fluka)

    # Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_numubar_honda_interpolated = np.interp(energy_norm, energy_honda, flux_atmo_numubar_honda)

    # Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
    integral_numubar_fluka = np.trapz(flux_numubar_fluka_interpolated, energy_norm)

    # Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
    integral_numubar_honda = np.trapz(flux_numubar_honda_interpolated, energy_norm)

    # Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
    # (np.array of float):
    flux_numubar_fluka_interesting = np.interp(energy_neutrino, energy_fluka, flux_numubar_fluka)

    # Normalize flux_numubar_fluka_interesting at Super-K to the muon-antineutrino flux at JUNO,
    # in 1/(MeV * s * cm**2) (np.array of float):
    flux_numubar_juno = flux_numubar_fluka_interesting * integral_numubar_honda / integral_numubar_fluka

    """ Taking account neutrino oscillation from the Appendix B of the paper of Fogli et al. from 2004 with title
    "Three-generation flavor transitions and decays of supernova relic neutrinos" (like in ccatmospheric_background_v2):
    """
    # Integer, that defines, if oscillation is considered (oscillation = 1) or not (oscillation = 0):
    oscillation = 1
    # survival probability of electron-antineutrinos (prob_e_to_e = 0.67 if oscillation considered,
    # prob_e_to_e = 1.0 if oscillation not considered):

    # oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos)
    # (prob_mu_to_e = 0.17 if oscillation considered, prob_mu_to_e = 0.0 if oscillation not considered):

    if oscillation == 0:
        prob_e_to_e = 1.0
        prob_mu_to_e = 0.0
    elif oscillation == 1:
        prob_e_to_e = 0.67
        prob_mu_to_e = 0.17

    # total electron-antineutrino flux in the INTERESTING part (10 to 115 MeV) of FLUKA simulation normalized to
    # JUNO site (HONDA) in 1/(MeV * cm**2 * s), (np.array of float):
    flux_total_ccatmospheric_nu_e_bar = prob_e_to_e * flux_nuebar_juno + prob_mu_to_e * flux_numubar_juno

    # Theoretical spectrum (in events per MeV) of electron-antineutrinos ("number of positron-events") from
    # inverse beta decay on free protons (from paper 0903.5323.pdf, equ. 64) (np.array of float):
    theo_spectrum_ccatmospheric = (flux_total_ccatmospheric_nu_e_bar * crosssection *
                                   detection_efficiency * n_target * t * exposure_ratio_muon)

    # number of neutrinos from CC atmospheric background in JUNO detector after "time":
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


def ccatmospheric_background_v4(energy_neutrino, energy_visible, binning_energy_visible, crosssection, n_target,
                                t, detection_efficiency, mass_proton, mass_neutron, mass_positron, exposure_ratio_muon):
    """ Simulate the atmospheric Charged Current electron-antineutrino background:

        ! Version 4:    atmospheric CC background on C12 is added (channel nu_e_bar + C12 -> positron + neutron + X)
                        (see folder /home/astro/blum/juno/atmoNC/other_atmo_background/)!

        the atmospheric CC neutrino flux is based on the simulations of HONDA for the JUNO location

        Results of the HONDA simulation are based on the paper of Honda2015: 'Atmospheric neutrino flux calculation
        using the NRLMSISE-00 atmospheric model')

        The HONDA flux is simulated for JUNO site only for energies above 100 MeV. In the range below 100 MeV the
        spectral shape of the FLUKA simulation in the range from 10 to 100 MeV is used, but normalized to the flux
        from the HONDA simulation.

        Detailed information about the calculations are in the python script 'atmospheric_flux.py'.

        Convolution of the theoretical spectrum with gaussian distribution is calculated with the
        function convolution()

        The paper of Battistoni2005 'The atmospheric neutrino fluxes below 100 MeV: The FLUKA results' is described in
        detail in the my notes. In the paper the electron- and muon-antineutrino flux is simulated for energies
        from 10 MeV to 100 MeV.

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
    :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

    :return:spectrum_ccatmospheric: spectrum of electron-antineutrinos of CC atmospheric background (np.array of float)
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

    # INFO-me: in the notes "Vergleich Atmospheric Flux aus Paper mit Julia's Werten", the flux of Julia was higher
    # than in the data of the paper. BUT in Julia's talk in Strasbourg, the spectrum was lower than in the previous
    # spectrum. Now the spectrum of Julia and from atmospheric_flux.py are more similar.

    """ Theoretical spectrum of atmospheric charged-current background: """

    """ Results of the FLUKA simulation (from the paper of Battistoni2005 'The atmospheric neutrino fluxes below 
    100 MeV: The FLUKA results'): """
    # Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
    energy_fluka = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133,
                             150, 168, 188, 211, 237, 266, 299, 335, 376, 422, 473, 531, 596, 668, 750, 841, 944])

    # differential flux from FLUKA in energy for no oscillation for electron-antineutrinos for solar average at the site
    # of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float).
    # Assumption: for energy = 0 MeV, the flux is also 0!
    flux_nuebar_fluka = 10 ** (-4) * np.array([0, 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                               83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9, 21.3,
                                               18.3, 15.4, 12.9, 10.6, 8.80, 7.13, 5.75, 4.60, 3.68, 2.88, 2.28,
                                               1.87, 1.37, 1.06, 0.800])

    # differential flux from FLUKA in energy for no oscillation for muon-antineutrinos for solar average at the site of
    # Super-K, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float).
    # Assumption: for energy = 0 MeV, the flux is also 0!
    flux_numubar_fluka = 10 ** (-4) * np.array([0, 116., 128., 136., 150., 158., 162., 170., 196., 177., 182., 183.,
                                                181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0, 55.6,
                                                47.6, 40.8, 34.1, 28.6, 23.5, 19.3, 15.7, 12.6, 10.2, 8.15, 6.48,
                                                5.02, 3.94, 3.03, 2.33, 1.79])

    """ Results of the HONDA simulation (based on the paper of Honda2015: 'Atmospheric neutrino flux calculation using
    the NRLMSISE-00 atmospheric model'): """
    # Neutrino energy in MeV from the table from file HONDA_juno-ally-01-01-solmin.d (is equal to neutrino energy
    # in HONDA_juno-ally-01-01-solmax.d) (np.array of float):
    energy_honda = 10 ** 3 * np.array([1.0000E-01, 1.1220E-01, 1.2589E-01, 1.4125E-01, 1.5849E-01, 1.7783E-01,
                                       1.9953E-01, 2.2387E-01, 2.5119E-01, 2.8184E-01, 3.1623E-01, 3.5481E-01,
                                       3.9811E-01, 4.4668E-01, 5.0119E-01, 5.6234E-01, 6.3096E-01, 7.0795E-01,
                                       7.9433E-01, 8.9125E-01, 1.0000E+00])

    """ for solar minimum (HONDA_juno-ally-01-01-solmin.d): """
    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar minimum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    # INFO-me: Solid angle (Raumwinkel) of the whole spherical angle  is = 4*pi sr! -> factor 4*pi must be correct!
    flux_nuebar_min_honda = 10 ** (-7) * 4 * np.pi * np.array([2.9367E+03, 2.5746E+03, 2.2332E+03, 1.9206E+03,
                                                               1.6395E+03, 1.3891E+03, 1.1679E+03, 9.7454E+02,
                                                               8.0732E+02, 6.6312E+02, 5.4052E+02, 4.3731E+02,
                                                               3.5122E+02, 2.8033E+02, 2.2264E+02, 1.7581E+02,
                                                               1.3804E+02, 1.0776E+02, 8.3623E+01, 6.4555E+01,
                                                               4.9632E+01])

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar minimum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_numubar_min_honda = 10 ** (-7) * 4 * np.pi * np.array([6.2903E+03, 5.5084E+03, 4.8032E+03, 4.1620E+03,
                                                                3.5763E+03, 3.0444E+03, 2.5663E+03, 2.1426E+03,
                                                                1.7736E+03, 1.4575E+03, 1.1890E+03, 9.6400E+02,
                                                                7.7693E+02, 6.2283E+02, 4.9647E+02, 3.9325E+02,
                                                                3.1003E+02, 2.4324E+02, 1.9004E+02, 1.4788E+02,
                                                                1.1447E+02])

    """ for solar maximum (HONDA_juno-ally-01-01-solmax.d): """
    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar maximum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_nuebar_max_honda = 10 ** (-7) * 4 * np.pi * np.array([2.7733E+03, 2.4332E+03, 2.1124E+03, 1.8187E+03,
                                                               1.5545E+03, 1.3190E+03, 1.1105E+03, 9.2820E+02,
                                                               7.7040E+02, 6.3403E+02, 5.1790E+02, 4.1997E+02,
                                                               3.3811E+02, 2.7054E+02, 2.1539E+02, 1.7049E+02,
                                                               1.3418E+02, 1.0499E+02, 8.1651E+01, 6.3166E+01,
                                                               4.8654E+01])

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar maximum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_numubar_max_honda = 10 ** (-7) * 4 * np.pi * np.array([5.8966E+03, 5.1676E+03, 4.5104E+03, 3.9127E+03,
                                                                3.3665E+03, 2.8701E+03, 2.4238E+03, 2.0277E+03,
                                                                1.6821E+03, 1.3857E+03, 1.1333E+03, 9.2144E+02,
                                                                7.4476E+02, 5.9875E+02, 4.7865E+02, 3.8024E+02,
                                                                3.0060E+02, 2.3645E+02, 1.8519E+02, 1.4444E+02,
                                                                1.1204E+02])

    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar AVERAGE at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_atmo_nuebar_honda = (flux_nuebar_min_honda + flux_nuebar_max_honda) / 2

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar AVERAGE at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_atmo_numubar_honda = (flux_numubar_min_honda + flux_numubar_max_honda) / 2

    """ Extrapolate the HONDA flux to the energies of the FLUKA simulation from 10 MeV to 100 MeV: """
    """ Assumption:
        1. the shape of the FLUKA flux as function of energy do NOT depend on the location
            -> the shape of the flux at Super-K can also be used at JUNO site

        2. the absolute value of the FLUKA flux at Super-K should be normalized to the location of JUNO
            ->  therefore get the normalization factor by comparing the HONDA flux and the FLUKA flux in the energy 
                range from 100 MeV to 1 GeV        
    """
    # define the energy-array, in which the normalization will be calculated (neutrino energy in MeV)
    # (np.array of float):
    energy_norm = np.arange(min(energy_honda), max(energy_fluka) + 0.1, 0.1)

    """ For electron antineutrinos: """
    # Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_nuebar_fluka_interpolated = np.interp(energy_norm, energy_fluka, flux_nuebar_fluka)

    # Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_nuebar_honda_interpolated = np.interp(energy_norm, energy_honda, flux_atmo_nuebar_honda)

    # Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
    integral_nuebar_fluka = np.trapz(flux_nuebar_fluka_interpolated, energy_norm)

    # Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
    integral_nuebar_honda = np.trapz(flux_nuebar_honda_interpolated, energy_norm)

    # Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 115 MeV, in 1/(MeV*s*cm**2)
    # (np.array of float):
    flux_nuebar_fluka_interesting = np.interp(energy_neutrino, energy_fluka, flux_nuebar_fluka)

    # Normalize flux_nuebar_fluka_interesting at Super-K to the electron-antineutrino flux at JUNO,
    # in 1/(MeV * s * cm**2) (np.array of float):
    flux_nuebar_juno = flux_nuebar_fluka_interesting * integral_nuebar_honda / integral_nuebar_fluka

    """ For muon antineutrinos: """
    # Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_numubar_fluka_interpolated = np.interp(energy_norm, energy_fluka, flux_numubar_fluka)

    # Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_numubar_honda_interpolated = np.interp(energy_norm, energy_honda, flux_atmo_numubar_honda)

    # Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
    integral_numubar_fluka = np.trapz(flux_numubar_fluka_interpolated, energy_norm)

    # Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
    integral_numubar_honda = np.trapz(flux_numubar_honda_interpolated, energy_norm)

    # Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
    # (np.array of float):
    flux_numubar_fluka_interesting = np.interp(energy_neutrino, energy_fluka, flux_numubar_fluka)

    # Normalize flux_numubar_fluka_interesting at Super-K to the muon-antineutrino flux at JUNO,
    # in 1/(MeV * s * cm**2) (np.array of float):
    flux_numubar_juno = flux_numubar_fluka_interesting * integral_numubar_honda / integral_numubar_fluka

    """ Taking account neutrino oscillation from the Appendix B of the paper of Fogli et al. from 2004 with title
    "Three-generation flavor transitions and decays of supernova relic neutrinos" (like in ccatmospheric_background_v2):
    """
    # Integer, that defines, if oscillation is considered (oscillation = 1) or not (oscillation = 0):
    oscillation = 1
    # survival probability of electron-antineutrinos (prob_e_to_e = 0.67 if oscillation considered,
    # prob_e_to_e = 1.0 if oscillation not considered):

    # oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos)
    # (prob_mu_to_e = 0.17 if oscillation considered, prob_mu_to_e = 0.0 if oscillation not considered):

    if oscillation == 0:
        prob_e_to_e = 1.0
        prob_mu_to_e = 0.0

    elif oscillation == 1:
        prob_e_to_e = 0.67
        prob_mu_to_e = 0.17

    # total electron-antineutrino flux in the INTERESTING part (10 to 115 MeV) of FLUKA simulation normalized to
    # JUNO site (HONDA) in 1/(MeV * cm**2 * s), (np.array of float):
    flux_total_ccatmospheric_nu_e_bar = prob_e_to_e * flux_nuebar_juno + prob_mu_to_e * flux_numubar_juno

    # Theoretical spectrum (in events per MeV) of electron-antineutrinos ("number of positron-events") from
    # inverse beta decay on free protons (from paper 0903.5323.pdf, equ. 64) (np.array of float):
    theo_spectrum_ccatmospheric_proton = (flux_total_ccatmospheric_nu_e_bar * crosssection *
                                          detection_efficiency * n_target * t * exposure_ratio_muon)

    # Spectrum of the atmospheric charged-current electron-antineutrino background on protons in 1/MeV,
    # theoretical spectrum is convolved with gaussian distribution:
    spectrum_ccatmospheric_proton = convolution(energy_neutrino, energy_visible, binning_energy_visible,
                                                theo_spectrum_ccatmospheric_proton, mass_proton, mass_neutron,
                                                mass_positron)

    """ cross-section for atmospheric CC background on C12 (nu_e_bar + C12 -> positron + neutron + X) from Yoshida, 
    2008, 'NEUTRINO-NUCLEUS REACTION CROSS SECTIONS FOR LIGHT ELEMENT SYNTHESIS IN SUPERNOVA EXPLOSIONS': """
    # information about cross-section from Yoshida paper in xsec_C12_Yoshida_2008.ods:
    # neutrino energy in MeV:
    energy_yoshida = np.arange(1, 150+1, 1)
    # cross-section of nu_e_bar + C12 -> positron + neutron + X (all channels, where 1 neutron is produced) in cm**2:
    xsec_nuebar_yoshida_data = 10 ** (-42) * np.array([0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
                                                       0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
                                                       0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
                                                       1.17E-03, 5.77E-03, 1.41E-02, 3.00E-02, 5.66E-02, 9.92E-02,
                                                       1.59E-01, 2.40E-01, 3.44E-01, 4.75E-01, 6.36E-01, 8.32E-01,
                                                       1.07E+00, 1.34E+00, 1.66E+00, 2.02E+00, 2.43E+00, 2.90E+00,
                                                       3.43E+00, 4.01E+00, 4.66E+00, 5.37E+00, 6.15E+00, 7.00E+00,
                                                       7.93E+00, 8.93E+00, 1.00E+01, 1.12E+01, 1.24E+01, 1.37E+01,
                                                       1.51E+01, 1.66E+01, 1.82E+01, 1.99E+01, 2.17E+01, 2.35E+01,
                                                       2.55E+01, 2.75E+01, 2.96E+01, 3.18E+01, 3.41E+01, 3.65E+01,
                                                       3.89E+01, 4.15E+01, 4.41E+01, 4.68E+01, 4.96E+01, 5.25E+01,
                                                       5.54E+01, 5.85E+01, 6.15E+01, 6.47E+01, 6.79E+01, 7.12E+01,
                                                       7.46E+01, 7.80E+01, 8.14E+01, 8.50E+01, 8.85E+01, 9.21E+01,
                                                       9.58E+01, 9.95E+01, 1.03E+02, 1.07E+02, 1.11E+02, 1.15E+02,
                                                       1.19E+02, 1.22E+02, 1.26E+02, 1.30E+02, 1.34E+02, 1.38E+02,
                                                       1.42E+02, 1.46E+02, 1.50E+02, 1.54E+02, 1.58E+02, 1.62E+02,
                                                       1.66E+02, 1.70E+02, 1.74E+02, 1.78E+02, 1.82E+02, 1.86E+02,
                                                       1.90E+02, 1.94E+02, 1.98E+02, 2.02E+02, 2.06E+02, 2.09E+02,
                                                       2.13E+02, 2.17E+02, 2.21E+02, 2.25E+02, 2.28E+02, 2.32E+02,
                                                       2.36E+02, 2.40E+02, 2.43E+02, 2.47E+02, 2.51E+02, 2.54E+02,
                                                       2.58E+02, 2.61E+02, 2.65E+02, 2.69E+02, 2.72E+02, 2.76E+02,
                                                       2.79E+02, 2.82E+02, 2.86E+02, 2.89E+02, 2.93E+02, 2.96E+02,
                                                       2.99E+02, 3.03E+02, 3.06E+02, 3.09E+02, 3.12E+02, 3.16E+02,
                                                       3.19E+02, 3.22E+02, 3.25E+02, 3.28E+02, 3.32E+02, 3.35E+02,
                                                       3.38E+02, 3.41E+02, 3.44E+02, 3.47E+02, 3.50E+02, 3.53E+02])
    # interpolate cross-section with energy_neutrino:
    xsec_nuebar_yoshida = np.interp(energy_neutrino, energy_yoshida, xsec_nuebar_yoshida_data)

    # number of C12 targets in whole LS:
    n_c12 = number_c12_atoms(17.7)

    # Theoretical spectrum (in events per MeV) of electron-antineutrinos from CC interaction on C12
    # (nu_e_bar + C12 -> positron + neutron + X) (array of float):
    theo_spectrum_ccatmospheric_c12 = (flux_total_ccatmospheric_nu_e_bar * xsec_nuebar_yoshida * detection_efficiency
                                       * n_c12 * t * exposure_ratio_muon)

    # Theoretical spectrum is a function of the neutrino energy (normally from 10 MeV to 120 MeV). When you now want to
    # convolve it with the energy resolution and the correlation between E_e and E_nu, you will get negative energies
    # for E_nu < 17.3 MeV! Therefore only take the theoretical spectrum from 17.3 MeV to 120 MeV:
    # INFO-me: min(energy_neutrino) must be = 10 MeV and bin-width of energy_neutrino must be = 0.01 MeV
    index_17_3_mev = int((17.3 - min(energy_neutrino)) * 1 / 0.01)
    energy_neutrino_1 = energy_neutrino[index_17_3_mev:]
    theo_spectrum_ccatmospheric_c12 = theo_spectrum_ccatmospheric_c12[index_17_3_mev:]
    print(energy_neutrino_1)
    print(theo_spectrum_ccatmospheric_c12)

    # Spectrum of the atmospheric charged-current electron-antineutrino background on C12
    # (nu_e_bar + C12 -> positron + neutron + X) in 1/MeV, theoretical spectrum is convolved with gaussian distribution:
    spectrum_ccatmospheric_c12 = convolution_neutron_b11(energy_neutrino_1, energy_visible, binning_energy_visible,
                                                         theo_spectrum_ccatmospheric_c12, mass_neutron, mass_positron)

    # calculate total atmospheric CC background:
    spectrum_ccatmospheric = spectrum_ccatmospheric_proton + spectrum_ccatmospheric_c12

    # calculate total number of events in spectrum_ccatmospheric:
    n_neutrino_ccatmospheric_vis = np.trapz(spectrum_ccatmospheric, energy_visible)

    # calculate total number of events from theoretical spectrum:
    n_neutrino_ccatmospheric_theo = (np.trapz(theo_spectrum_ccatmospheric_proton, energy_neutrino) +
                                     np.trapz(theo_spectrum_ccatmospheric_c12, energy_neutrino_1))

    return (spectrum_ccatmospheric, n_neutrino_ccatmospheric_vis,
            n_neutrino_ccatmospheric_theo, oscillation, prob_e_to_e, prob_mu_to_e, n_c12)


def ccatmospheric_background_v5(energy_neutrino, crosssection, n_target, t, detection_efficiency, exposure_ratio_muon):
    """ Simulate the atmospheric Charged Current electron-antineutrino background:

        ! Version 5:
        - only theoretical spectrum is calculated! IBD kinematics are considered afterwards!

        - visible energy is calculated with neutrino energy and the IBD kinematics (theta is considered).

        - theoretical spectrum is not convolved with gaussian distribution to get visible spectrum !

        Results of the HONDA simulation are based on the paper of Honda2015: 'Atmospheric neutrino flux calculation
        using the NRLMSISE-00 atmospheric model')

        The HONDA flux is simulated for JUNO site only for energies above 100 MeV. In the range below 100 MeV the
        spectral shape of the FLUKA simulation in the range from 10 to 100 MeV is used, but normalized to the flux
        from the HONDA simulation.

        Detailed information about the calculations are in the python script 'atmospheric_flux.py'.

        Convolution of the theoretical spectrum with gaussian distribution is calculated with the
        function convolution()

        The paper of Battistoni2005 'The atmospheric neutrino fluxes below 100 MeV: The FLUKA results' is described in
        detail in the my notes. In the paper the electron- and muon-antineutrino flux is simulated for energies
        from 10 MeV to 100 MeV.

    :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
    :param crosssection: IBD cross-section in cm**2 (np.array of float), produced with the function sigma_ibd()
    :param n_target: number of free protons in JUNO (float)
    :param t: exposure time in seconds (float)
    :param detection_efficiency: detection efficiency of IBD in JUNO (float)
    :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

    :return:spectrum_ccatmospheric: spectrum of electron-antineutrinos of CC atmospheric background (np.array of float)
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

    # INFO-me: in the notes "Vergleich Atmospheric Flux aus Paper mit Julia's Werten", the flux of Julia was higher
    # than in the data of the paper. BUT in Julia's talk in Strasbourg, the spectrum was lower than in the previous
    # spectrum. Now the spectrum of Julia and from atmospheric_flux.py are more similar.

    """ Theoretical spectrum of atmospheric charged-current background: """

    """ Results of the FLUKA simulation (from the paper of Battistoni2005 'The atmospheric neutrino fluxes below 
    100 MeV: The FLUKA results'): """
    # Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
    energy_fluka = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133,
                             150, 168, 188, 211, 237, 266, 299, 335, 376, 422, 473, 531, 596, 668, 750, 841, 944])

    # differential flux from FLUKA in energy for no oscillation for electron-antineutrinos for solar average at the site
    # of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float).
    # Assumption: for energy = 0 MeV, the flux is also 0!
    flux_nuebar_fluka = 10 ** (-4) * np.array([0, 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                               83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9, 21.3,
                                               18.3, 15.4, 12.9, 10.6, 8.80, 7.13, 5.75, 4.60, 3.68, 2.88, 2.28,
                                               1.87, 1.37, 1.06, 0.800])

    # differential flux from FLUKA in energy for no oscillation for muon-antineutrinos for solar average at the site of
    # Super-K, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float).
    # Assumption: for energy = 0 MeV, the flux is also 0!
    flux_numubar_fluka = 10 ** (-4) * np.array([0, 116., 128., 136., 150., 158., 162., 170., 196., 177., 182., 183.,
                                                181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0, 55.6,
                                                47.6, 40.8, 34.1, 28.6, 23.5, 19.3, 15.7, 12.6, 10.2, 8.15, 6.48,
                                                5.02, 3.94, 3.03, 2.33, 1.79])

    """ Results of the HONDA simulation (based on the paper of Honda2015: 'Atmospheric neutrino flux calculation using
    the NRLMSISE-00 atmospheric model'): """
    # Neutrino energy in MeV from the table from file HONDA_juno-ally-01-01-solmin.d (is equal to neutrino energy
    # in HONDA_juno-ally-01-01-solmax.d) (np.array of float):
    energy_honda = 10 ** 3 * np.array(
        [1.0000E-01, 1.1220E-01, 1.2589E-01, 1.4125E-01, 1.5849E-01, 1.7783E-01, 1.9953E-01,
         2.2387E-01, 2.5119E-01, 2.8184E-01, 3.1623E-01, 3.5481E-01, 3.9811E-01, 4.4668E-01,
         5.0119E-01, 5.6234E-01, 6.3096E-01, 7.0795E-01, 7.9433E-01, 8.9125E-01,
         1.0000E+00])

    """ for solar minimum (HONDA_juno-ally-01-01-solmin.d): """
    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar minimum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    # INFO-me: Solid angle (Raumwinkel) of the whole spherical angle  is = 4*pi sr! -> factor 4*pi must be correct!
    flux_nuebar_min_honda = 10 ** (-7) * 4 * np.pi * np.array(
        [2.9367E+03, 2.5746E+03, 2.2332E+03, 1.9206E+03, 1.6395E+03,
         1.3891E+03, 1.1679E+03, 9.7454E+02, 8.0732E+02, 6.6312E+02,
         5.4052E+02, 4.3731E+02, 3.5122E+02, 2.8033E+02, 2.2264E+02,
         1.7581E+02, 1.3804E+02, 1.0776E+02, 8.3623E+01, 6.4555E+01,
         4.9632E+01])

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar minimum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_numubar_min_honda = 10 ** (-7) * 4 * np.pi * np.array(
        [6.2903E+03, 5.5084E+03, 4.8032E+03, 4.1620E+03, 3.5763E+03,
         3.0444E+03, 2.5663E+03, 2.1426E+03, 1.7736E+03, 1.4575E+03,
         1.1890E+03, 9.6400E+02, 7.7693E+02, 6.2283E+02, 4.9647E+02,
         3.9325E+02, 3.1003E+02, 2.4324E+02, 1.9004E+02, 1.4788E+02,
         1.1447E+02])

    """ for solar maximum (HONDA_juno-ally-01-01-solmax.d): """
    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar maximum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_nuebar_max_honda = 10 ** (-7) * 4 * np.pi * np.array(
        [2.7733E+03, 2.4332E+03, 2.1124E+03, 1.8187E+03, 1.5545E+03,
         1.3190E+03, 1.1105E+03, 9.2820E+02, 7.7040E+02, 6.3403E+02,
         5.1790E+02, 4.1997E+02, 3.3811E+02, 2.7054E+02, 2.1539E+02,
         1.7049E+02, 1.3418E+02, 1.0499E+02, 8.1651E+01, 6.3166E+01,
         4.8654E+01])

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar maximum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_numubar_max_honda = 10 ** (-7) * 4 * np.pi * np.array(
        [5.8966E+03, 5.1676E+03, 4.5104E+03, 3.9127E+03, 3.3665E+03,
         2.8701E+03, 2.4238E+03, 2.0277E+03, 1.6821E+03, 1.3857E+03,
         1.1333E+03, 9.2144E+02, 7.4476E+02, 5.9875E+02, 4.7865E+02,
         3.8024E+02, 3.0060E+02, 2.3645E+02, 1.8519E+02, 1.4444E+02,
         1.1204E+02])

    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar AVERAGE at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    # flux_atmo_nuebar_honda = (flux_nuebar_min_honda + flux_nuebar_max_honda) / 2
    flux_atmo_nuebar_honda = flux_nuebar_min_honda

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar AVERAGE at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    # flux_atmo_numubar_honda = (flux_numubar_min_honda + flux_numubar_max_honda) / 2
    flux_atmo_numubar_honda = flux_numubar_min_honda

    """ Extrapolate the HONDA flux to the energies of the FLUKA simulation from 10 MeV to 100 MeV: """
    """ Assumption:
        1. the shape of the FLUKA flux as function of energy do NOT depend on the location
            -> the shape of the flux at Super-K can also be used at JUNO site

        2. the absolute value of the FLUKA flux at Super-K should be normalized to the location of JUNO
            ->  therefore get the normalization factor by comparing the HONDA flux and the FLUKA flux in the energy 
                range from 100 MeV to 1 GeV        
    """
    # define the energy-array, in which the normalization will be calculated (neutrino energy in MeV)
    # (np.array of float):
    energy_norm = np.arange(min(energy_honda), max(energy_fluka) + 0.1, 0.1)

    """ For electron antineutrinos: """
    # Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_nuebar_fluka_interpolated = np.interp(energy_norm, energy_fluka, flux_nuebar_fluka)

    # Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_nuebar_honda_interpolated = np.interp(energy_norm, energy_honda, flux_atmo_nuebar_honda)

    # Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
    integral_nuebar_fluka = np.trapz(flux_nuebar_fluka_interpolated, energy_norm)

    # Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
    integral_nuebar_honda = np.trapz(flux_nuebar_honda_interpolated, energy_norm)

    # Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 115 MeV, in 1/(MeV*s*cm**2)
    # (np.array of float):
    flux_nuebar_fluka_interesting = np.interp(energy_neutrino, energy_fluka, flux_nuebar_fluka)

    # Normalize flux_nuebar_fluka_interesting at Super-K to the electron-antineutrino flux at JUNO,
    # in 1/(MeV * s * cm**2) (np.array of float):
    flux_nuebar_juno = flux_nuebar_fluka_interesting * integral_nuebar_honda / integral_nuebar_fluka

    """ For muon antineutrinos: """
    # Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_numubar_fluka_interpolated = np.interp(energy_norm, energy_fluka, flux_numubar_fluka)

    # Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_numubar_honda_interpolated = np.interp(energy_norm, energy_honda, flux_atmo_numubar_honda)

    # Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
    integral_numubar_fluka = np.trapz(flux_numubar_fluka_interpolated, energy_norm)

    # Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
    integral_numubar_honda = np.trapz(flux_numubar_honda_interpolated, energy_norm)

    # Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
    # (np.array of float):
    flux_numubar_fluka_interesting = np.interp(energy_neutrino, energy_fluka, flux_numubar_fluka)

    # Normalize flux_numubar_fluka_interesting at Super-K to the muon-antineutrino flux at JUNO,
    # in 1/(MeV * s * cm**2) (np.array of float):
    flux_numubar_juno = flux_numubar_fluka_interesting * integral_numubar_honda / integral_numubar_fluka

    """ Taking account neutrino oscillation from the Appendix B of the paper of Fogli et al. from 2004 with title
    "Three-generation flavor transitions and decays of supernova relic neutrinos" (like in ccatmospheric_background_v2):
    """
    # Integer, that defines, if oscillation is considered (oscillation = 1) or not (oscillation = 0):
    oscillation = 1
    # survival probability of electron-antineutrinos (prob_e_to_e = 0.67 if oscillation considered,
    # prob_e_to_e = 1.0 if oscillation not considered):

    # oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos)
    # (prob_mu_to_e = 0.17 if oscillation considered, prob_mu_to_e = 0.0 if oscillation not considered):

    if oscillation == 0:
        prob_e_to_e = 1.0
        prob_mu_to_e = 0.0
    elif oscillation == 1:
        prob_e_to_e = 0.67
        prob_mu_to_e = 0.17

    # total electron-antineutrino flux in the INTERESTING part (10 to 115 MeV) of FLUKA simulation normalized to
    # JUNO site (HONDA) in 1/(MeV * cm**2 * s), (np.array of float):
    flux_total_ccatmospheric_nu_e_bar = prob_e_to_e * flux_nuebar_juno + prob_mu_to_e * flux_numubar_juno

    # Theoretical spectrum (in events per MeV) of electron-antineutrinos ("number of positron-events") from
    # inverse beta decay on free protons (from paper 0903.5323.pdf, equ. 64) (np.array of float):
    theo_spectrum_ccatmospheric = (flux_total_ccatmospheric_nu_e_bar * crosssection *
                                   detection_efficiency * n_target * t * exposure_ratio_muon)

    # number of neutrinos from CC atmospheric background in JUNO detector after "time":
    n_neutrino_ccatmospheric_theo = np.trapz(theo_spectrum_ccatmospheric, energy_neutrino)

    return theo_spectrum_ccatmospheric, n_neutrino_ccatmospheric_theo, oscillation, prob_e_to_e, prob_mu_to_e


def ccatmospheric_background_v6(energy_neutrino, t, detection_efficiency, exposure_ratio_muon):
    """ Simulate the atmospheric Charged Current electron-antineutrino background:

        ! Version 6:    visible energy is calculated with neutrino energy and the IBD kinematics (theta is considered).
                        theoretical spectrum is not convolved with gaussian distribution to get visible spectrum

                        atmospheric CC background on C12 is added (channel nu_e_bar + C12 -> positron + neutron + X)
                        (see folder /home/astro/blum/juno/atmoNC/other_atmo_background/)    !

        the atmospheric CC neutrino flux is based on the simulations of HONDA for the JUNO location

        Results of the HONDA simulation are based on the paper of Honda2015: 'Atmospheric neutrino flux calculation
        using the NRLMSISE-00 atmospheric model')

        The HONDA flux is simulated for JUNO site only for energies above 100 MeV. In the range below 100 MeV the
        spectral shape of the FLUKA simulation in the range from 10 to 100 MeV is used, but normalized to the flux
        from the HONDA simulation.

        Detailed information about the calculations are in the python script 'atmospheric_flux.py'.

        Convolution of the theoretical spectrum with gaussian distribution is calculated with the
        function convolution()

        The paper of Battistoni2005 'The atmospheric neutrino fluxes below 100 MeV: The FLUKA results' is described in
        detail in the my notes. In the paper the electron- and muon-antineutrino flux is simulated for energies
        from 10 MeV to 100 MeV.

    :param energy_neutrino: energy corresponding to the electron-antineutrino energy in MeV (np.array of float)
    :param t: exposure time in seconds (float)
    :param detection_efficiency: detection efficiency of IBD in JUNO (float)
    :param exposure_ratio_muon: exposure ratio due to muon veto cut (muon cuts leads to dead time of detector)

    :return:theo_spectrum_ccatmospheric: Theoretical spectrum of the atmospheric CC electron-antineutrino background
            in 1/MeV (number of events as function of the electron-antineutrino energy) (np.array of float64)
            n_neutrino_ccatmospheric_theo: number of atmospheric CC electron-antineutrino events in JUNO after "time"
            (float64)
            oscillation: oscillation is considered for oscillation=1, oscillation is not considered for oscillation=0
            (integer)
            prob_e_to_e: survival probability of electron-antineutrinos (electron-antineutrinos oscillate to
            electron-antineutrinos) (float)
            prob_mu_to_e: oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos) (float)
    """
    # INFO-me: in the notes "Vergleich Atmospheric Flux aus Paper mit Julia's Werten", the flux of Julia was higher
    # than in the data of the paper. BUT in Julia's talk in Strasbourg, the spectrum was lower than in the previous
    # spectrum. Now the spectrum of Julia and from atmospheric_flux.py are more similar.

    """ Theoretical spectrum of atmospheric charged-current background: """

    """ Results of the FLUKA simulation (from the paper of Battistoni2005 'The atmospheric neutrino fluxes below 
    100 MeV: The FLUKA results'): """
    # Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
    energy_fluka = np.array([0, 13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133,
                             150, 168, 188, 211, 237, 266, 299, 335, 376, 422, 473, 531, 596, 668, 750, 841, 944])

    # differential flux from FLUKA in energy for no oscillation for electron-antineutrinos for solar average at the site
    # of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float).
    # Assumption: for energy = 0 MeV, the flux is also 0!
    flux_nuebar_fluka = 10 ** (-4) * np.array([0, 63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                               83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9, 21.3,
                                               18.3, 15.4, 12.9, 10.6, 8.80, 7.13, 5.75, 4.60, 3.68, 2.88, 2.28,
                                               1.87, 1.37, 1.06, 0.800])

    # differential flux from FLUKA in energy for no oscillation for muon-antineutrinos for solar average at the site of
    # Super-K, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float).
    # Assumption: for energy = 0 MeV, the flux is also 0!
    flux_numubar_fluka = 10 ** (-4) * np.array([0, 116., 128., 136., 150., 158., 162., 170., 196., 177., 182., 183.,
                                                181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0, 55.6,
                                                47.6, 40.8, 34.1, 28.6, 23.5, 19.3, 15.7, 12.6, 10.2, 8.15, 6.48,
                                                5.02, 3.94, 3.03, 2.33, 1.79])

    """ Results of the HONDA simulation (based on the paper of Honda2015: 'Atmospheric neutrino flux calculation using
    the NRLMSISE-00 atmospheric model'): """
    # Neutrino energy in MeV from the table from file HONDA_juno-ally-01-01-solmin.d (is equal to neutrino energy
    # in HONDA_juno-ally-01-01-solmax.d) (np.array of float):
    energy_honda = 10 ** 3 * np.array([1.0000E-01, 1.1220E-01, 1.2589E-01, 1.4125E-01, 1.5849E-01, 1.7783E-01,
                                       1.9953E-01, 2.2387E-01, 2.5119E-01, 2.8184E-01, 3.1623E-01, 3.5481E-01,
                                       3.9811E-01, 4.4668E-01, 5.0119E-01, 5.6234E-01, 6.3096E-01, 7.0795E-01,
                                       7.9433E-01, 8.9125E-01, 1.0000E+00])

    """ for solar minimum (HONDA_juno-ally-01-01-solmin.d): """
    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar minimum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    # INFO-me: Solid angle (Raumwinkel) of the whole spherical angle  is = 4*pi sr! -> factor 4*pi must be correct!
    flux_nuebar_min_honda = 10 ** (-7) * 4 * np.pi * np.array([2.9367E+03, 2.5746E+03, 2.2332E+03, 1.9206E+03,
                                                               1.6395E+03, 1.3891E+03, 1.1679E+03, 9.7454E+02,
                                                               8.0732E+02, 6.6312E+02, 5.4052E+02, 4.3731E+02,
                                                               3.5122E+02, 2.8033E+02, 2.2264E+02, 1.7581E+02,
                                                               1.3804E+02, 1.0776E+02, 8.3623E+01, 6.4555E+01,
                                                               4.9632E+01])

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar minimum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_numubar_min_honda = 10 ** (-7) * 4 * np.pi * np.array([6.2903E+03, 5.5084E+03, 4.8032E+03, 4.1620E+03,
                                                                3.5763E+03, 3.0444E+03, 2.5663E+03, 2.1426E+03,
                                                                1.7736E+03, 1.4575E+03, 1.1890E+03, 9.6400E+02,
                                                                7.7693E+02, 6.2283E+02, 4.9647E+02, 3.9325E+02,
                                                                3.1003E+02, 2.4324E+02, 1.9004E+02, 1.4788E+02,
                                                                1.1447E+02])

    """ for solar maximum (HONDA_juno-ally-01-01-solmax.d): """
    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar maximum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_nuebar_max_honda = 10 ** (-7) * 4 * np.pi * np.array([2.7733E+03, 2.4332E+03, 2.1124E+03, 1.8187E+03,
                                                               1.5545E+03, 1.3190E+03, 1.1105E+03, 9.2820E+02,
                                                               7.7040E+02, 6.3403E+02, 5.1790E+02, 4.1997E+02,
                                                               3.3811E+02, 2.7054E+02, 2.1539E+02, 1.7049E+02,
                                                               1.3418E+02, 1.0499E+02, 8.1651E+01, 6.3166E+01,
                                                               4.8654E+01])

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar maximum at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    flux_numubar_max_honda = 10 ** (-7) * 4 * np.pi * np.array([5.8966E+03, 5.1676E+03, 4.5104E+03, 3.9127E+03,
                                                                3.3665E+03, 2.8701E+03, 2.4238E+03, 2.0277E+03,
                                                                1.6821E+03, 1.3857E+03, 1.1333E+03, 9.2144E+02,
                                                                7.4476E+02, 5.9875E+02, 4.7865E+02, 3.8024E+02,
                                                                3.0060E+02, 2.3645E+02, 1.8519E+02, 1.4444E+02,
                                                                1.1204E+02])

    # all-direction averaged flux for no oscillation for electron-antineutrinos for solar AVERAGE at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    # flux_atmo_nuebar_honda = (flux_nuebar_min_honda + flux_nuebar_max_honda) / 2
    flux_atmo_nuebar_honda = flux_nuebar_min_honda

    # all-direction averaged flux for no oscillation for muon-antineutrinos for solar AVERAGE at the site of JUNO
    # (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
    # flux_atmo_numubar_honda = (flux_numubar_min_honda + flux_numubar_max_honda) / 2
    flux_atmo_numubar_honda = flux_numubar_min_honda

    """ Extrapolate the HONDA flux to the energies of the FLUKA simulation from 10 MeV to 100 MeV: """
    """ Assumption:
        1. the shape of the FLUKA flux as function of energy do NOT depend on the location
            -> the shape of the flux at Super-K can also be used at JUNO site

        2. the absolute value of the FLUKA flux at Super-K should be normalized to the location of JUNO
            ->  therefore get the normalization factor by comparing the HONDA flux and the FLUKA flux in the energy 
                range from 100 MeV to 1 GeV        
    """
    # define the energy-array, in which the normalization will be calculated (neutrino energy in MeV)
    # (np.array of float):
    energy_norm = np.arange(min(energy_honda), max(energy_fluka) + 0.1, 0.1)

    """ For electron antineutrinos: """
    # Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_nuebar_fluka_interpolated = np.interp(energy_norm, energy_fluka, flux_nuebar_fluka)

    # Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_nuebar_honda_interpolated = np.interp(energy_norm, energy_honda, flux_atmo_nuebar_honda)

    # Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
    integral_nuebar_fluka = np.trapz(flux_nuebar_fluka_interpolated, energy_norm)

    # Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
    integral_nuebar_honda = np.trapz(flux_nuebar_honda_interpolated, energy_norm)

    # Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 115 MeV, in 1/(MeV*s*cm**2)
    # (np.array of float):
    flux_nuebar_fluka_interesting = np.interp(energy_neutrino, energy_fluka, flux_nuebar_fluka)

    # Normalize flux_nuebar_fluka_interesting at Super-K to the electron-antineutrino flux at JUNO,
    # in 1/(MeV * s * cm**2) (np.array of float):
    flux_nuebar_juno = flux_nuebar_fluka_interesting * integral_nuebar_honda / integral_nuebar_fluka

    """ For muon antineutrinos: """
    # Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_numubar_fluka_interpolated = np.interp(energy_norm, energy_fluka, flux_numubar_fluka)

    # Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
    # in 1/(MeV * cm**2 * s) (np.array of float):
    flux_numubar_honda_interpolated = np.interp(energy_norm, energy_honda, flux_atmo_numubar_honda)

    # Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
    integral_numubar_fluka = np.trapz(flux_numubar_fluka_interpolated, energy_norm)

    # Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
    integral_numubar_honda = np.trapz(flux_numubar_honda_interpolated, energy_norm)

    # Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
    # (np.array of float):
    flux_numubar_fluka_interesting = np.interp(energy_neutrino, energy_fluka, flux_numubar_fluka)

    # Normalize flux_numubar_fluka_interesting at Super-K to the muon-antineutrino flux at JUNO,
    # in 1/(MeV * s * cm**2) (np.array of float):
    flux_numubar_juno = flux_numubar_fluka_interesting * integral_numubar_honda / integral_numubar_fluka

    """ Taking account neutrino oscillation from the Appendix B of the paper of Fogli et al. from 2004 with title
    "Three-generation flavor transitions and decays of supernova relic neutrinos" (like in ccatmospheric_background_v2):
    """
    # Integer, that defines, if oscillation is considered (oscillation = 1) or not (oscillation = 0):
    oscillation = 1
    # survival probability of electron-antineutrinos (prob_e_to_e = 0.67 if oscillation considered,
    # prob_e_to_e = 1.0 if oscillation not considered):

    # oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos)
    # (prob_mu_to_e = 0.17 if oscillation considered, prob_mu_to_e = 0.0 if oscillation not considered):

    if oscillation == 0:
        prob_e_to_e = 1.0
        prob_mu_to_e = 0.0

    elif oscillation == 1:
        prob_e_to_e = 0.67
        prob_mu_to_e = 0.17

    # total electron-antineutrino flux in the INTERESTING part (10 to 115 MeV) of FLUKA simulation normalized to
    # JUNO site (HONDA) in 1/(MeV * cm**2 * s), (np.array of float):
    flux_total_ccatmospheric_nu_e_bar = prob_e_to_e * flux_nuebar_juno + prob_mu_to_e * flux_numubar_juno

    """ cross-section for atmospheric CC background on C12 (nu_e_bar + C12 -> positron + neutron + X) from Yoshida, 
    2008, 'NEUTRINO-NUCLEUS REACTION CROSS SECTIONS FOR LIGHT ELEMENT SYNTHESIS IN SUPERNOVA EXPLOSIONS': """
    # information about cross-section from Yoshida paper in xsec_C12_Yoshida_2008.ods:
    # neutrino energy in MeV:
    energy_yoshida = np.arange(1, 179+1, 1)
    # cross-section of nu_e_bar + C12 -> positron + neutron + X (all channels, where 1 neutron is produced) in cm**2:
    xsec_nuebar_yoshida_data = 10 ** (-42) * np.array([0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
                                                       0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
                                                       0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
                                                       1.17E-03, 5.77E-03, 1.41E-02, 3.00E-02, 5.66E-02, 9.92E-02,
                                                       1.59E-01, 2.40E-01, 3.44E-01, 4.75E-01, 6.36E-01, 8.32E-01,
                                                       1.07E+00, 1.34E+00, 1.66E+00, 2.02E+00, 2.43E+00, 2.90E+00,
                                                       3.43E+00, 4.01E+00, 4.66E+00, 5.37E+00, 6.15E+00, 7.00E+00,
                                                       7.93E+00, 8.93E+00, 1.00E+01, 1.12E+01, 1.24E+01, 1.37E+01,
                                                       1.51E+01, 1.66E+01, 1.82E+01, 1.99E+01, 2.17E+01, 2.35E+01,
                                                       2.55E+01, 2.75E+01, 2.96E+01, 3.18E+01, 3.41E+01, 3.65E+01,
                                                       3.89E+01, 4.15E+01, 4.41E+01, 4.68E+01, 4.96E+01, 5.25E+01,
                                                       5.54E+01, 5.85E+01, 6.15E+01, 6.47E+01, 6.79E+01, 7.12E+01,
                                                       7.46E+01, 7.80E+01, 8.14E+01, 8.50E+01, 8.85E+01, 9.21E+01,
                                                       9.58E+01, 9.95E+01, 1.03E+02, 1.07E+02, 1.11E+02, 1.15E+02,
                                                       1.19E+02, 1.22E+02, 1.26E+02, 1.30E+02, 1.34E+02, 1.38E+02,
                                                       1.42E+02, 1.46E+02, 1.50E+02, 1.54E+02, 1.58E+02, 1.62E+02,
                                                       1.66E+02, 1.70E+02, 1.74E+02, 1.78E+02, 1.82E+02, 1.86E+02,
                                                       1.90E+02, 1.94E+02, 1.98E+02, 2.02E+02, 2.06E+02, 2.09E+02,
                                                       2.13E+02, 2.17E+02, 2.21E+02, 2.25E+02, 2.28E+02, 2.32E+02,
                                                       2.36E+02, 2.40E+02, 2.43E+02, 2.47E+02, 2.51E+02, 2.54E+02,
                                                       2.58E+02, 2.61E+02, 2.65E+02, 2.69E+02, 2.72E+02, 2.76E+02,
                                                       2.79E+02, 2.82E+02, 2.86E+02, 2.89E+02, 2.93E+02, 2.96E+02,
                                                       2.99E+02, 3.03E+02, 3.06E+02, 3.09E+02, 3.12E+02, 3.16E+02,
                                                       3.19E+02, 3.22E+02, 3.25E+02, 3.28E+02, 3.32E+02, 3.35E+02,
                                                       3.38E+02, 3.41E+02, 3.44E+02, 3.47E+02, 3.50E+02, 3.53E+02,
                                                       3.56E+02, 3.59E+02, 3.62E+02, 3.65E+02, 3.68E+02, 3.71E+02,
                                                       3.74E+02, 3.77E+02, 3.80E+02, 3.82E+02, 3.85E+02, 3.88E+02,
                                                       3.91E+02, 3.94E+02, 3.97E+02, 3.99E+02, 4.02E+02, 4.05E+02,
                                                       4.08E+02, 4.11E+02, 4.13E+02, 4.16E+02, 4.19E+02, 4.22E+02,
                                                       4.24E+02, 4.27E+02, 4.30E+02, 4.32E+02, 4.35E+02])
    # interpolate cross-section with energy_neutrino:
    xsec_nuebar_yoshida = np.interp(energy_neutrino, energy_yoshida, xsec_nuebar_yoshida_data)

    # number of C12 targets in whole LS:
    n_c12 = number_c12_atoms(17.7)

    # Theoretical spectrum (in events per MeV) of electron-antineutrinos from CC interaction on C12
    # (nu_e_bar + C12 -> positron + neutron + X) (array of float):
    theo_spectrum_ccatmospheric_c12 = (flux_total_ccatmospheric_nu_e_bar * xsec_nuebar_yoshida * detection_efficiency
                                       * n_c12 * t * exposure_ratio_muon)

    # number of events from theoretical spectrum:
    n_neutrino_ccatmospheric_theo = np.trapz(theo_spectrum_ccatmospheric_c12, energy_neutrino)

    return theo_spectrum_ccatmospheric_c12, n_neutrino_ccatmospheric_theo, oscillation, prob_e_to_e, prob_mu_to_e, n_c12


def number_c12_atoms(radius_cut):
    """
    Copy of the function number_c12_atoms() from NC_background_functions.py.

    function to calculate the number of C12 atoms in the JUNO liquid scintillator for a specific volume of the central
    detector.

    :param radius_cut: radius, which defines the fiducial volume in the central detector in meter, normally 17m is used
    as radius of the fiducial volume like in the calculation of the IBD detection efficiency on page 39 of the yellow
    book (float)
    :return: number of C12 atoms (float)
    """
    # there are 20 ktons LS (which mainly contains LAB) in the central detector with R = 17.7 m.
    # mass LS in ktons:
    mass_ls = 20
    # radius of central detector in meter:
    radius_cd = 17.7
    # INFO-me: approximation, that LS consists only of LAB
    # mass of LS for volume cut with 'radius_cut' in tons:
    mass_ls_cut = radius_cut**3 / radius_cd**3 * mass_ls * 1000

    # the number of C12 atoms depends on the structure formula of LAB. LAB is C_6 H_5 C_n H_(2n+1), where n = 10, 11,
    # 12 or 13. Therefore the number of C12 atoms must be calculated for n= 10, 11, 12 and 13.

    " number of C12 in one LAB molecule: "
    num_c12_lab_n10 = 16
    num_c12_lab_n11 = 17
    num_c12_lab_n12 = 18
    num_c12_lab_n13 = 19

    # atomic mass number u in tons:
    u_in_tons = 1.6605 * 10**(-30)

    " mass of one LAB molecule in tons "
    mass_lab_n10 = (num_c12_lab_n10*12 + 26*1) * u_in_tons
    mass_lab_n11 = (num_c12_lab_n11*12 + 28*1) * u_in_tons
    mass_lab_n12 = (num_c12_lab_n12*12 + 30*1) * u_in_tons
    mass_lab_n13 = (num_c12_lab_n13*12 + 32*1) * u_in_tons

    # number of C12 for different n:
    number_c12_n10 = mass_ls_cut / mass_lab_n10 * num_c12_lab_n10
    number_c12_n11 = mass_ls_cut / mass_lab_n11 * num_c12_lab_n11
    number_c12_n12 = mass_ls_cut / mass_lab_n12 * num_c12_lab_n12
    number_c12_n13 = mass_ls_cut / mass_lab_n13 * num_c12_lab_n13

    # to calculate the number of C12 atoms for this fiducial volume, take the average:
    number_c12_cut = (number_c12_n10 + number_c12_n11 + number_c12_n12 + number_c12_n13) / 4

    return number_c12_cut


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


def limit_annihilation_crosssection_v2(s_90, dm_mass, j_avg, n_target, time_in_sec, epsilon_ibd, epsilon_muveto,
                                       epsilon_psd, mass_neutron, mass_proton, mass_positron):
    """
    New in v2: muon veto efficiency and PSD efficiency is considered correctly!!!

    Function to calculate the 90 percent upper probability limit of the averaged self-annihilation cross-section
    times the relative velocity of the annihilating particles.

    :param s_90: 90 percent upper probability limit of the signal contribution (from output_analysis_v1.py) (float)
    :param dm_mass: Dark matter mass in MeV (float)
    :param j_avg: angular-averaged dark matter intensity over the whole Milky Way (float)
    :param n_target: number of targets in the JUNO detector, equivalent to the number of free protons (float)
    :param time_in_sec: exposure time in seconds (float)
    :param epsilon_ibd: detection efficiency of the JUNO detector for Inverse Beta Decay (float)
    :param epsilon_muveto: muon veto efficiency from muon veto cut (float)
    :param epsilon_psd: PSD efficiency for the specific DM mass
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
              (j_avg * r_solar * rho_0**2 * ibd_crosssection * n_target * time_in_sec
               * epsilon_ibd * epsilon_muveto * epsilon_psd))

    return result


def limit_neutrino_flux(s_90, dm_mass, n_target, time_in_sec, epsilon_ibd, mass_neutron, mass_proton, mass_positron):
    """
    Function to calculate the 90 percent upper probability limit of the electron-antineutrino flux from
    DM annihilation in the Milky Way.

    :param s_90: 90 percent upper probability limit of the signal contribution (from output_analysis_v1.py) (float)
    :param dm_mass: Dark matter mass in MeV (float)
    :param n_target: number of targets in the JUNO detector, equivalent to the number of free protons (float)
    :param time_in_sec: exposure time in seconds (float)
    :param epsilon_ibd: detection efficiency of the JUNO detector for Inverse Beta Decay (float)
    :param mass_neutron: mass of the neutron in MeV (float)
    :param mass_proton: mass of the proton in MeV (float)
    :param mass_positron: mass of the positron in MeV (float)

    :return: 90 % upper probability limit of the electron-antineutrino flux (float)
    """
    # Inverse Beta Decay cross-section in cm**2 (float):
    ibd_crosssection = sigma_ibd(dm_mass, mass_neutron-mass_proton, mass_positron)

    # calculate the 90 % upper limit of the electron-antineutrino flux from DM annihilation in the Milky Way in
    # electron-antineutrinos/(cm**2 s) (float):
    result = s_90 / (ibd_crosssection * n_target * time_in_sec * epsilon_ibd)

    return result


def limit_neutrino_flux_v2(s_90, dm_mass, n_target, time_in_sec, epsilon_ibd, epsilon_muveto,
                           epsilon_psd, mass_neutron, mass_proton, mass_positron):
    """
    New in v2: muon veto efficiency and PSD efficiency is considered correctly!!!

    Function to calculate the 90 percent upper probability limit of the electron-antineutrino flux from
    DM annihilation in the Milky Way.

    :param s_90: 90 percent upper probability limit of the signal contribution (from output_analysis_v1.py) (float)
    :param dm_mass: Dark matter mass in MeV (float)
    :param n_target: number of targets in the JUNO detector, equivalent to the number of free protons (float)
    :param time_in_sec: exposure time in seconds (float)
    :param epsilon_ibd: detection efficiency of the JUNO detector for Inverse Beta Decay (float)
    :param epsilon_muveto: muon veto efficiency from muon veto cut (float)
    :param epsilon_psd: PSD efficiency for the specific DM mass
    :param mass_neutron: mass of the neutron in MeV (float)
    :param mass_proton: mass of the proton in MeV (float)
    :param mass_positron: mass of the positron in MeV (float)

    :return: 90 % upper probability limit of the electron-antineutrino flux (float)
    """
    # Inverse Beta Decay cross-section in cm**2 (float):
    ibd_crosssection = sigma_ibd(dm_mass, mass_neutron-mass_proton, mass_positron)

    # calculate the 90 % upper limit of the electron-antineutrino flux from DM annihilation in the Milky Way in
    # electron-antineutrinos/(cm**2 s) (float):
    result = s_90 / (ibd_crosssection * n_target * time_in_sec * epsilon_ibd * epsilon_muveto * epsilon_psd)

    return result

