""" Script to calculate the atmospheric neutrino flux at the location of JUNO in the energy range from 10 to 100 MeV

    Honda has simulated the nu_E_bar and nu_Mu_bar flux for solar minimum and solar maximum in the energy range from
    100 MeV to 10 TeV at the location of JUNO.

    The FLUKA simulations has calculated the nu_E_bar and nu_Mu_bar flux for solar average in the energy range from
    13 MeV to 944 MeV at the location of Super-K.

    In this script the two simulation results are put together to get the nu_E_bar and nu_Mu_bar flux for solar average
    in the energy range from 13 MeV to 100 MeV at the location of JUNO.

"""

import numpy as np
from matplotlib import pyplot as plt

# TODO-me: Why do the flux differs from Julia's one????

# define the neutrino energy in MeV:
energy_interesting = np.arange(13, 115.01, 0.01)
energy_total = np.arange(13, 944.01, 0.01)
energy_norm = np.arange(100, 944.1, 0.1)

""" Results of the FLUKA simulation (from the paper of Battistoni2005 'The atmospheric neutrino fluxes below 100 MeV: 
    The FLUKA results'). 
    """
""" These fluxes are used in the simulation of the atmoCC background with the function ccatmospheric_background_v2().
    
    Assumptions or estimations made for the simulation of the atmospheric anti-neutrino flux by FLUKA simulation:
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
            1.5 INFO-me: there is a difference is flux between the FLUKA results and the results of the Bartol group
                -> three main ingredients: the primary spectrum, the particle production model, and the 3-dimensional
                spherical geometrical representation of Earth and atmosphere vs. the flat geometry
                INFO-me: the primary spectrum is known to produce more neutrinos at low energy with respect to
                previous choices
            1.6 INFO-me: the systematic error on particle production in FLUKA in the whole energy region
                between pion threshold and 100 MeV is not larger than 15 % (from comparison with accelerator results)
            1.7 INFO-me: the most important uncertainties are still related to the knowledge of the primary spectrum
                and in part to the hadronic interaction models
                
            1.8 at the site of Super-K (geographical latitude of 36.4°N) (geo. latitude of JUNO = 22.6°N)
                INFO-me: -> Flux is overestimated a bit, because of the lower geographical latitude of JUNO
                
    INFO-me: the overall uncertainty on the absolute value of the fluxes is estimated to be smaller than 25 %
    
    2. this data is linear interpolated to get the differential flux corresponding to the binning in E1,
        INFO-me: it is estimated, that for e1_atmo = 0 MeV flux_atmo is also 0.
    """

# Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
energy_FLUKA = np.array([13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133, 150,
                         168, 188, 211, 237, 266, 299, 335, 376, 422, 473, 531, 596, 668, 750, 841, 944])

# differential flux in energy for no oscillation for electron-antineutrinos for solar average at the site
# of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuEbar_FLUKA = 10 ** (-4) * np.array([63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                           83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9, 21.3,
                                           18.3, 15.4, 12.9, 10.6, 8.80, 7.13, 5.75, 4.60, 3.68, 2.88, 2.28,
                                           1.87, 1.37, 1.06, 0.800])

# differential flux in energy for no oscillation for muon-antineutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuMubar_FLUKA = 10 ** (-4) * np.array([116., 128., 136., 150., 158., 162., 170., 196., 177., 182., 183.,
                                            181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0, 55.6, 47.6,
                                            40.8, 34.1, 28.6, 23.5, 19.3, 15.7, 12.6, 10.2, 8.15, 6.48, 5.02, 3.94,
                                            3.03, 2.33, 1.79])

""" Results of the HONDA simulation (based on the paper of Honda2015: 'Atmospheric neutrino flux calculation using the
    NRLMSISE-00 atmospheric model').
    """
""" Assumptions or estimations made for the simulation of the atmospheric anti-neutrino flux simulation by HONDA:
    1. the NRLMSISE-00 global atmospheric model of Picone (2002) is used, replacing the U.S.-standard 1976 atmospheric
       model, which has no positional or seasonal variations.
       
    2. the international geomagnetic reference field (IGRF) geomagnetic field model is used in their calculation.
    2.1 the geomagnetic field strongly affects the atmospheric neutrino flux, and is largely different in the polar
        and equatorial regions. 
    2.2 The extension in the paper is also the study of atmospheric neutrino flux under these widely different 
        geomagnetic field conditions
    
    3.  "In our 3D calculations of the atmospheric neutrino flux, we followed the motion of all the cosmic rays, which
        penetrate the rigidity cutoff, and their secondaries. Then we examine all the neutrinos produced during their
        propagation in the atmosphere and register the neutrinos which hit the virtual detector assumed around 
        the target neutrino observation site." (page 2 of the paper)
        
    4. "NRLMSISE-00 is an empirical, global model of the Earth’s atmosphere from ground to space. It models the
        temperatures and densities of the atmosphere’s components. However, the air density profile is the most 
        important quantity in the calculation of atmospheric neutrino flux." (page 3 of the paper)
        
    5. Error estimation:
        "The total error is a little lower than 10% in the energy region 1–10 GeV. The error increases outside of this
         energy region due to the small number of available muon observation data at the lower energies, and due to 
         the uncertainty of kaon production at higher energies." (page of the paper)
         
         
    Atmospheric Neutrino Flux Tables for One-Year-Average (HAKKM, 2014) 
    -> All-direction averaged flux for JUNO WITHOUT mountain over the detector (solar minimum and solar maximum)

    """


# Neutrino energy in MeV from the table from file HONDA_juno-ally-01-01-solmin.d (is equal to neutrino energy
# in HONDA_juno-ally-01-01-solmax.d) (np.array of float):
energy_HONDA = 10**3 * np.array([1.0000E-01, 1.1220E-01, 1.2589E-01, 1.4125E-01, 1.5849E-01, 1.7783E-01, 1.9953E-01,
                                 2.2387E-01, 2.5119E-01, 2.8184E-01, 3.1623E-01, 3.5481E-01, 3.9811E-01, 4.4668E-01,
                                 5.0119E-01, 5.6234E-01, 6.3096E-01, 7.0795E-01, 7.9433E-01, 8.9125E-01, 1.0000E+00])

""" for solar minimum (HONDA_juno-ally-01-01-solmin.d): """
# all-direction averaged flux for no oscillation for electron-antineutrinos for solar minimum at the site of JUNO
# (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
# TODO: is the factor 4*pi correct corresponding to 1/sr?
flux_nuEbar_MIN_HONDA = 10**(-7) * 4*np.pi * np.array([2.9367E+03, 2.5746E+03, 2.2332E+03, 1.9206E+03, 1.6395E+03,
                                                       1.3891E+03, 1.1679E+03, 9.7454E+02, 8.0732E+02, 6.6312E+02,
                                                       5.4052E+02, 4.3731E+02, 3.5122E+02, 2.8033E+02, 2.2264E+02,
                                                       1.7581E+02, 1.3804E+02, 1.0776E+02, 8.3623E+01, 6.4555E+01,
                                                       4.9632E+01])

# all-direction averaged flux for no oscillation for muon-antineutrinos for solar minimum at the site of JUNO
# (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuMubar_MIN_HONDA = 10**(-7) * 4*np.pi * np.array([6.2903E+03, 5.5084E+03, 4.8032E+03, 4.1620E+03, 3.5763E+03,
                                                        3.0444E+03, 2.5663E+03, 2.1426E+03, 1.7736E+03, 1.4575E+03,
                                                        1.1890E+03, 9.6400E+02, 7.7693E+02, 6.2283E+02, 4.9647E+02,
                                                        3.9325E+02, 3.1003E+02, 2.4324E+02, 1.9004E+02, 1.4788E+02,
                                                        1.1447E+02])

""" for solar maximum (HONDA_juno-ally-01-01-solmax.d): """
# all-direction averaged flux for no oscillation for electron-antineutrinos for solar maximum at the site of JUNO
# (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuEbar_MAX_HONDA = 10**(-7) * 4*np.pi * np.array([2.7733E+03, 2.4332E+03, 2.1124E+03, 1.8187E+03, 1.5545E+03,
                                                       1.3190E+03, 1.1105E+03, 9.2820E+02, 7.7040E+02, 6.3403E+02,
                                                       5.1790E+02, 4.1997E+02, 3.3811E+02, 2.7054E+02, 2.1539E+02,
                                                       1.7049E+02, 1.3418E+02, 1.0499E+02, 8.1651E+01, 6.3166E+01,
                                                       4.8654E+01])

# all-direction averaged flux for no oscillation for muon-antineutrinos for solar maximum at the site of JUNO
# (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuMubar_MAX_HONDA = 10**(-7) * 4*np.pi * np.array([5.8966E+03, 5.1676E+03, 4.5104E+03, 3.9127E+03, 3.3665E+03,
                                                        2.8701E+03, 2.4238E+03, 2.0277E+03, 1.6821E+03, 1.3857E+03,
                                                        1.1333E+03, 9.2144E+02, 7.4476E+02, 5.9875E+02, 4.7865E+02,
                                                        3.8024E+02, 3.0060E+02, 2.3645E+02, 1.8519E+02, 1.4444E+02,
                                                        1.1204E+02])

# all-direction averaged flux for no oscillation for electron-antineutrinos for solar AVERAGE at the site of JUNO
# (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_atmo_nuEbar_HONDA = (flux_nuEbar_MIN_HONDA + flux_nuEbar_MAX_HONDA) / 2

# all-direction averaged flux for no oscillation for muon-antineutrinos for solar AVERAGE at the site of JUNO
# (WITHOUT mountain over the detector), in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_atmo_nuMubar_HONDA = (flux_nuMubar_MIN_HONDA + flux_nuMubar_MAX_HONDA) / 2


""" Extrapolate the HONDA flux to the energies of the FLUKA simulation from 10 MeV to 100 MeV: """
""" Assumption:
    1. the shape of the FLUKA flux as function of energy do NOT depend on the location
        -> the shape of the flux at Super-K can also be used at JUNO site
        
    2. the absolute value of the FLUKA flux at Super-K should be normalized to the location of JUNO
        ->  therefore get the normalization factor by comparing the HONDA flux and the FLUKA flux in the energy range 
            from 100 MeV to 1 GeV
            
    """

""" For electron antineutrinos: """
# Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuEbar_FLUKA_interpolated = np.interp(energy_norm, energy_FLUKA, flux_nuEbar_FLUKA)

# Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuEbar_HONDA_interpolated = np.interp(energy_norm, energy_HONDA, flux_atmo_nuEbar_HONDA)

# Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
integral_nuEbar_FLUKA = np.trapz(flux_nuEbar_FLUKA_interpolated, energy_norm)

# Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
integral_nuEbar_HONDA = np.trapz(flux_nuEbar_HONDA_interpolated, energy_norm)

# Interpolate the INTERESTING part og the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*sÜcm**2)
# (np.array of float):
flux_nuEbar_FLUKA_interesting = np.interp(energy_interesting, energy_FLUKA, flux_nuEbar_FLUKA)

# Normalize flux_nuEbar_FLUKA_interesting at Super-K to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2)
# (np.array of float):
flux_nuEbar_JUNO = flux_nuEbar_FLUKA_interesting * integral_nuEbar_HONDA / integral_nuEbar_FLUKA

# Normalize the whole FLUKA flux to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2), (contains the
# interesting part from 13 MeV to 115 MeV and the part from 113 MeV to 944 MeV to compare it to HONDA):
flux_nuEbar_FLUKA_normalized = flux_nuEbar_FLUKA * integral_nuEbar_HONDA / integral_nuEbar_FLUKA


""" For muon antineutrinos: """
# Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuMubar_FLUKA_interpolated = np.interp(energy_norm, energy_FLUKA, flux_nuMubar_FLUKA)

# Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuMubar_HONDA_interpolated = np.interp(energy_norm, energy_HONDA, flux_atmo_nuMubar_HONDA)

# Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
integral_nuMubar_FLUKA = np.trapz(flux_nuMubar_FLUKA_interpolated, energy_norm)

# Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
integral_nuMubar_HONDA = np.trapz(flux_nuMubar_HONDA_interpolated, energy_norm)

# Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
# (np.array of float):
flux_nuMubar_FLUKA_interesting = np.interp(energy_interesting, energy_FLUKA, flux_nuMubar_FLUKA)

# Normalize flux_nuEbar_FLUKA_interesting at Super-K to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2)
# (np.array of float):
flux_nuMubar_JUNO = flux_nuMubar_FLUKA_interesting * integral_nuMubar_HONDA / integral_nuMubar_FLUKA

# Normalize the whole FLUKA flux to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2), (contains the
# interesting part from 13 MeV to 115 MeV and the part from 113 MeV to 944 MeV to compare it to HONDA):
flux_nuMubar_FLUKA_normalized = flux_nuMubar_FLUKA * integral_nuMubar_HONDA / integral_nuMubar_FLUKA


""" Taking account neutrino oscillation from the Appendix B of the paper of Fogli et al. from 2004 with title
    "Three-generation flavor transitions and decays of supernova relic neutrinos" (like in ccatmospheric_background_v2):
    """
# survival probability of electron-antineutrinos (prob_e_to_e = 0.67):
prob_e_to_e = 0.67
# oscillation probability (muon-antineutrinos oscillate to electron-antineutrinos) (prob_mu_to_e = 0.17):
prob_mu_to_e = 0.17

# total electron-antineutrino flux at Super-K in 1/(MeV * cm**2 * s) (np.array of float):
flux_FLUKA = prob_e_to_e * flux_nuEbar_FLUKA + prob_mu_to_e * flux_nuMubar_FLUKA

# total electron-antineutrino flux at JUNO in 1/(MeV * cm**2 * s) (np.array of float):
flux_HONDA = prob_e_to_e * flux_atmo_nuEbar_HONDA + prob_mu_to_e * flux_atmo_nuMubar_HONDA

# total electron-antineutrino flux of FLUKA simulation normalized to JUNO site (HONDA) in 1/(MeV * cm**2 * s)
# (np.array of float):
flux_FLUKA_normalized = prob_e_to_e * flux_nuEbar_FLUKA_normalized + prob_mu_to_e * flux_nuMubar_FLUKA_normalized

# total electron-antineutrino flux in the INTERESTING part of FLUKA simulation normalized to JUNO site (HONDA) in
# 1/(MeV * cm**2 * s), (np.array of float):
flux_JUNO = prob_e_to_e * flux_nuEbar_JUNO + prob_mu_to_e * flux_nuMubar_JUNO


""" display the different fluxes: """
h1 = plt.figure(1)
plt.plot(energy_FLUKA, flux_nuEbar_FLUKA, 'rx:', label='nu_e_bar flux at Super-K (FLUKA)')
plt.plot(energy_HONDA, flux_atmo_nuEbar_HONDA, 'bx:', label='nu_e_bar flux at JUNO (HONDA)')
plt.plot(energy_FLUKA, flux_nuEbar_FLUKA_normalized, 'g:', label='nu_e_bar flux of FLUKA normalized to JUNO site')
plt.plot(energy_FLUKA, flux_nuMubar_FLUKA, 'rx--', label='nu_mu_bar flux at Super-K (FLUKA)')
plt.plot(energy_HONDA, flux_atmo_nuMubar_HONDA, 'bx--', label='nu_mu_bar flux at JUNO (HONDA)')
plt.plot(energy_FLUKA, flux_nuMubar_FLUKA_normalized, 'g--', label='nu_mu_bar of FLUKA normalized to JUNO site')
plt.xlabel('neutrino energy in MeV')
plt.ylabel('differential neutrino flux in 1/(MeV cm² s)')
plt.title('Antineutrino fluxes for solar average at different locations WITHOUT oscillation')
plt.legend()
plt.grid()

h2 = plt.figure(2)
plt.semilogy(energy_FLUKA, flux_nuEbar_FLUKA, 'r:', label='nu_e_bar flux at Super-K (FLUKA)')
plt.semilogy(energy_HONDA, flux_atmo_nuEbar_HONDA, 'b:', label='nu_e_bar flux at JUNO (HONDA)')
plt.semilogy(energy_FLUKA, flux_nuEbar_FLUKA_normalized, 'g:', label='nu_e_bar flux of FLUKA normalized to JUNO site')
plt.semilogy(energy_FLUKA, flux_nuMubar_FLUKA, 'r--', label='nu_mu_bar flux at Super-K (FLUKA)')
plt.semilogy(energy_HONDA, flux_atmo_nuMubar_HONDA, 'b--', label='nu_mu_bar flux at JUNO (HONDA)')
plt.semilogy(energy_FLUKA, flux_nuMubar_FLUKA_normalized, 'g--', label='nu_mu_bar of FLUKA normalized to JUNO site')
plt.xlabel('neutrino energy in MeV')
plt.ylabel('differential neutrino flux in 1/(MeV cm² s)')
plt.title('Antineutrino fluxes for solar average at different locations WITHOUT oscillation')
plt.legend()
plt.grid()

h3 = plt.figure(3)
plt.plot(energy_FLUKA, flux_FLUKA, 'rx-', label='total nu_e_bar flux at Super-K (FLUKA)')
plt.plot(energy_HONDA, flux_HONDA, 'bx-', label='total nu_e_bar flux at JUNO (HONDA)')
plt.plot(energy_FLUKA, flux_FLUKA_normalized, 'g-', label='total nu_e_bar flux of FLUKA normalized to JUNO site')
plt.xlabel('neutrino energy in MeV')
plt.ylabel('differential neutrino flux in 1/(MeV cm² s)')
plt.title('Total electron-antineutrino fluxes for solar average at different locations WITH oscillation')
plt.legend()
plt.grid()

h4 = plt.figure(4)
plt.semilogy(energy_FLUKA, flux_FLUKA, 'r-', label='total nu_e_bar flux at Super-K (FLUKA)')
plt.semilogy(energy_HONDA, flux_HONDA, 'b-', label='total nu_e_bar flux at JUNO (HONDA)')
plt.semilogy(energy_FLUKA, flux_FLUKA_normalized, 'g-', label='total nu_e_bar flux of FLUKA normalized to JUNO site')
plt.xlabel('neutrino energy in MeV')
plt.ylabel('differential neutrino flux in 1/(MeV cm² s)')
plt.title('Total electron-antineutrino fluxes for solar average at different locations WITH oscillation')
plt.legend()
plt.grid()

h5 = plt.figure(5)
plt.plot(energy_HONDA, flux_nuEbar_MIN_HONDA, 'k:', label='nu_e_bar flux for solar minimum')
plt.plot(energy_HONDA, flux_nuEbar_MAX_HONDA, 'k--', label='nu_e_bar flux for solar maximum')
plt.plot(energy_HONDA, flux_nuMubar_MIN_HONDA, 'm:', label='nu_mu_bar flux for solar minimum')
plt.plot(energy_HONDA, flux_nuMubar_MAX_HONDA, 'm--', label='nu_mu_bar flux for solar maximum')
plt.xlabel('neutrino energy in MeV')
plt.ylabel('differential neutrino flux in 1/(MeV cm² s)')
plt.title('Antineutrino fluxes at JUNO site for solar minimum and solar maximum, respectively \n'
          '(HONDA simulation, without oscillation)')
plt.legend()
plt.grid()
plt.show()
