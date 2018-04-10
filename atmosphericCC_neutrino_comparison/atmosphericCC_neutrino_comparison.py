""" Script to observe the effect of neutrino oscillation to the low energy Charged-Current atmospheric
    electron-antineutrino background in JUNO.
    Oscillation-calculation of different paper are considered:
    - Battistoni's paper 'The atmospheric neutrino flux below 100 MeV: the FLUKA results' from 2005
    (1-s2.0-S0927650505000526-main)
    - paper of Peres and Smirnov 'Oscillation of very low energy atmospheric neutrinos' from 2009 (arXiv: 0903.5323)
    - paper of Fogli, Lisi, Mirizzi and Montanino 'Three-generation flavor transition and decays of supernova
    relic neutrinos' from 2004 (arXiv: hep-ph/0401227)
    - paper of Barr 'Flux of atmospheric neutrinos' from 1989 (PhysRevD.39.3532.pdf)
    and paper of Gaisser 'Cosmic-ray neutrinos in the atmosphere' from 1988 (PhysRevD.38.85.pdf)
"""


import numpy as np
from matplotlib import pyplot


def sigma_ibd(e, delta, mass_positron):
    """ IBD cross-section in cm**2 for neutrinos with E=energy, equation (25) from paper 0302005_IBDcrosssection:
        simple approximation which agrees with full result of paper within few per-mille
        for neutrino energies <= 300 MeV:
        energy: energy corresponding to the electron-antineutrino energy in MeV (np.array of float OR float)
        delta: difference mass_neutron minus mass_proton in MeV (float)
        mass_positron: mass of the positron in MeV (float)
        """
    # positron energy defined as energy - delta in MeV (np.array of float64 or float):
    energy_positron = e - delta
    # positron momentum defined as sqrt(energy_positron**2-mass_positron**2) in MeV (np.array of float64 or float):
    momentum_positron = np.sqrt(energy_positron ** 2 - mass_positron ** 2)
    # IBD cross-section in cm**2 (array of float64 or float):
    sigma = (10 ** (-43) * momentum_positron * energy_positron *
             e ** (-0.07056 + 0.02018 * np.log(e) - 0.001953 * np.log(e) ** 3))
    return sigma


# electron-antineutrino energy in MeV:
energy = np.arange(10, 105, 0.01)
# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON
# total exposure time in seconds (float):
time = 10 * 3.156 * 10 ** 7
# total exposure time in years (float):
t_years = 10
# Number of free protons (target particles) for IBD in JUNO (float):
N_target = 1.45 * 10 ** 33
# detection efficiency of IBD in JUNO, from physics_report.pdf, page 40, table 2.1
# (combined efficiency of energy cut, time cut, vertex cut, Muon veto, fiducial volume) (float):
detection_eff = 0.73

""" atmospheric CC electron- and muon-antineutrino fluxes are taken from table 2 and 3 of Battistoni's paper: 
    Assumptions:
    - solar average
    - at site of Super Kamiokande
    - this data is linear interpolated to get the differential flux corresponding to the binning in E1,
    it is estimated that for e1_atmo = 0 MeV, flux_atmo is also 0
"""
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


""" atmospheric CC electron- and muon-antineutrino fluxes are taken from table 2 of Barr's paper:
    Assumptions:
    - solar maximum
    - angular averaged
    - at site of Kamiokande
    - fluxes are in the unit of 1/(cm**2 * s * MeV * sr)
    """
# neutrino energy in GeV from table 2:
e2_atmo = np.array([0.07, 0.09, 0.12, 0.17])
# neutrino energy in MeV:
e2_atmo = e2_atmo * 1000
# angular-averaged fluxes for electron-antineutrinos at solar maximum in (m**2 sr s MeV)**-1:
flux_ccatmo_nu_e_bar_Barr = np.array([87/20, 72/20, 109/40, 111/60])
# angular-averaged flux in (cm**2 * s * MeV)**-1:
flux_ccatmo_nu_e_bar_Barr = flux_ccatmo_nu_e_bar_Barr/10000*(4*np.pi)
# angular-averaged fluxes for muon-antineutrinos at solar maximum in (m**2 sr s MeV)**-1:
flux_ccatmo_nu_mu_bar_Barr = np.array([190/20, 156/20, 235/40, 236/60])
# angular-averaged flux in (cm**2 * s * MeV)**-1:
flux_ccatmo_nu_mu_bar_Barr = flux_ccatmo_nu_mu_bar_Barr/10000*(4*np.pi)


""" NO oscillation is considered: P(nu_e_bar -> nu_e_bar) = 1 and P(nu_mu_bar -> nu_e_bar) = 0: (paper Battistoni) """
# Survival probability of electron-antineutrino to electron-antineutrino:
P_ee_battistoni = 1.0
# Probability of muon-antineutrino to electron-antineutrino:
P_mue_battistoni = 0.0
# Total electron-antineutrino flux at site of Super-Kamiokande WITH oscillation considered (in units of 1/(MeV*cm**2*s):
flux_ccatmo_noosc = flux_ccatmo_nu_e_bar * P_ee_battistoni + flux_ccatmo_nu_mu_bar * P_mue_battistoni
# spectrum of electron-antineutrinos in JUNO after 10 years (in units of 1/MeV):
spectrum_noosc = flux_ccatmo_noosc * sigma_ibd(energy, DELTA, MASS_POSITRON) * detection_eff * N_target * time

""" WITH oscillation like described in paper of Peres and Smirnov (in the two layer case and for very low energies): """
# used values:
sin2_th13_peres = 0.0214
sin2_th23_peres = 0.437
cp_phase_delta_peres = 1.35 * np.pi
# Survival probability of electron-antineutrino to electron-antineutrino:
P_ee_peres = 0.558
# Probability of muon-antineutrino to electron-antineutrino:
P_mue_peres = 0.2205
# Total electron-antineutrino flux at site of Super-Kamiokande WITH oscillation considered (in units of 1/(MeV*cm**2*s):
flux_ccatmo_peres = flux_ccatmo_nu_e_bar * P_ee_peres + flux_ccatmo_nu_mu_bar * P_mue_peres
# spectrum of electron-antineutrinos in JUNO after 10 years (in units of 1/MeV):
spectrum_peres = flux_ccatmo_peres * sigma_ibd(energy, DELTA, MASS_POSITRON) * detection_eff * N_target * time

""" WITH oscillation like described in paper of Fogli in the Appendix 
    (oscillation for very low neutrino energies (<< 1 GeV): """
# Assumptions:
# - isotropic atmospheric neutrino flux (produced at h=20km from ground level
# - equal components of neutrino and antineutrino
# - matter effects are estimated through a constant-density approximation for the Earth
# - consider representative neutrino energy E = 100 MeV
# - average over all incoming neutrino directions
# - oscillation parameters used:
sin2_th12_fogli = 0.29
sin2_th23_fogli = 0.50
delta_m2 = 7.2e-5  # in eV**2
Delta_m2 = 2.0e-3  # in eV**2
# Two different values for sin2_th13:
sin2_th13_fogli_1 = 0.0
sin2_th13_fogli_2 = 0.067

# Survival probability of electron-antineutrino to electron-antineutrino for sin2_th13_fogli_1:
P_ee_fogli_1 = 0.77
# Probability of muon-antineutrino to electron-antineutrino for sin2_th13_fogli_1:
P_mue_fogli_1 = 0.11
# Total electron-antineutrino flux at site of Super-Kamiokande WITH oscillation considered (in units of 1/(MeV*cm**2*s)
# for sin2_th13_fogli_1:
flux_ccatmo_fogli_1 = flux_ccatmo_nu_e_bar * P_ee_fogli_1 + flux_ccatmo_nu_mu_bar * P_mue_fogli_1
# spectrum of electron-antineutrinos in JUNO after 10 years (in units of 1/MeV) for sin2_th13_fogli_1:
spectrum_fogli_1 = flux_ccatmo_fogli_1 * sigma_ibd(energy, DELTA, MASS_POSITRON) * detection_eff * N_target * time

# Survival probability of electron-antineutrino to electron-antineutrino for sin2_th13_fogli_2:
P_ee_fogli_2 = 0.67
# Probability of muon-antineutrino to electron-antineutrino for sin2_th13_fogli_2:
P_mue_fogli_2 = 0.17
# Total electron-antineutrino flux at site of Super-Kamiokande WITH oscillation considered (in units of 1/(MeV*cm**2*s)
# for sin2_th13_fogli_2:
flux_ccatmo_fogli_2 = flux_ccatmo_nu_e_bar * P_ee_fogli_2 + flux_ccatmo_nu_mu_bar * P_mue_fogli_2
# spectrum of electron-antineutrinos in JUNO after 10 years (in units of 1/MeV) for sin2_th13_fogli_2:
spectrum_fogli_2 = flux_ccatmo_fogli_2 * sigma_ibd(energy, DELTA, MASS_POSITRON) * detection_eff * N_target * time


""" Calculate the deviation of the flux with oscillation and without oscillation in percent: """
difference_noOsc = (flux_ccatmo_noosc - flux_ccatmo_noosc) / flux_ccatmo_noosc*100
difference_Peres = (flux_ccatmo_peres - flux_ccatmo_noosc) / flux_ccatmo_noosc*100
difference_Fogli_1 = (flux_ccatmo_fogli_1 - flux_ccatmo_noosc) / flux_ccatmo_noosc*100
difference_Fogli_2 = (flux_ccatmo_fogli_2 - flux_ccatmo_noosc) / flux_ccatmo_noosc*100

""" Deviation of the Peres-flux and the Fogli-1-flux to the flux of Fogli-2 in percent: """
deviation_peres = (flux_ccatmo_peres - flux_ccatmo_fogli_2) / flux_ccatmo_fogli_2*100
deviation_fogli_1 = (flux_ccatmo_fogli_1 - flux_ccatmo_fogli_2) / flux_ccatmo_fogli_2*100
deviation_fogli_2 = (flux_ccatmo_fogli_2 - flux_ccatmo_fogli_2) / flux_ccatmo_fogli_2*100

# Display the electron-antineutrino fluxes with the settings below:
h1 = pyplot.figure(1)
pyplot.plot(energy, flux_ccatmo_noosc, label='no oscillation (Battistoni nu_e_bar)')
pyplot.plot(energy, flux_ccatmo_peres, label='Peres (P_ee={0:.3f}, P_mue={1:.3f}'.format(P_ee_peres, P_mue_peres))
pyplot.plot(energy, flux_ccatmo_fogli_1, label='Fogli (sin2_th13 = 0)')
pyplot.plot(energy, flux_ccatmo_fogli_2, label='Fogli (sin2_th13 = 0.067)')
pyplot.ylim(ymin=0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Electron-antineutrino flux in 1/(MeV * cm**2 * s)")
pyplot.title("Flux of atmospheric CC electron-antineutrinos for different oscillation probabilities")
pyplot.legend()
pyplot.grid()

# Display the electron-antineutrino fluxes with the settings below:
h3 = pyplot.figure(3)
pyplot.plot(energy, flux_ccatmo_nu_e_bar, label=' Battistoni nu_e_bar')
pyplot.plot(energy, flux_ccatmo_nu_mu_bar, label='Battistoni nu_mu_bar')
pyplot.plot(e2_atmo, flux_ccatmo_nu_e_bar_Barr, '+-', label='Barr nu_e_bar')
pyplot.plot(e2_atmo, flux_ccatmo_nu_mu_bar_Barr, '+-', label='Barr nu_mu_bar')
pyplot.ylim(ymin=0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Electron-antineutrino flux in 1/(MeV * cm**2 * s)")
pyplot.title("Flux of atmospheric CC electron- and muon-antineutrinos without oscillation")
pyplot.legend()
pyplot.grid()


# Display the electron-antineutrino spectrum at the detector with the setting below:
h2 = pyplot.figure(2)
pyplot.plot(energy, spectrum_noosc, label='no oscillation')
pyplot.plot(energy, spectrum_peres, label='Peres (P_ee={0:.3f}, P_mue={1:.3f})'.format(P_ee_peres, P_mue_peres))
pyplot.plot(energy, spectrum_fogli_1, label='Fogli (sin2_th13 = 0)')
pyplot.plot(energy, spectrum_fogli_2, label='Fogli (sin2_th13 = 0.067)')
pyplot.ylim(ymin=0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Electron-antineutrino spectrum in 1/(MeV)")
pyplot.title("Spectrum of atmospheric CC electron-antineutrinos at JUNO detector after 10 years for different "
             "oscillation probabilities")
pyplot.legend()
pyplot.grid()


# Display the deviation of the electron-antineutrino fluxes with the settings below:
h4 = pyplot.figure(4)
pyplot.plot(energy, difference_noOsc, label='no oscillation (Battistoni nu_e_bar)')
pyplot.plot(energy, difference_Peres, label='Peres (P_ee={0:.3f}, P_mue={1:.3f})'.format(P_ee_peres, P_mue_peres))
pyplot.plot(energy, difference_Fogli_1, label='Fogli (sin2_th13 = 0)')
pyplot.plot(energy, difference_Fogli_2, label='Fogli (sin2_th13 = 0.067)')
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Deviation of the flux with oscillation to no oscillation in percent")
pyplot.title("Deviation of the flux of atmospheric CC electron-antineutrinos to the flux without oscillation")
pyplot.legend()
pyplot.grid()


# Display the deviation of the electron-antineutrino fluxes with the settings below:
h5 = pyplot.figure(5)
pyplot.plot(energy, deviation_peres, label='Peres (P_ee={0:.3f}, P_mue={1:.3f})'.format(P_ee_peres, P_mue_peres))
pyplot.plot(energy, deviation_fogli_1, label='Fogli (sin2_th13 = 0)')
pyplot.plot(energy, deviation_fogli_2, label='Fogli (sin2_th13 = 0.067)')
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Deviation of the flux of Peres and Fogli-1 to the flux of Fogli-2 in percent")
pyplot.title("Deviation of the flux of atmospheric CC electron-antineutrinos to the flux of Fogli with "
             "sin2_th13 = 0.067")
pyplot.legend()
pyplot.grid()

pyplot.show()
