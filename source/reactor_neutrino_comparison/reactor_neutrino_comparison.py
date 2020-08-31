""" Script to calculate the reactor electron-antineutrino flux and spectrum from different paper:
    - from the offline generator KRLReactorFlux.cc and KRLReactorFlux.hh (is based on VogelEngel1989_PhysRevD.39.3378)
    - from Mueller2011_PhysRevC.83.054615 (is reference [5] in paper Vogel2016_1603.08990)
    - from Huber2011_PhysRevC.84.024617 (is reference [6] in paper Vogel2016_1603.08990)
    - from figure 1 of Fallot2012_PhysRevLett.109.202504 (is reference [26] in Novella2015_1512.03366)
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
energy = np.arange(2, 16, 0.1)
# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON
# total exposure time in years (float):
t_years = 10
# Number of free protons (target particles) for IBD in JUNO (float):
N_target = 1.45 * 10 ** 33
# detection efficiency of IBD in JUNO, from physics_report.pdf, page 40, table 2.1
# (combined efficiency of energy cut, time cut, vertex cut, Muon veto, fiducial volume) (float):
detection_eff = 0.67
# distance reactor to detector in meter:
L_m = 5.25*10**4
# distance reactor to detector in centimeter:
L_cm = L_m * 100
# Total thermal power of the Yangjiang and Taishan nuclear power plants (from PhysicsReport), in GW:
power_th = 35.73
# Fractions (data taken from PhysicsReport_1507.05613, page 136) (averaged value of the Daya Bay nuclear cores):
Fraction235U = 0.577
Fraction239Pu = 0.295
Fraction238U = 0.076
Fraction241Pu = 0.052


""" Flux of reactor electron-antineutrinos calculated with the code described 
in KRLReactorFlux.cc and KRLReactorFlux.hh (based on VogelEngel1989_PhysRevD.39.3378): (the spectra is overestimated 
by a factor of 2 to 3 between 8 and 12 MeV): """
# Coefficients from KRLReactorFlux.cc (data taken from Vogel and Engel 1989 (PhysRevD.39.3378), table 1):
Coeff235U_vogel = np.array([0.870, -0.160, -0.0910])
Coeff239Pu_vogel = np.array([0.896, -0.239, -0.0981])
Coeff238U_vogel = np.array([0.976, -0.162, -0.0790])
Coeff241Pu_vogel = np.array([0.793, -0.080, -0.1085])
# fit to the electron-antineutrino spectrum, in units of electron-antineutrinos/(MeV * fission)
# (from KRLReactorFlux.cc and Vogel/Engel 1989 (PhysRevD.39.3378), equation 4):
U235_vogel = np.exp(Coeff235U_vogel[0] + Coeff235U_vogel[1] * energy + Coeff235U_vogel[2] * energy ** 2)
Pu239_vogel = np.exp(Coeff239Pu_vogel[0] + Coeff239Pu_vogel[1] * energy + Coeff239Pu_vogel[2] * energy ** 2)
U238_vogel = np.exp(Coeff238U_vogel[0] + Coeff238U_vogel[1] * energy + Coeff238U_vogel[2] * energy ** 2)
Pu241_vogel = np.exp(Coeff241Pu_vogel[0] + Coeff241Pu_vogel[1] * energy + Coeff241Pu_vogel[2] * energy ** 2)
# add the weighted sum of the terms, electron-antineutrino spectrum in units of electron-antineutrinos/(MeV * fission)
# (data taken from KRLReactorFLux.cc):
spectrum_vogel = U235_vogel*Fraction235U + Pu239_vogel*Fraction239Pu + U238_vogel*Fraction238U + \
                 Pu241_vogel*Fraction241Pu
# There are 3.125*10**19 fissions/GW/second, spectrum in units of electron-antineutrino/(MeV * GW * s):
spectrum_vogel1 = spectrum_vogel * 3.125*10**19
# There are about 3.156*10**7 seconds in a year, spectrum in units of electron-antineutrino/(MeV * GW * year):
spectrum_vogel_total = spectrum_vogel1 * 3.156*10**7
# electron-antineutrino flux in units of electron-antineutrino/(MeV * year):
flux_vogel = spectrum_vogel_total * power_th


""" Flux of reactor electron-antineutrinos calculated with the fit described in Mueller2011_PhysRevC.83.054615: """
# coefficients of the polynomial fit of order 5 (table VI):
Coeff235U_mueller = np.array([3.217, -3.111, 1.395, -3.690*10**(-1), 4.445*10**(-2), -2.053*10**(-3)])
Coeff238U_mueller = np.array([4.833*10**(-1), 1.927*10**(-1), -1.283*10**(-1), -6.762*10**(-3), 2.233*10**(-3),
                              -1.536*10**(-4)])
Coeff239Pu_mueller = np.array([6.413, -7.432, 3.535, -8.820*10**(-1), 1.025*10**(-1), -4.550*10**(-3)])
Coeff241Pu_mueller = np.array([3.251, -3.204, 1.428, -3.675*10**(-1), 4.254*10**(-2), -1.896*10**(-3)])
# fit to the electron-antineutrino spectrum, in units of electron-antineutrinos/(MeV * fission)
# (fit function equ. (12) of Mueller2011_PhysRevC.83.054615):
U235_mueller = np.exp(Coeff235U_mueller[0] + Coeff235U_mueller[1] * energy +
                      Coeff235U_mueller[2] * energy ** 2 + Coeff235U_mueller[3] * energy ** 3 +
                      Coeff235U_mueller[4] * energy ** 4 + Coeff235U_mueller[5] * energy ** 5)
U238_mueller = np.exp(Coeff238U_mueller[0] + Coeff238U_mueller[1] * energy +
                      Coeff238U_mueller[2] * energy ** 2 + Coeff238U_mueller[3] * energy ** 3 +
                      Coeff238U_mueller[4] * energy ** 4 + Coeff238U_mueller[5] * energy ** 5)
Pu239_mueller = np.exp(Coeff239Pu_mueller[0] + Coeff239Pu_mueller[1] * energy +
                       Coeff239Pu_mueller[2] * energy ** 2 + Coeff239Pu_mueller[3] * energy ** 3 +
                       Coeff239Pu_mueller[4] * energy ** 4 + Coeff239Pu_mueller[5] * energy ** 5)
Pu241_mueller = np.exp(Coeff241Pu_mueller[0] + Coeff241Pu_mueller[1] * energy +
                       Coeff241Pu_mueller[2] * energy ** 2 + Coeff241Pu_mueller[3] * energy ** 3 +
                       Coeff241Pu_mueller[4] * energy ** 4 + Coeff241Pu_mueller[5] * energy ** 5)
# add the weighted sum of the terms:
spectrum_mueller = Fraction235U*U235_mueller + Fraction238U*U238_mueller + Fraction239Pu*Pu239_mueller + \
                   Fraction241Pu*Pu241_mueller
# There are 3.125*10**19 fissions/GW/second, spectrum in units of electron-antineutrino/(MeV * GW * s):
spectrum_mueller1 = spectrum_mueller * 3.125*10**19
# There are about 3.156*10**7 seconds in a year, spectrum in units of electron-antineutrino/(MeV * GW * year):
spectrum_mueller_total = spectrum_mueller1 * 3.156*10**7
# electron-antineutrino flux in units of electron-antineutrino/(MeV * year):
flux_mueller = spectrum_mueller_total * power_th


""" Flux of reactor electron-antineutrinos calculated with the fit described in Huber2011_PhysRevC.84.024617: """
# coefficients of the polynomial fit of order 5 (table III) (238U is NOT considered):
Coeff235U_huber = np.array([4.367, -4.577, 2.100, -5.294*10**(-1), 6.186*10**(-2), -2.777*10**(-3)])
Coeff239Pu_huber = np.array([4.757, -5.392, 2.563, -6.596*10**(-1), 7.820*10**(-2), -3.536*10**(-3)])
Coeff241Pu_huber = np.array([2.990, -2.882, 1.278, -3.343*10**(-1), 3.905*10**(-2), -1.754*10**(-3)])
# fit to the electron-antineutrino spectrum, in units of electron-antineutrinos/(MeV * fission)
# (fit function equ. (23) of Huber2011_PhysRevC.84.024617):
U235_huber = np.exp(Coeff235U_huber[0] + Coeff235U_huber[1] * energy +
                    Coeff235U_huber[2] * energy ** 2 + Coeff235U_huber[3] * energy ** 3 +
                    Coeff235U_huber[4] * energy ** 4 + Coeff235U_huber[5] * energy ** 5)
Pu239_huber = np.exp(Coeff239Pu_huber[0] + Coeff239Pu_huber[1] * energy +
                     Coeff239Pu_huber[2] * energy ** 2 + Coeff239Pu_huber[3] * energy ** 3 +
                     Coeff239Pu_huber[4] * energy ** 4 + Coeff239Pu_huber[5] * energy ** 5)
Pu241_huber = np.exp(Coeff241Pu_huber[0] + Coeff241Pu_huber[1] * energy +
                     Coeff241Pu_huber[2] * energy ** 2 + Coeff241Pu_huber[3] * energy ** 3 +
                     Coeff241Pu_huber[4] * energy ** 4 + Coeff241Pu_huber[5] * energy ** 5)
# add the weighted sum of the terms:
spectrum_huber = Fraction235U*U235_huber + Fraction239Pu*Pu239_huber + Fraction241Pu*Pu241_huber
# There are 3.125*10**19 fissions/GW/second, spectrum in units of electron-antineutrino/(MeV * GW * s):
spectrum_huber1 = spectrum_huber * 3.125*10**19
# There are about 3.156*10**7 seconds in a year, spectrum in units of electron-antineutrino/(MeV * GW * year):
spectrum_huber_total = spectrum_huber1 * 3.156*10**7
# electron-antineutrino flux in units of electron-antineutrino/(MeV * year):
flux_huber = spectrum_huber_total * power_th


""" Flux of reactor electron-antineutrinos calculated with the data from figure 1 of paper 
    Fallot2012_PhysRevLett.109.202504 (data digitized with Engauge Digitizer): """
# corresponding electron-antineutrino energy for U235 in MeV:
energy_fallot_U235 = np.array([0.3841, 0.7464, 1.1087, 1.471, 1.7609, 2.1232, 2.4855, 2.8478, 3.1377, 3.5, 3.8623,
                               4.1522, 4.5145, 4.8043, 5.0942, 5.3841, 5.7464, 6.0362, 6.3261, 6.6159, 6.9058, 7.1957,
                               7.4855, 7.7754, 7.9928, 8.1377, 8.3551, 8.6449, 8.8623, 9.1522, 9.442, 9.7319, 9.8768,
                               10.1667, 10.529, 10.8188, 11.1087, 11.3986, 11.6884, 11.8333, 12.0507, 12.1232, 12.1957,
                               12.2681, 12.3406, 12.4855, 12.7754, 13.1377, 13.3551, 13.5725, 13.9348, 14.2246, 14.442,
                               14.7319, 14.9493, 15.0217, 15.0942, 15.1667, 15.3841, 15.6739, 15.8913, 16.1087])
# electron-antineutrino flux from figure 1 for U235 in antineutrinos/(MeV*fission):
flux_fallot_U235 = np.array([1.548, 1.797, 1.797, 1.548, 1.334, 1.149, 0.852, 0.734, 0.5446, 0.4041, 0.3481, 0.2224,
                             0.165, 0.1225, 0.0909, 0.06741, 0.05001, 0.03196, 0.02371, 0.01759, 0.01124, 0.007186,
                             0.004592, 0.002528, 0.001392, 0.000766, 0.0003632, 0.0002321, 0.0001484, 0.0000948,
                             0.00006059, 0.00003872, 0.00002132, 0.00001362, 0.00001011, 0.000006459, 0.000004792,
                             0.000003063, 0.000001686, 0.000000928, 0.0000005931, 0.0000002812, 0.0000001334,
                             0.00000006323, 0.00000002998, 0.0000000165, 0.00000001225, 0.00000000783, 0.000000005001,
                             0.000000003196, 0.000000002043, 0.000000001515, 0.000000000969, 0.000000000619,
                             0.0000000003407, 0.0000000001616, 0.00000000003632, 0.00000000001722, 0.00000000001101,
                             0.00000000000817, 0.000000000004495, 0.000000000002475])

# corresponding electron-antineutrino energy for U238 in MeV:
energy_fallot_U238 = np.array([0.2391, 0.6014, 0.9638, 1.3261, 1.6159, 1.9783, 2.3406, 2.7029, 2.9928, 3.3551, 3.7174,
                               4.0797, 4.3696, 4.7319, 5.0217, 5.3841, 5.6739, 6.0362, 6.3261, 6.6159, 6.9783, 7.2681,
                               7.558, 7.8478, 8.1377, 8.3551, 8.6449, 8.9348, 9.2246, 9.5145, 9.8043, 10.0942, 10.3841,
                               10.7464, 11.0362, 11.3261, 11.5435, 11.7609, 11.9783, 12.1957, 12.2681, 12.3406, 12.413,
                               12.7754, 13.0652, 13.3551, 13.7174, 14.0072, 14.2971, 14.587, 14.8768, 15.0217, 15.0942,
                               15.1667, 15.2391, 15.4565, 15.7464, 16.0362, 16.1812])
# electron-antineutrino flux from figure 1 for U238 in electron-antineutrinos/(MeV*fission):
flux_fallot_U238 = np.array([2.422, 2.812, 2.812, 2.087, 1.797, 1.548, 1.149, 0.989, 0.852, 0.734, 0.5446, 0.4041,
                             0.3481, 0.2582, 0.165, 0.1422, 0.0909, 0.06741, 0.04308, 0.03196, 0.02371, 0.01515,
                             0.00969, 0.00619, 0.003956, 0.002178, 0.001392, 0.000889, 0.0005684, 0.0003632, 0.0002695,
                             0.0001722, 0.0001278, 0.0000817, 0.00006059, 0.00003872, 0.00002475, 0.00001582,
                             0.00000871, 0.000004792, 0.000002272, 0.000001077, 0.0000005109, 0.0000003791,
                             0.0000003265, 0.0000002812, 0.0000002087, 0.0000001334, 0.0000000989, 0.00000006323,
                             0.00000004041, 0.00000002224, 0.00000001055, 0.000000001124, 0.0000000005332,
                             0.0000000001616, 0.0000000001032, 0.00000000006598, 0.00000000003632])

# corresponding electron-antineutrino energy for Pu239 in MeV:
energy_fallot_Pu239 = np.array([0.3841, 0.7464, 1.0362, 1.3986, 1.7609, 2.0507, 2.413, 2.7029, 3.0652, 3.4275, 3.7174,
                                4.0072, 4.3696, 4.6594, 5.0217, 5.3116, 5.6014, 5.8913, 6.2536, 6.5435, 6.8333, 7.1232,
                                7.413, 7.7029, 7.9203, 8.1377, 8.2101, 8.3551, 8.5725, 8.8623, 9.1522, 9.442, 9.6594,
                                9.9493, 10.1667, 10.4565, 10.7464, 11.0362, 11.3986, 11.6159, 11.7609, 11.9783,
                                12.1957, 12.3406, 12.4855, 12.7754, 13.0652, 13.2826, 13.4275, 13.5725, 13.6449,
                                13.7174, 14.0072, 14.2971, 14.5145, 14.8043, 15.0217, 15.0942, 15.1667, 15.2391,
                                15.4565, 15.7464, 16.0362, 16.1812])
# electron-antineutrino flux from figure 1 for Pu239 in electron-antineutrinos/(MeV*fission):
flux_fallot_Pu239 = np.array([1.696, 2.309, 1.979, 1.453, 1.453, 1.067, 0.784, 0.5758, 0.4935, 0.3625, 0.2662, 0.1955,
                              0.1436, 0.1055, 0.0775, 0.04877, 0.0307, 0.02255, 0.01656, 0.01216, 0.00766, 0.004819,
                              0.003034, 0.00191, 0.00103, 0.0005557, 0.0002569, 0.0001386, 0.0000872, 0.00004706,
                              0.00002962, 0.00001865, 0.00001174, 0.00000739, 0.000003986, 0.000002927, 0.00000215,
                              0.000001579, 0.000000994, 0.0000005363, 0.0000002893, 0.0000001821, 0.0000000982,
                              0.00000004542, 0.0000000245, 0.00000001542, 0.00000000971, 0.000000006111,
                              0.000000002825, 0.000000001524, 0.000000000705, 0.0000000003258, 0.0000000002393,
                              0.0000000001506, 0.0000000000948, 0.00000000005967, 0.00000000003219, 0.00000000001488,
                              0.000000000006881, 0.000000000003181, 0.000000000002336, 0.000000000001471,
                              0.000000000000793, 0.000000000000428])

# corresponding electron-antineutrino energy for Pu241 in MeV:
energy_fallot_Pu241 = np.array([0.3841, 0.7464, 1.0362, 1.3986, 1.7609, 2.1232, 2.413, 2.7754, 3.0652, 3.4275, 3.7899,
                                4.0797, 4.442, 4.7319, 5.0217, 5.3841, 5.6739, 5.9638, 6.3261, 6.6159, 6.9058, 7.1957,
                                7.4855, 7.7029, 7.9928, 8.1377, 8.3551, 8.5, 8.7899, 9.0797, 9.3696, 9.587, 9.8768,
                                10.0942, 10.4565, 10.7464, 11.0362, 11.3261, 11.6159, 11.7609, 11.9783, 12.1232,
                                12.1957, 12.3406, 12.413, 12.4855, 12.8478, 13.2101, 13.5, 13.7899, 14.0797, 14.442,
                                14.6594, 14.9493, 15.0217, 15.0942, 15.1667, 15.3841, 15.6739, 15.9638, 16.1087])
# electron-antineutrino flux from figure 1 for Pu241 in electron-antineutrinos/(MeV*fission):
flux_fallot_Pu241 = np.array([1.98, 2.695, 2.31, 1.697, 1.454, 1.247, 0.916, 0.785, 0.5765, 0.4234, 0.3629, 0.2666,
                              0.1958, 0.1438, 0.1057, 0.06652, 0.04886, 0.03076, 0.0226, 0.0166, 0.01045, 0.006579,
                              0.004142, 0.002235, 0.001407, 0.000759, 0.0003511, 0.0001895, 0.0001193, 0.0000751,
                              0.00005516, 0.00002976, 0.00001874, 0.00001011, 0.00000743, 0.000005456, 0.000004007,
                              0.000002523, 0.000001361, 0.000000735, 0.0000003397, 0.0000001833, 0.0000000848,
                              0.0000000392, 0.00000002115, 0.00000000978, 0.00000000838, 0.00000000719, 0.000000005278,
                              0.000000003877, 0.000000002848, 0.000000002092, 0.000000001317, 0.000000000829,
                              0.0000000001773, 0.00000000001754, 0.000000000003751, 0.000000000002362,
                              0.000000000001735, 0.000000000000936, 0.0000000000005051])

# linear interpolation of the data of the fluxes with respect to energy:
U235_fallot = np.interp(energy, energy_fallot_U235, flux_fallot_U235)
U238_fallot = np.interp(energy, energy_fallot_U238, flux_fallot_U238)
Pu239_fallot = np.interp(energy, energy_fallot_Pu239, flux_fallot_Pu239)
Pu241_fallot = np.interp(energy, energy_fallot_Pu241, flux_fallot_Pu241)
# add the weighted sum of the terms:
spectrum_fallot = Fraction235U*U235_fallot + Fraction238U*U238_fallot + Fraction239Pu*Pu239_fallot + \
                  Fraction241Pu*Pu241_fallot
# calculate weighted sum for Fallot data (not interpolated):
# spectrum_fallot_data = (Fraction235U*flux_fallot_U235 + Fraction238U*flux_fallot_U238 +
#                         Fraction239Pu*flux_fallot_Pu239 + Fraction241Pu*flux_fallot_Pu241)
# There are 3.125*10**19 fissions/GW/second, spectrum in units of electron-antineutrino/(MeV * GW * s):
spectrum_fallot1 = spectrum_fallot * 3.125*10**19
# spectrum_fallot1_data = spectrum_fallot_data * 3.125*10**19
# There are about 3.156*10**7 seconds in a year, spectrum in units of electron-antineutrino/(MeV * GW * year):
spectrum_fallot_total = spectrum_fallot1 * 3.156*10**7
# spectrum_fallot_total_data = spectrum_fallot1_data * 3.156*10**7
# electron-antineutrino flux in units of electron-antineutrino/(MeV * year):
flux_fallot = spectrum_fallot_total * power_th
# flux_fallot_data = spectrum_fallot_total_data * power_th

""" Consider Neutrino oscillation for NORMAL HIERARCHY from NuOscillation.cc: """
"""
# Oscillation parameters:
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
P21 = sin2_2th12 * cos4_th13 * np.sin(1.267 * Dm2_21 * L_m / energy)**2
P31 = sin2_2th13 * cos2_th12 * np.sin(1.267 * Dm2_31 * L_m / energy)**2
P32 = sin2_2th13 * sin2_th12 * np.sin(1.267 * Dm2_32 * L_m / energy)**2
# Survival probability of electron-antineutrinos:
Prob = 1. - P21 - P31 - P32
"""


""" Calculate the spectrum at the detector without oscillation: """
# Vogel paper in electron-antineutrino/MeV:
vogel_spectrum = 1/(4*np.pi*L_cm**2) * flux_vogel * sigma_ibd(energy, DELTA, MASS_POSITRON) * \
                 detection_eff * N_target * t_years
# Mueller paper in electron-antineutrino/MeV:
mueller_spectrum = 1/(4*np.pi*L_cm**2) * flux_mueller * sigma_ibd(energy, DELTA, MASS_POSITRON) * \
                   detection_eff * N_target * t_years
# Huber paper in electron-antineutrino/MeV:
huber_spectrum = 1/(4*np.pi*L_cm**2) * flux_huber * sigma_ibd(energy, DELTA, MASS_POSITRON) * detection_eff \
                 * N_target * t_years
# Fallot paper in electron-antineutrino/MeV:
fallot_spectrum = 1/(4*np.pi*L_cm**2) * flux_fallot * sigma_ibd(energy, DELTA, MASS_POSITRON) \
                      * detection_eff * N_target * t_years


# Display the electron-antineutrino fluxes with the settings below:
h1 = pyplot.figure(1)
pyplot.plot(energy, flux_vogel, label='electron-antineutrino flux of Vogel1989')
pyplot.plot(energy, flux_mueller, label='electron-antineutrino flux of Mueller2011')
# pyplot.plot(energy, flux_huber, label='electron-antineutrino flux of Huber2011 (without 238U)')
pyplot.plot(energy, flux_fallot, label='electron-antineutrino flux of Fallot2012')
pyplot.ylim(ymin=0)
# pyplot.xticks(np.arange(4.0, E1[-1]), 2.0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Electron-antineutrino flux in 1/(MeV * year)")
pyplot.title("Electron-antineutrino reactor flux of different paper")
pyplot.legend()
pyplot.grid()

# Display the electron-antineutrino fluxes with the settings below:
h2 = pyplot.figure(2)
pyplot.semilogy(energy, flux_vogel, label='electron-antineutrino flux of Vogel1989')
pyplot.semilogy(energy, flux_mueller, label='electron-antineutrino flux of Mueller2011')
# pyplot.semilogy(energy, flux_huber, label='electron-antineutrino flux of Huber2011 (without 238U)')
pyplot.semilogy(energy, flux_fallot, "-x", label='electron-antineutrino flux of Fallot2012')
pyplot.ylim(ymin=0)
# pyplot.xticks(np.arange(4.0, E1[-1]), 2.0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Electron-antineutrino flux in 1/(MeV * year)")
pyplot.title("Electron-antineutrino reactor flux of different paper")
pyplot.legend()
pyplot.grid()

# Display the electron-antineutrino spectrum at the detector with the setting below:
h3 = pyplot.figure(3)
pyplot.plot(energy, vogel_spectrum, label='spectrum of Vogel1989')
pyplot.plot(energy, mueller_spectrum, label='spectrum of Mueller2011')
# pyplot.plot(energy, huber_spectrum, label='spectrum of Huber2011 (without 238U)')
pyplot.plot(energy, fallot_spectrum, label='spectrum of Fallot2012')
pyplot.ylim(ymin=0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Electron-antineutrino spectrum in 1/(MeV)")
pyplot.title("Electron-antineutrino spectrum at JUNO detector for different paper after {0:d} years".format(t_years))
pyplot.legend()
pyplot.grid()

# Display the electron-antineutrino spectrum at the detector with the setting below:
h4 = pyplot.figure(4)
pyplot.semilogy(energy, vogel_spectrum, label='spectrum of Vogel1989')
pyplot.semilogy(energy, mueller_spectrum, label='spectrum of Mueller2011')
# pyplot.semilogy(energy, huber_spectrum, label='spectrum of Huber2011 (without 238U)')
pyplot.semilogy(energy, fallot_spectrum, label='spectrum of Fallot2012')
pyplot.ylim(ymin=0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Electron-antineutrino spectrum in 1/(MeV)")
pyplot.title("Electron-antineutrino spectrum at JUNO detector for different paper after {0:d} years".format(t_years))
pyplot.legend()
pyplot.grid()

h5 = pyplot.figure(5)
pyplot.plot(energy_fallot_U235, flux_fallot_U235, label="real data")
pyplot.plot(energy, U235_fallot, label="interpolated")
pyplot.legend()
pyplot.grid()

pyplot.show()
