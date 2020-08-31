""" script to display the reactor background spectrum:

    The reactor flux is taken from function reactor background_v3() of script gen_spectrum_functions.py

    Data of the flux is taken from paper Fallot2012_PhysRevLett.109.202504.pdf




"""
import numpy as np
from matplotlib import pyplot as plt
from gen_spectrum_functions import sigma_ibd

# set neutrino energy in MeV:
E_nu_min = 1.90
E_nu_max = 16
E_nu_interval = 0.05
energy_neutrino = np.arange(E_nu_min, E_nu_max+E_nu_interval, E_nu_interval)

# mass of positron in MeV (reference PDG 2016) (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (reference PDG 2016) (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (reference PDG 2016) (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON

# calculate IBD cross-section in cm^2:
crosssection = sigma_ibd(energy_neutrino, DELTA, MASS_POSITRON)

# set number of free protons:
n_target = 1.45 * 10**33

# total exposure time in years:
t_years = 10
# total exposure time in seconds:
t = t_years * 3.156 * 10 ** 7

# set IBD detection efficiency:
detection_efficiency = 0.67005

# muon veto cut efficiency:
exposure_ratio_muon = 0.9717


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

""" Consider neutrino oscillation from NuOscillation.cc: """
# Oscillation parameters:
# distance reactor to detector in meter (float):
l_m = 5.25 * 10 ** 4
# distance reactor to detector in centimeter (float):
l_cm = l_m * 100
# mixing angles from PDG 2016 (same in NuOscillation.cc) (float):
sin2_th12 = 0.297
# mass squared differences in eV**2 from PDG 2016 (same in NuOscillation.cc) (float):
dm2_21 = 7.37 * 10 ** (-5)

""" NORMAL ORDERING: """
# mixing angles from PDG 2016 (same in NuOscillation.cc) (float):
sin2_th13_NH = 0.0214
# mass squared differences in eV**2 from PDG 2016 (same in NuOscillation.cc) (float):
dm2_NH = 2.50 * 10 ** (-3)
dm2_31_NH = dm2_NH + dm2_21 / 2.0
dm2_32_NH = dm2_31_NH - dm2_21
# calculate the other parameters (float):
cos2_th12 = 1. - sin2_th12
sin2_2th12 = 4. * sin2_th12 * cos2_th12
cos2_th13_NH = 1. - sin2_th13_NH
sin2_2th13_NH = 4. * sin2_th13_NH * cos2_th13_NH
cos4_th13_NH = cos2_th13_NH ** 2
# With these parameters calculate survival probability of electron-antineutrinos (np.array of float):
p21_NH = sin2_2th12 * cos4_th13_NH * np.sin(1.267 * dm2_21 * l_m / energy_neutrino) ** 2
p31_NH = sin2_2th13_NH * cos2_th12 * np.sin(1.267 * dm2_31_NH * l_m / energy_neutrino) ** 2
p32_NH = sin2_2th13_NH * sin2_th12 * np.sin(1.267 * dm2_32_NH * l_m / energy_neutrino) ** 2
# Survival probability of electron-antineutrinos for Normal Hierarchy (np.array of float):
prob_oscillation_NH = 1. - p21_NH - p31_NH - p32_NH

""" INVERTED ORDERING: """
# mixing angles from PDG 2016 (same in NuOscillation.cc) (float):
sin2_th13_IH = 0.0218
# mass squared differences in eV**2 from PDG 2016 (same in NuOscillation.cc) (float):
dm2_IH = -2.46 * 10 ** (-3)
dm2_32_IH = dm2_IH - dm2_21 / 2.0
dm2_31_IH = dm2_32_IH + dm2_21

# calculate the other parameters (float):
cos2_th12 = 1. - sin2_th12
sin2_2th12 = 4. * sin2_th12 * cos2_th12
cos2_th13_IH = 1. - sin2_th13_IH
sin2_2th13_IH = 4. * sin2_th13_IH * cos2_th13_IH
cos4_th13_IH = cos2_th13_IH ** 2
# With these parameters calculate survival probability of electron-antineutrinos (np.array of float):
p21_IH = sin2_2th12 * cos4_th13_IH * np.sin(1.267 * dm2_21 * l_m / energy_neutrino) ** 2
p31_IH = sin2_2th13_IH * cos2_th12 * np.sin(1.267 * dm2_31_IH * l_m / energy_neutrino) ** 2
p32_IH = sin2_2th13_IH * sin2_th12 * np.sin(1.267 * dm2_32_IH * l_m / energy_neutrino) ** 2
# Survival probability of electron-antineutrinos for Normal Hierarchy (np.array of float):
prob_oscillation_IH = 1. - p21_IH - p31_IH - p32_IH

""" Theoretical reactor electron-antineutrino spectrum in JUNO with oscillation """
# Theoretical spectrum in JUNO for normal hierarchy with oscillation in units of electron-antineutrinos/MeV
# in "t_years" years (np.array of float):
theo_spectrum_reactor_NH = (1 / (4 * np.pi * l_cm ** 2) * flux_reactor * crosssection * detection_efficiency *
                            n_target * t * prob_oscillation_NH * exposure_ratio_muon)
# Theoretical spectrum in JUNO for inverted hierarchy with oscillation in units of electron-antineutrinos/MeV
# in "t_years" years (np.array of float):
theo_spectrum_reactor_IH = (1 / (4 * np.pi * l_cm ** 2) * flux_reactor * crosssection * detection_efficiency *
                            n_target * t * prob_oscillation_IH * exposure_ratio_muon)

""" display spectra: """
# between E_nu_min and 13 MeV:
# get index corresponding to 13 MeV:
E_nu_1 = E_nu_max
index_E_nu_1 = int((E_nu_1 - E_nu_min) / E_nu_interval)
# take neutrino energy and spectra between 0.5 and 13 MeV:
energy_neutrino_1 = energy_neutrino[0:index_E_nu_1]
spectrum_1_NH = theo_spectrum_reactor_NH[0:index_E_nu_1]
spectrum_1_IH = theo_spectrum_reactor_IH[0:index_E_nu_1]

# between 10 MeV and 50 MeV:
# get index corresponding to 10 MeV:
E_nu_2 = 10.0
index_E_nu_2 = int((E_nu_2 - E_nu_min) / E_nu_interval)
# take neutrino energy and spectra between 10 and 50 MeV:
energy_neutrino_2 = energy_neutrino[index_E_nu_2:]
spectrum_2_NH = theo_spectrum_reactor_NH[index_E_nu_2:]
spectrum_2_IH = theo_spectrum_reactor_IH[index_E_nu_2:]


h1 = plt.figure(1, figsize=(11, 6))
plt.plot(energy_neutrino_1, spectrum_1_NH, "b--", label="Normal ordering")
plt.plot(energy_neutrino_1, spectrum_1_IH, "b-", label="Inverted ordering")
plt.axis([E_nu_min, E_nu_max, 0, 1.25*max(spectrum_1_NH)])
plt.xlabel("Neutrino energy $E_{\\nu}$ in MeV", fontsize=12)
plt.ylabel("$\\bar{\\nu}_e$ events per MeV", fontsize=12)
plt.title("Reactor $\\bar{\\nu}_e$ energy spectrum in JUNO after 10 years\n"
          "(for normal and inverted neutrino mass ordering)")
plt.grid()
plt.legend()

# this is an inset axes over the main axes
a = plt.axes([.51, .35, .38, .38])
plt.plot(energy_neutrino_2, spectrum_2_NH, "b-")
plt.fill_between(energy_neutrino_2, np.zeros(len(spectrum_2_NH)), spectrum_2_NH, color="b", alpha=0.6,
                 label="reactor $\\bar{\\nu}_e$ spectrum")
plt.vlines(15.0, 0.0, 1.27, color="r", linewidth=2,
           label="$\\bar{\\nu}_e$ signal for DM with\n$m_{DM}=15$ MeV ($N_S=1.27$)")
plt.xlim(xmin=E_nu_2, xmax=E_nu_max)
plt.ylim(ymin=0, ymax=10)
plt.xlabel("Neutrino energy $E_{\\nu}$ in MeV", fontsize=10)
plt.ylabel("$\\bar{\\nu}_e$ events per MeV", fontsize=10)
plt.grid()
plt.legend(fontsize=10)

plt.show()










