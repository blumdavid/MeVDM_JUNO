""" script to estimate the event rate of neutrinos from DM annihilation in the Milky Way for the THEIA experiment:


"""
import numpy as np
from matplotlib import pyplot as plt
from gen_spectrum_functions import sigma_ibd

# mass of positron in MeV (reference PDG 2016) (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (reference PDG 2016) (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (reference PDG 2016) (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON

# set DM masses in MeV:
mass_dm = np.arange(20, 110, 10)

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

# calculate IBD cross-section in cm**2:
cross_section_IBD = sigma_ibd(mass_dm, DELTA, MASS_POSITRON)

# number of free protons in THEIA per kton:
number_per_kt = 6.73 * 10**31
# mass of THEIA_25 in ktons:
mass_Theia25 = 25
# mass of THEIA_100 in ktons:
mass_Theia100 = 100
# total number of protons for Theia_25:
number_Theia25 = number_per_kt * mass_Theia25
# total number of protons for Theia_100:
number_Theia100 = number_per_kt * mass_Theia100

# exposure time in years:
time_years = 10.0
# exposure time in seconds:
time = time_years * 3.156 * 10 ** 7

# calculate number of events per MeV for Theia_25:
number_events_theia25 = phi_signal * cross_section_IBD * number_Theia25 * time

# calculate number of events per MeV for Theia_100:
number_events_theia100 = phi_signal * cross_section_IBD * number_Theia100 * time


print(number_events_theia25)

print(number_events_theia100)





