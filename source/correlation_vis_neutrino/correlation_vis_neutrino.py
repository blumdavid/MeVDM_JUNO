""" Script to compare the different correlations between E_visible and E_neutrino:
    Calculations are based on the paper of Vogel and Beacom: 'Angular distribution of neutron inverse beta decay'
    in Physical Review D, vol. 60, 053003.
    Also KRLInverseBeta.cc and KRLInverseBeta.hh from JUNO offline generator InverseBeta is based on this paper.
    The third approach of calculating the visible energy is based on paper of Strumia and Vissani:
    'Precise quasielastic neutrino/nucleon cross section', 0302055_IBDcrosssection.pdf
    """

import numpy as np
from matplotlib import pyplot

# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536
# average nucleon mass in MeV (float):
MASS_AVERAGE = (MASS_PROTON + MASS_NEUTRON)/2
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON

# Neutrino energy in MeV:
E_neutrino = np.arange(5, 130, 5)
# E_neutrino = 20
theta = np.pi
# theta = np.arange(0, np.pi, 0.1)
cos_theta = np.cos(theta)

# Simple approximation for low neutrino energies: (described in paper of Vogel/Beacom and Strumia/Vissani):
# At zeroth order of 1/M:
# total positron energy in MeV:
E0_positron_simple = E_neutrino - DELTA
# positron momentum in MeV:
p0_positron = np.sqrt(E0_positron_simple ** 2 - MASS_POSITRON ** 2)
# positron velocity in units of c=1:
v0_positron = p0_positron / E0_positron_simple
# prompt visible energy in the detector in MeV:
E0_visible_simple = E0_positron_simple + MASS_POSITRON


# Calculation described in Vogel/Beacom paper: in first order of 1/M (depend on cos(theta)):
# y**2-value from equ. 11:
ysquare = (DELTA**2 - MASS_POSITRON**2) / 2
# total positron energy in MeV:
E1_positron_Vogel = E0_positron_simple * (1 - E_neutrino / MASS_AVERAGE * (1 - v0_positron * cos_theta)) \
                    - ysquare / MASS_AVERAGE
# prompt visible energy in the detector in MeV:
E1_visible_Vogel = E1_positron_Vogel + MASS_POSITRON


# Calculation described in Vogel/Beacom paper: in first order of 1/M:
# BUT: independent of cos(theta) -> integration of E1_positron_Vogel over theta from 0 to 2*pi divided by 2*pi!!!
# Total positron energy in MeV:
E1_positron_Vogel_average = E0_positron_simple - E0_positron_simple*E_neutrino/MASS_AVERAGE - ysquare/MASS_AVERAGE
# prompt visible energy in the detector in MeV:
E1_visible_Vogel_average = E1_positron_Vogel_average + MASS_POSITRON

# Calculation described in Strumia/Vissani: large part of the effect due to the recoil of the nucleon
# useful term, page 3:
s = 2 * MASS_PROTON * E_neutrino + MASS_PROTON**2
# neutrino energy in center of mass (CM) frame in MeV, page 4, equ. 13:
E_neutrino_CM = (s - MASS_PROTON**2) / (2 * np.sqrt(s))
# positron energy in CM frame in MeV, page 4, equ. 13:
E_positron_CM = (s - MASS_NEUTRON**2 + MASS_POSITRON**2) / (2 * np.sqrt(s))
# useful term, page 3, equ. 12:
delta = (MASS_NEUTRON**2 - MASS_PROTON**2 - MASS_POSITRON**2) / (2 * MASS_PROTON)
# Average lepton energy in MeV, which can be approximated (at better than 1 percent below ~ 100 MeV) by
# (page 5, equ. 16):
E_positron_Strumia = E_neutrino - delta - E_neutrino_CM * E_positron_CM / MASS_PROTON
# prompt visible energy in the detector in MeV:
E_visible_Strumia = E_positron_Strumia + MASS_POSITRON


h1 = pyplot.figure(1)
pyplot.plot(E_neutrino, E1_visible_Vogel, 'r-.', label='Vogel (first order in 1/M, depend on cos(theta))')
pyplot.plot(E_neutrino, E1_visible_Vogel_average, 'b:', label='Vogel (first order in 1/M, integrated over theta)')
pyplot.plot(E_neutrino, E0_visible_simple, 'b--', label='simple approach')
pyplot.plot(E_neutrino, E_visible_Strumia, 'g', label='Strumia (independent of cos(theta))')
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("prompt visible energy in MeV")
pyplot.title("Prompt visible energy for calculations of different papers for theta = {0:.2f}".format(theta))
pyplot.legend()
pyplot.grid()

"""
h2 = pyplot.figure(2)
pyplot.plot(E_neutrino, E1_visible_Vogel / E0_visible_simple, 'r')
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("visible energy 1st order / visible energy 0th order")
pyplot.title("Ratio of prompt visible energy in first to zeroth order in 1/M for theta = {0:.1f}".format(theta))
pyplot.grid()
"""
"""
h3 = pyplot.figure(1)
pyplot.plot(cos_theta, E1_visible_Vogel, 'r', label="first order")
pyplot.plot(cos_theta, E0_visible_simple, 'b', label="zeroth order")
pyplot.xlabel("cos(theta)")
pyplot.ylabel("prompt visible energy in MeV")
pyplot.title("visible energy in first and zeroth order of 1/M for E_neutrino = {0:.2f}MeV".format(E_neutrino))
pyplot.legend()
pyplot.grid()
"""

pyplot.show()
