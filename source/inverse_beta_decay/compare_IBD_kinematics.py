""" compare IBD kinematics of different papers:

    1. Vogel, Beacom: 'Angular distribution of neutron inverse beta decay', Phys.Rev D, v60, 053003

    2. Strumia, Vissani: 'Precise quasielastic neutrino/nucleon cross-section', PhysLett B 564 (2003) 42–54



"""
import numpy as np
from matplotlib import pyplot as plt
from gen_spectrum_functions import energy_positron_1

# neutrino energy in MeV:
E_nu = np.arange(10, 100.5, 0.5)
# cosine of theta:
cos_theta_1 = 0
cos_theta_2 = -1
cos_theta_3 = 1

""" Natural constants: """
# velocity of light in vacuum, in cm/s (reference PDG 2016) (float constant):
C_LIGHT = 2.998 * 10 ** 10
# mass of positron in MeV (reference PDG 2016) (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (reference PDG 2016) (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (reference PDG 2016) (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON

""" paper of Vogel: """
# positron energy in MeV at zeroth order of 1/M_proton:
E_pos_0 = E_nu - DELTA
# calculate positron momentum in MeV:
p_pos_0 = np.sqrt(E_pos_0**2 - MASS_POSITRON**2)
# calculate positron velocity in MeV:
v_pos_0 = p_pos_0 / E_pos_0

# positron energy in MeV at first order of 1/M_proton (equation 11:
E_pos_Vogel_1 = energy_positron_1(cos_theta_1, E_pos_0, E_nu, v_pos_0, MASS_PROTON, DELTA, MASS_POSITRON)
E_pos_Vogel_2 = energy_positron_1(cos_theta_2, E_pos_0, E_nu, v_pos_0, MASS_PROTON, DELTA, MASS_POSITRON)
E_pos_Vogel_3 = energy_positron_1(cos_theta_3, E_pos_0, E_nu, v_pos_0, MASS_PROTON, DELTA, MASS_POSITRON)

""" paper of Sturmia: """
# epsilon:
epsilon = E_nu / MASS_PROTON
# delta:
delta = (MASS_NEUTRON**2 - MASS_PROTON**2 - MASS_POSITRON**2) / (2 * MASS_PROTON)
# kappa:
kappa_1 = (1 + epsilon)**2 - (epsilon * cos_theta_1)**2
kappa_2 = (1 + epsilon)**2 - (epsilon * cos_theta_2)**2
kappa_3 = (1 + epsilon)**2 - (epsilon * cos_theta_3)**2

# positron energy in MeV (equation 21):
E_pos_Strumia_1 = ((E_nu - delta) * (1 + epsilon) + epsilon * cos_theta_1 *
                   np.sqrt((E_nu - delta)**2 - MASS_POSITRON**2 * kappa_1)) / kappa_1
E_pos_Strumia_2 = ((E_nu - delta) * (1 + epsilon) + epsilon * cos_theta_2 *
                   np.sqrt((E_nu - delta)**2 - MASS_POSITRON**2 * kappa_2)) / kappa_2
E_pos_Strumia_3 = ((E_nu - delta) * (1 + epsilon) + epsilon * cos_theta_3 *
                   np.sqrt((E_nu - delta)**2 - MASS_POSITRON**2 * kappa_3)) / kappa_3

plt.plot(E_nu, E_pos_Vogel_1, "r-", label="Vogel, cos(theta) = 0")
plt.plot(E_nu, E_pos_Strumia_1, "b-", label="Strumia, cos(theta) = 0")
plt.plot(E_nu, E_pos_Vogel_2, "r:", label="Vogel, cos(theta) = -1")
plt.plot(E_nu, E_pos_Strumia_2, "b:", label="Strumia, cos(theta) = -1")
plt.plot(E_nu, E_pos_Vogel_3, "r--", label="Vogel, cos(theta) = 1")
plt.plot(E_nu, E_pos_Strumia_3, "b--", label="Strumia, cos(theta) = 1")
plt.xlabel("Neutrino energy in MeV")
plt.ylabel("Positron energy in MeV")
plt.legend()
plt.grid()
plt.show()






























