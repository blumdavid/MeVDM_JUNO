""" script to display the atmospheric CC background of electron neutrinos and anti-neutrinos interacting with C12:

    The different cross-sections from Yoshida_2008_Reference2_of_Kim_2009.pdf.

    nu_e_bar + p -> positron + n (as reference)

    nu_e + C12 -> electron + n + Y

    nu_e_bar + C12 -> positron + n + X

    Channels with 2 neutrons and where cross-section is calculated in the paper:

    nu_e + C12 -> electron + 2n + (p + C9,  2p + B8,    3p + Be7,   4p + Li6)

    nu_e_bar + C12 -> positron + 2n + (p + Be9,     2p + Li8)


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

""" get cross-section from xsec_C12_Yoshida_2008.ods 
    (data taken from https://iopscience.iop.org/article/10.1086/591266/fulltext/74086.tables.html): """
# neutrino energy in MeV:
energy_neurino = np.arange(10, 101, 1)

# path, where data is stored:
path_input = "/home/astro/blum/juno/atmoNC/other_atmo_background/"

# data of nu_e + C12:
data_nue_C12 = np.loadtxt(path_input + "xsec_C12_Yoshida_nue.txt")

# data of nu_e_bar + C12:
data_nuebar_C12 = np.loadtxt(path_input + "xsec_C12_Yoshida_nuebar.txt")

# cross-section of nu_e + C12 -> electron + n + Y in cm**2 corresponding to energy_neutrino:
xsec_nue_C12_electron_n_Y = 10**(-42) * data_nue_C12[:, 0]

# cross-section of nu_e + C12 -> electron + 2n + p + C9 in cm**2 corresponding to energy_neutrino:
xsec_nue_C12_electron_2n_p_C9 = 10**(-42) * data_nue_C12[:, 1]

# cross-section of nu_e + C12 -> electron + 2n + 2p + B8 in cm**2 corresponding to energy_neutrino:
xsec_nue_C12_electron_2n_2p_B8 = 10**(-42) * data_nue_C12[:, 2]

# cross-section of nu_e + C12 -> electron + 2n + 3p + Be7 in cm**2 corresponding to energy_neutrino:
xsec_nue_C12_electron_2n_3p_Be7 = 10**(-42) * data_nue_C12[:, 3]

# cross-section of nu_e + C12 -> electron + 2n + 4p + Li6 in cm**2 corresponding to energy_neutrino:
xsec_nue_C12_electron_2n_4p_Li6 = 10**(-42) * data_nue_C12[:, 4]

# calculate total cross-section nu_e + C12 -> electron + 2n + Z in cm**2:
xsec_nue_C12_electron_2n_Z = xsec_nue_C12_electron_2n_p_C9 + xsec_nue_C12_electron_2n_2p_B8 + \
                             xsec_nue_C12_electron_2n_3p_Be7 + xsec_nue_C12_electron_2n_4p_Li6

# cross-section of nu_e_bar + C12 -> positron + n + X in cm**2 corresponding to energy_neutrino:
xsec_nuebar_C12_positron_n_X = 10**(-42) * data_nuebar_C12[:, 0]

# cross-section of nu_e_bar + C12 -> positron + 2n + B10 in cm**2 corresponding to energy_neutrino:
xsec_nuebar_C12_positron_2n_B10 = 10**(-42) * data_nuebar_C12[:, 1]

# cross-section of nu_e_bar + C12 -> positron + 2n + p + Be9 in cm**2 corresponding to energy_neutrino:
xsec_nuebar_C12_positron_2n_p_Be9 = 10**(-42) * data_nuebar_C12[:, 2]

# cross-section of nu_e_bar + C12 -> positron + 2n + 2p + Li8 in cm**2 corresponding to energy_neutrino:
xsec_nuebar_C12_positron_2n_2p_Li8 = 10**(-42) * data_nuebar_C12[:, 3]

# calculate total cross-section nu_e_bar + C12 -> positron + 2n + W in cm**2:
xsec_nuebar_C12_positron_2n_W = xsec_nuebar_C12_positron_2n_B10 + xsec_nuebar_C12_positron_2n_p_Be9 + \
                                xsec_nuebar_C12_positron_2n_2p_Li8

# calculate cross-section of IBD (nu_e_bar + p -> positron + n) in cm**2:
xsec_IBD = sigma_ibd(energy_neurino, DELTA, MASS_POSITRON)

# integrate the cross-sections from 10 to 100 MeV to get total cross-section:
total_xsec_IBD = np.trapz(xsec_IBD, energy_neurino)
total_xsec_nue_C12_electron_n_Y = np.trapz(xsec_nue_C12_electron_n_Y, energy_neurino)
total_xsec_nue_C12_electron_2n_Z = np.trapz(xsec_nue_C12_electron_2n_Z, energy_neurino)
total_xsec_nuebar_C12_positron_n_X = np.trapz(xsec_nuebar_C12_positron_n_X, energy_neurino)
total_xsec_nuebar_C12_positron_2n_W = np.trapz(xsec_nuebar_C12_positron_2n_W, energy_neurino)

print("total cross-section in cm^2 (10 MeV to 100 MeV):")
print("\nIBD:")
print(total_xsec_IBD)
print("\nnu_e + C12 -> electron + n + Y:")
print(total_xsec_nue_C12_electron_n_Y)
print("\nnu_e + C12 -> electron + 2n + Z:")
print(total_xsec_nue_C12_electron_2n_Z)
print("\nnu_e_bar + C12 -> positron + n + X:")
print(total_xsec_nuebar_C12_positron_n_X)
print("\nnu_e_bar + C12 -> positron + 2n + W:")
print(total_xsec_nuebar_C12_positron_2n_W)


plt.figure(1, figsize=(9, 6))
plt.semilogy(energy_neurino, xsec_IBD, "k-", label="$\\bar{\\nu}_e + p \\rightarrow e^+ + n$")
plt.semilogy(energy_neurino, xsec_nue_C12_electron_n_Y, "b-", label="$\\nu_e +^{12}$C $\\rightarrow e^- + n + Y$")
plt.semilogy(energy_neurino, xsec_nue_C12_electron_2n_Z, "b--", label="$\\nu_e +^{12}$C $\\rightarrow e^- + 2n + Z_1$")
plt.semilogy(energy_neurino, xsec_nuebar_C12_positron_n_X, "r-", label="$\\bar{\\nu}_e +^{12}$C "
                                                                       "$\\rightarrow e^+ + n + X$")
plt.semilogy(energy_neurino, xsec_nuebar_C12_positron_2n_W, "r--", label="$\\bar{\\nu}_e +^{12}$C "
                                                                         "$\\rightarrow e^+ + 2n + Z_2$")
plt.xlim(10, 100)
plt.ylim(ymin=10**(-43))
plt.xlabel("Neutrino energy in MeV", fontsize=12)
plt.ylabel("cross-section in cm$^2$", fontsize=12)
plt.title("Neutrino charged current interaction cross-sections of $\\nu_e$ and $\\bar{\\nu}_e$ on $^{12}$C",
          fontsize=14)
plt.grid()
plt.legend()
plt.show()




